"""Robust preprocessing for photographed handwritten math -> MobileNet input.

This is the v3 rewrite of the original Canny/Hough -> 4 corners -> homography
pipeline.  The main changes are driven by failures observed on real photos:

1. Detect a coarse equation region *before* perspective correction.  The equation
   hint tells geometric code which planar surface actually matters, so table grain,
   bezels, styluses, and unrelated long edges are less likely to win.
2. Estimate a smooth writing-surface mask around that equation.  If the visible
   surface is a clean quadrilateral, rectify it directly.  This also handles a
   useful missing-edge case: when the physical page extends outside the photo,
   the visible surface is clipped by the image frame and can still form a valid
   quadrilateral for rectifying the visible portion.
3. Never trust a quadrilateral from geometry alone.  Hough/contour candidates must
   also contain the equation hint and pass plausibility checks.
4. Replace the old texture-sensitive connected-component crop with a multi-cue
   ink detector (local luminance + chroma residual) plus scale-aware grouping.
5. Keep the final raster at height=64 with proportional variable width.
6. Keep a learned-rectifier hook for a future DocTr++-style dense mapper or a
   learned homography regressor.

No deep-learning framework is required by this file; dependencies are only OpenCV
and NumPy.  A learned rectifier may be injected as a callable.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np


Array = np.ndarray
Line = Tuple[float, float, float, float]
BBox = Tuple[int, int, int, int]
LearnedRectifier = Callable[[Array], Union[Array, Tuple[Array, float]]]


# ---------------------------------------------------------------------------
# Configuration / metadata
# ---------------------------------------------------------------------------


@dataclass
class PreprocessConfig:
    target_height: int = 64
    binarize_method: str = "otsu"  # "otsu" or "adaptive"
    crop_padding_ratio: float = 0.14
    crop_min_padding_px: int = 6
    enable_perspective: bool = True
    enable_partial_rectification: bool = True
    enable_surface_guidance: bool = True
    imagenet_normalize: bool = False

    # Perspective safety limits.
    min_quad_area_ratio: float = 0.035
    max_quad_area_ratio: float = 1.60
    max_corner_excursion_ratio: float = 0.55
    max_rectified_scale: float = 1.75

    # Equation localization.  The anchor threshold intentionally suppresses small
    # UI text and fine background texture; small math marks are re-added later.
    equation_anchor_min_height_ratio: float = 0.015
    equation_anchor_max_height_ratio: float = 0.28

    # Surface-mask thresholds.
    surface_min_area_ratio: float = 0.02
    surface_max_area_ratio: float = 0.90
    surface_min_quad_iou: float = 0.78


@dataclass
class PerspectiveInfo:
    method: str
    confidence: float
    homography: Optional[Array] = None
    reason: str = ""


@dataclass
class CropInfo:
    applied: bool
    bbox_xyxy: Optional[BBox] = None
    confidence: float = 0.0
    reason: str = ""


@dataclass
class SurfaceInfo:
    mask: Optional[Array] = None
    quad: Optional[Array] = None
    confidence: float = 0.0
    touches_frame: bool = False
    reason: str = ""


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _line_length(line: Sequence[float]) -> float:
    x1, y1, x2, y2 = line
    return float(math.hypot(x2 - x1, y2 - y1))


def _line_angle(line: Sequence[float]) -> float:
    x1, y1, x2, y2 = line
    return float(math.atan2(y2 - y1, x2 - x1) % math.pi)


def _angle_distance(a: float, b: float) -> float:
    d = abs(a - b) % math.pi
    return min(d, math.pi - d)


def _homogeneous_line(line: Sequence[float]) -> Array:
    x1, y1, x2, y2 = map(float, line)
    l = np.cross(
        np.array([x1, y1, 1.0], dtype=np.float64),
        np.array([x2, y2, 1.0], dtype=np.float64),
    )
    n = math.hypot(float(l[0]), float(l[1]))
    return l if n < 1e-12 else l / n


def _line_intersection(l1: Sequence[float], l2: Sequence[float]) -> Optional[Tuple[float, float]]:
    p = np.cross(_homogeneous_line(l1), _homogeneous_line(l2))
    if abs(float(p[2])) < 1e-9:
        return None
    p = p / p[2]
    if not np.all(np.isfinite(p[:2])):
        return None
    return float(p[0]), float(p[1])


def _order_points(pts: Sequence[Sequence[float]]) -> Array:
    pts = np.asarray(pts, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError("Expected four 2D points")
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    out = np.zeros((4, 2), dtype=np.float32)
    out[0] = pts[np.argmin(s)]
    out[2] = pts[np.argmax(s)]
    out[1] = pts[np.argmin(d)]
    out[3] = pts[np.argmax(d)]
    return out


def _polygon_area(pts: Array) -> float:
    return float(abs(cv2.contourArea(np.asarray(pts, dtype=np.float32))))


def _is_convex_quad(pts: Array) -> bool:
    return bool(cv2.isContourConvex(np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)))


def _bbox_union(a: BBox, b: BBox) -> BBox:
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def _bbox_center(b: BBox) -> Tuple[float, float]:
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def _bbox_area(b: BBox) -> float:
    return float(max(0, b[2] - b[0]) * max(0, b[3] - b[1]))


def _bbox_sample_points(b: BBox) -> List[Tuple[float, float]]:
    x0, y0, x1, y1 = b
    return [
        ((x0 + x1) / 2.0, (y0 + y1) / 2.0),
        (float(x0), float(y0)),
        (float(x1), float(y0)),
        (float(x1), float(y1)),
        (float(x0), float(y1)),
    ]


def _validate_quad(corners: Array, image_shape: Sequence[int], config: PreprocessConfig) -> Tuple[bool, str]:
    h, w = image_shape[:2]
    corners = np.asarray(corners, dtype=np.float32)
    if corners.shape != (4, 2) or not np.all(np.isfinite(corners)):
        return False, "corners are missing/non-finite"
    corners = _order_points(corners)
    if not _is_convex_quad(corners):
        return False, "quadrilateral is not convex"

    area_ratio = _polygon_area(corners) / max(1.0, float(h * w))
    if area_ratio < config.min_quad_area_ratio:
        return False, f"quadrilateral too small (area ratio={area_ratio:.3f})"
    if area_ratio > config.max_quad_area_ratio:
        return False, f"quadrilateral too large (area ratio={area_ratio:.3f})"

    mx = config.max_corner_excursion_ratio * w
    my = config.max_corner_excursion_ratio * h
    if (
        np.any(corners[:, 0] < -mx)
        or np.any(corners[:, 0] > (w - 1) + mx)
        or np.any(corners[:, 1] < -my)
        or np.any(corners[:, 1] > (h - 1) + my)
    ):
        return False, "one or more corners are too far outside the frame"

    tl, tr, br, bl = corners
    widths = (np.linalg.norm(br - bl), np.linalg.norm(tr - tl))
    heights = (np.linalg.norm(tr - br), np.linalg.norm(tl - bl))
    if min(*widths, *heights) < 0.035 * min(h, w):
        return False, "quadrilateral has a degenerate side"
    return True, "ok"


def _quad_equation_score(corners: Array, equation_hint: Optional[BBox]) -> Tuple[bool, float, str]:
    """Semantic gate: the candidate surface must actually contain the equation."""
    if equation_hint is None:
        return True, 0.55, "no equation hint available"

    poly = _order_points(corners).astype(np.float32).reshape(-1, 1, 2)
    tests = [cv2.pointPolygonTest(poly, p, False) >= 0 for p in _bbox_sample_points(equation_hint)]
    inside = sum(bool(v) for v in tests)
    if not tests[0]:
        return False, 0.0, "equation center lies outside quadrilateral"
    if inside < 4:
        return False, 0.0, f"only {inside}/5 equation hint points lie inside quadrilateral"

    qarea = max(1.0, _polygon_area(corners))
    eratio = _bbox_area(equation_hint) / qarea
    if eratio > 0.72:
        return False, 0.0, "quadrilateral is implausibly tight around the equation"
    if eratio < 0.001:
        return False, 0.0, "quadrilateral is implausibly huge relative to the equation"

    # Prefer candidates that surround the equation without spanning the entire scene.
    size_score = float(np.clip(1.0 - abs(math.log10(max(eratio, 1e-6)) - math.log10(0.10)) / 3.0, 0.0, 1.0))
    inside_score = inside / 5.0
    score = 0.45 + 0.30 * inside_score + 0.20 * size_score
    return True, float(min(0.90, score)), f"equation containment {inside}/5, equation/quad area={eratio:.3f}"


def _transform_points(points: Array, H: Array) -> Array:
    pts = np.asarray(points, dtype=np.float32).reshape(1, -1, 2)
    return cv2.perspectiveTransform(pts, np.asarray(H, dtype=np.float64))[0]


def _denominator_is_safe(H: Array, image_shape: Sequence[int]) -> bool:
    h, w = image_shape[:2]
    xs = np.linspace(0.0, w - 1.0, 5)
    ys = np.linspace(0.0, h - 1.0, 5)
    vals = np.asarray(
        [H[2, 0] * x + H[2, 1] * y + H[2, 2] for y in ys for x in xs],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(vals)):
        return False
    if np.min(np.abs(vals)) < 0.08 * max(1e-9, float(np.max(np.abs(vals)))):
        return False
    return bool(np.all(vals > 0) or np.all(vals < 0))


def _warp_white(image: Array, H: Array, output_size: Tuple[int, int]) -> Array:
    out_w, out_h = map(int, output_size)
    border = 255 if image.ndim == 2 else (255, 255, 255)
    return cv2.warpPerspective(
        image,
        np.asarray(H, dtype=np.float64),
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )


def _warp_from_quad(
    image: Array,
    corners: Array,
    output_size: Optional[Tuple[int, int]] = None,
) -> Optional[Tuple[Array, Array]]:
    tl, tr, br, bl = _order_points(corners)
    width = max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))
    height = max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))
    out_w = max(2, int(round(width)))
    out_h = max(2, int(round(height)))
    if output_size is not None:
        out_w, out_h = map(int, output_size)
    if out_w < 10 or out_h < 10:
        return None
    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    H = cv2.getPerspectiveTransform(_order_points(corners), dst)
    if not _denominator_is_safe(H, image.shape):
        return None
    return _warp_white(image, H, (out_w, out_h)), H


def _normalized_pixel_transform(image_shape: Sequence[int]) -> Tuple[Array, Array]:
    h, w = image_shape[:2]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    scale = float(max(h, w))
    to_norm = np.array(
        [[1.0 / scale, 0.0, -cx / scale], [0.0, 1.0 / scale, -cy / scale], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return to_norm, np.linalg.inv(to_norm)


def _fit_homography_to_canvas(H: Array, image_shape: Sequence[int], max_scale: float) -> Optional[Tuple[Array, Tuple[int, int]]]:
    h, w = image_shape[:2]
    if not _denominator_is_safe(H, image_shape):
        return None
    src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    dst = _transform_points(src, H)
    if not np.all(np.isfinite(dst)):
        return None
    mn, mx = dst.min(axis=0), dst.max(axis=0)
    raw_w, raw_h = float(mx[0] - mn[0]), float(mx[1] - mn[1])
    if raw_w < 2 or raw_h < 2:
        return None
    scale = min(1.0, max_scale * w / raw_w, max_scale * h / raw_h)
    if scale <= 0 or not math.isfinite(scale):
        return None
    T = np.array(
        [[scale, 0.0, -mn[0] * scale], [0.0, scale, -mn[1] * scale], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    Hf = T @ H
    out_w = max(2, int(math.ceil(raw_w * scale)))
    out_h = max(2, int(math.ceil(raw_h * scale)))
    if out_w * out_h < 0.12 * w * h:
        return None
    return Hf, (out_w, out_h)


# ---------------------------------------------------------------------------
# Equation localization
# ---------------------------------------------------------------------------


def _ink_likelihood_mask(image: Array) -> Array:
    """Detect dark/chromatic handwriting against its local background.

    We intentionally use a *one-sided* dark residual here instead of absolute local
    contrast.  On real table photos, bright wood-grain ridges and paper borders were
    otherwise promoted to handwriting-scale components.  Colored ink is recovered by
    the Lab chroma residual, but is still required to be at least slightly darker than
    its immediate background.

    If your deployment includes white chalk/white ink on dark boards, add a separate
    bright-ink branch rather than mixing bright texture into this default mask.
    """
    if image.ndim == 2:
        gray_u8 = image
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        bgr = image
        gray_u8 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    h, w = gray_u8.shape[:2]
    gray = gray_u8.astype(np.float32)
    sigma = max(5.0, min(h, w) * 0.012)
    bg_gray = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    dark_resid = np.maximum(bg_gray - gray, 0.0)

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg_lab = cv2.GaussianBlur(lab, (0, 0), sigmaX=sigma, sigmaY=sigma)
    chroma_resid = np.sqrt(np.sum((lab[:, :, 1:] - bg_lab[:, :, 1:]) ** 2, axis=2))

    dark_thr = float(np.clip(np.percentile(dark_resid, 96.0), 14.0, 34.0))
    chroma_thr = float(np.clip(np.percentile(chroma_resid, 97.0), 10.0, 26.0))
    mask = ((dark_resid >= dark_thr) | ((chroma_resid >= chroma_thr) & (dark_resid >= 4.0))).astype(np.uint8) * 255

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return mask


def _is_clean_ink_canvas(image: Array) -> bool:
    """Return true for a rendered/scanned white canvas with sparse dark ink.

    MathWriting-style samples do not contain a photographed planar surface.  Their
    strokes can look like Hough boundary lines, so attempting perspective correction
    is both unsupported and destructive.  This deliberately conservative detector
    only fires when at least 90% of pixels are near white and the image is nearly
    achromatic.
    """
    bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    saturation = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1]
    return bool(
        np.mean(gray >= 245) >= 0.90
        and float(np.percentile(gray, 10)) >= 245.0
        and float(np.percentile(saturation, 90)) <= 8.0
    )


def _clean_canvas_ink_bbox(image: Array) -> Optional[BBox]:
    """Crop all visible ink on a clean canvas; never split one expression by gaps."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    # 250 retains anti-aliased grey pen pixels while excluding the white canvas.
    coords = cv2.findNonZero((gray < 250).astype(np.uint8))
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    return int(x), int(y), int(x + w), int(y + h)


def _collect_ink_components(mask: Array, config: PreprocessConfig) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    h, w = mask.shape[:2]
    image_area = float(h * w)
    num, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)

    small: List[Dict[str, Any]] = []
    anchors: List[Dict[str, Any]] = []
    anchor_min_h = max(9, int(round(config.equation_anchor_min_height_ratio * h)))
    anchor_max_h = max(anchor_min_h + 1, int(round(config.equation_anchor_max_height_ratio * h)))
    min_area = max(6, int(round(0.0000015 * image_area)))

    for i in range(1, num):
        x, y, cw, ch, area = map(int, stats[i])
        if area < min_area or cw <= 0 or ch <= 0:
            continue
        if area > 0.025 * image_area:
            continue
        fill = area / max(1.0, float(cw * ch))
        if fill < 0.012:
            continue

        comp = {
            "bbox": (x, y, x + cw, y + ch),
            "w": cw,
            "h": ch,
            "area": area,
            "fill": fill,
            "centroid": tuple(map(float, centroids[i])),
        }
        small.append(comp)

        if ch < anchor_min_h or ch > anchor_max_h:
            continue
        # Reject long page/table edges while keeping tall narrow symbols such as 1/y/∫.
        if cw > 0.24 * w and cw > 5.5 * ch:
            continue
        if ch > 0.24 * h and ch > 5.5 * cw:
            continue
        touches = x <= 1 or y <= 1 or x + cw >= w - 1 or y + ch >= h - 1
        if touches and (cw > 0.08 * w or ch > 0.08 * h):
            continue
        anchors.append(comp)

    return small, anchors


def _equation_candidates(image: Array, config: PreprocessConfig) -> Tuple[Array, List[Tuple[float, BBox, List[Dict[str, Any]], float, float]]]:
    mask = _ink_likelihood_mask(image)
    h, w = mask.shape[:2]
    small, anchors = _collect_ink_components(mask, config)
    if not anchors:
        return mask, []

    parent = list(range(len(anchors)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    # Group only components that share a plausible equation baseline/band.  This is
    # deliberately stricter than v2's transitive grouping, which let wood grain chain
    # across the entire photograph.
    for i, a in enumerate(anchors):
        ax0, ay0, ax1, ay1 = a["bbox"]
        acy = (ay0 + ay1) / 2.0
        for j in range(i + 1, len(anchors)):
            b = anchors[j]
            bx0, by0, bx1, by1 = b["bbox"]
            bcy = (by0 + by1) / 2.0
            rh = max(a["h"], b["h"])
            sh = max(1, min(a["h"], b["h"]))
            if rh / sh > 3.2:
                continue
            hgap = max(0, max(ax0, bx0) - min(ax1, bx1))
            if abs(acy - bcy) <= 0.78 * rh and hgap <= 3.6 * rh:
                union(i, j)

    groups: Dict[int, List[Dict[str, Any]]] = {}
    for i, c in enumerate(anchors):
        groups.setdefault(find(i), []).append(c)

    candidates: List[Tuple[float, BBox, List[Dict[str, Any]], float, float]] = []
    image_area = float(h * w)
    for group in groups.values():
        bbox = group[0]["bbox"]
        for c in group[1:]:
            bbox = _bbox_union(bbox, c["bbox"])
        x0, y0, x1, y1 = bbox
        bw, bh = x1 - x0, y1 - y0
        if bw < 3 or bh < 3 or bw * bh > 0.35 * image_area:
            continue

        ink_area = float(sum(c["area"] for c in group))
        median_h = float(np.median([c["h"] for c in group]))
        density = ink_area / max(1.0, float(bw * bh))
        aspect = bw / max(1.0, float(bh))
        nbonus = min(len(group), 10) ** 0.65
        shape_bonus = 1.0 + 0.15 * min(6.0, aspect)
        density_bonus = float(np.clip(density / 0.08, 0.50, 2.0))

        # Strongly downweight groups glued to the image boundary; this suppresses
        # keyboard/table/bezel structures observed in the iPad test image.
        border_margin = 0.012 * min(h, w)
        touches_group_border = x0 <= border_margin or y0 <= border_margin or x1 >= w - border_margin or y1 >= h - border_margin
        border_bonus = 0.32 if touches_group_border else 1.0

        score = ink_area * max(1.0, median_h) * nbonus * shape_bonus * density_bonus * border_bonus
        candidates.append((float(score), bbox, group, ink_area, median_h))

    candidates.sort(key=lambda t: t[0], reverse=True)

    # Grow candidates only with genuinely small, immediately adjacent components.
    # v2/v3-alpha used a large rectangular expansion window; on wood/table photos that
    # pulled page edges and texture into an otherwise correct equation box.
    grown_candidates: List[Tuple[float, BBox, List[Dict[str, Any]], float, float]] = []
    for score, bbox, group, ink_area, median_h in candidates:
        grown = bbox
        x0, y0, x1, y1 = bbox
        for c in small:
            # Anchor-scale or larger components were already handled by grouping.  The
            # growth pass is only for punctuation, short '=' bars, dots, superscripts,
            # and subscripts.
            if c["h"] > 0.95 * median_h or c["w"] > 2.8 * median_h:
                continue
            cx0, cy0, cx1, cy1 = c["bbox"]
            hgap = max(0, max(x0, cx0) - min(x1, cx1))
            vgap = max(0, max(y0, cy0) - min(y1, cy1))
            if hgap <= 0.45 * median_h and vgap <= 0.50 * median_h:
                grown = _bbox_union(grown, c["bbox"])
        grown_candidates.append((score, grown, group, ink_area, median_h))

    return mask, grown_candidates


def locate_equation_region(
    image: Array,
    *,
    config: Optional[PreprocessConfig] = None,
    return_mask: bool = False,
) -> Union[CropInfo, Tuple[CropInfo, Array]]:
    config = config or PreprocessConfig()
    h, w = image.shape[:2]
    if _is_clean_ink_canvas(image):
        bbox = _clean_canvas_ink_bbox(image)
        if bbox is not None:
            info = CropInfo(True, bbox, 0.98, "clean ink canvas; retained the bounding box of all visible ink")
            mask = ((cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image) < 250).astype(np.uint8) * 255
            return (info, mask) if return_mask else info
    mask, candidates = _equation_candidates(image, config)
    if not candidates:
        info = CropInfo(False, None, 0.0, "no coherent handwriting-scale component group")
        return (info, mask) if return_mask else info

    best_score, bbox, group, ink_area, _median_h = candidates[0]
    x0, y0, x1, y1 = bbox
    if (x1 - x0) * (y1 - y0) < 0.0005 * h * w:
        info = CropInfo(False, None, 0.0, "best equation candidate is too small")
        return (info, mask) if return_mask else info

    runner = candidates[1][0] if len(candidates) > 1 else 0.0
    separation = best_score / max(best_score + runner, 1e-9)
    size_fraction = min(1.0, _bbox_area(bbox) / max(1.0, 0.04 * h * w))
    count_bonus = min(1.0, len(group) / 5.0)
    confidence = float(min(0.94, 0.42 + 0.28 * separation + 0.12 * size_fraction + 0.12 * count_bonus))
    info = CropInfo(True, tuple(map(int, bbox)), confidence, f"selected coherent expression group with {len(group)} anchor components")
    return (info, mask) if return_mask else info


def crop_equation_region(
    image: Array,
    *,
    padding_ratio: float = 0.14,
    min_padding_px: int = 6,
    config: Optional[PreprocessConfig] = None,
    return_info: bool = False,
) -> Union[Array, Tuple[Array, CropInfo]]:
    config = config or PreprocessConfig(crop_padding_ratio=padding_ratio, crop_min_padding_px=min_padding_px)
    info = locate_equation_region(image, config=config)
    if not info.applied or info.bbox_xyxy is None:
        return (image, info) if return_info else image

    h, w = image.shape[:2]
    x0, y0, x1, y1 = info.bbox_xyxy
    crop_h = max(1, y1 - y0)
    pad = max(min_padding_px, int(round(padding_ratio * crop_h)))
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(w, x1 + pad)
    y1 = min(h, y1 + pad)

    cropped = image[y0:y1, x0:x1]
    out_info = CropInfo(True, (x0, y0, x1, y1), info.confidence, info.reason)
    return (cropped, out_info) if return_info else cropped


# Backwards-compatible private helper.
def _crop_to_content(image: Array, pad: int = 10) -> Array:
    return crop_equation_region(image, padding_ratio=0.14, min_padding_px=pad, return_info=False)


# ---------------------------------------------------------------------------
# Writing-surface estimation guided by the equation
# ---------------------------------------------------------------------------


def _surface_mask_from_equation(image: Array, equation_bbox: BBox, config: PreprocessConfig) -> SurfaceInfo:
    """Estimate the locally smooth planar surface that surrounds the equation.

    This is intentionally an appearance prior, not a generic segmentation model.
    It works well for common capture surfaces such as paper, sticky notes, whiteboards,
    and bright tablet screens.  When the mask is irregular it is kept only as guidance
    for line detection and is *not* forced into a quadrilateral.
    """
    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        bgr = image
    h, w = bgr.shape[:2]
    x0, y0, x1, y1 = equation_bbox
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Estimate surface color from a neighborhood surrounding the equation.  A robust
    # luminance slice removes much of the dark ink before taking the Lab median.
    ex0 = max(0, int(x0 - 0.15 * bw))
    ex1 = min(w, int(x1 + 0.15 * bw))
    ey0 = max(0, int(y0 - 0.80 * bh))
    ey1 = min(h, int(y1 + 0.80 * bh))
    roi = lab[ey0:ey1, ex0:ex1]
    if roi.size == 0:
        return SurfaceInfo(reason="empty surface sampling region")
    vals = roi.reshape(-1, 3)
    L = vals[:, 0]
    lo, hi = np.percentile(L, [35, 95])
    selected = vals[(L >= lo) & (L <= hi)]
    if len(selected) < 100:
        selected = vals
    median_lab = np.median(selected, axis=0)
    dsel = np.sqrt(np.sum((selected - median_lab) ** 2, axis=1))
    dmed = float(np.median(dsel))
    dmad = float(np.median(np.abs(dsel - dmed))) + 1e-6
    color_thr = float(np.clip(dmed + 4.0 * dmad, 10.0, 30.0))

    # Local texture gate: writing surfaces are usually smoother than surrounding wood,
    # fabric, keyboards, etc.  The threshold is learned from the equation neighborhood.
    mean = cv2.boxFilter(gray, -1, (11, 11))
    mean2 = cv2.boxFilter(gray * gray, -1, (11, 11))
    local_std = np.sqrt(np.maximum(mean2 - mean * mean, 0.0))
    sample_std = local_std[ey0:ey1, ex0:ex1]
    texture_thr = float(np.clip(np.percentile(sample_std, 70) + 2.0, 5.0, 12.0))

    color_dist = np.sqrt(np.sum((lab - median_lab) ** 2, axis=2))
    mask = ((color_dist <= color_thr) & (local_std <= texture_thr)).astype(np.uint8) * 255

    k = max(5, int(round(min(h, w) * 0.008))) | 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    num, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    hint = np.zeros((h, w), dtype=np.uint8)
    hx0 = max(0, int(x0 - 0.10 * bw))
    hx1 = min(w, int(x1 + 0.10 * bw))
    hy0 = max(0, int(y0 - 0.35 * bh))
    hy1 = min(h, int(y1 + 0.35 * bh))
    hint[hy0:hy1, hx0:hx1] = 1
    hint_area = max(1, int(np.count_nonzero(hint)))

    best: Optional[Tuple[float, int, int, int]] = None
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        area_ratio = area / max(1.0, float(h * w))
        if area_ratio < config.surface_min_area_ratio or area_ratio > config.surface_max_area_ratio:
            continue
        overlap = int(np.count_nonzero((labels == i) & (hint > 0)))
        overlap_ratio = overlap / hint_area
        score = overlap_ratio + 0.08 * min(1.0, area_ratio / 0.20)
        if best is None or score > best[0]:
            best = (float(score), i, area, overlap)

    if best is None:
        return SurfaceInfo(mask=mask, confidence=0.0, reason="no large smooth component overlaps the equation")

    component = (labels == best[1]).astype(np.uint8) * 255
    k2 = max(7, int(round(min(h, w) * 0.012))) | 1
    component = cv2.morphologyEx(component, cv2.MORPH_CLOSE, np.ones((k2, k2), np.uint8))

    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return SurfaceInfo(mask=component, confidence=0.25, reason="surface component has no contour")
    contour = max(contours, key=cv2.contourArea)
    contour_area = float(cv2.contourArea(contour))
    peri = float(cv2.arcLength(contour, True))
    if contour_area <= 0 or peri <= 0:
        return SurfaceInfo(mask=component, confidence=0.25, reason="degenerate surface contour")

    bx, by, bcw, bch = cv2.boundingRect(contour)
    touches_frame = bx <= 2 or by <= 2 or bx + bcw >= w - 2 or by + bch >= h - 2

    # Only accept a 4-point approximation if it becomes rectangular at a *small*
    # approximation tolerance.  An irregular iPad-screen interior blob, for example,
    # should not be forced into a quad merely because epsilon=0.05 eventually gives 4.
    quad: Optional[Array] = None
    chosen_eps: Optional[float] = None
    for eps in (0.005, 0.010, 0.015, 0.020, 0.025):
        approx = cv2.approxPolyDP(contour, eps * peri, True)
        if len(approx) == 4:
            quad = _order_points(approx.reshape(4, 2))
            chosen_eps = eps
            break

    if quad is None:
        overlap_ratio = best[3] / hint_area
        conf = float(min(0.68, 0.35 + 0.28 * overlap_ratio))
        return SurfaceInfo(
            mask=component,
            quad=None,
            confidence=conf,
            touches_frame=touches_frame,
            reason="smooth equation-bearing surface found, but its visible contour is not a clean quadrilateral",
        )

    valid, reason = _validate_quad(quad, image.shape, config)
    if not valid:
        return SurfaceInfo(mask=component, confidence=0.35, touches_frame=touches_frame, reason=f"surface quad rejected: {reason}")

    semantic_ok, semantic_score, semantic_reason = _quad_equation_score(quad, equation_bbox)
    if not semantic_ok:
        return SurfaceInfo(mask=component, confidence=0.35, touches_frame=touches_frame, reason=f"surface quad semantic gate failed: {semantic_reason}")

    # Compare the actual segmented surface with the proposed quadrilateral.  High IoU
    # is the key guard against manufacturing a rectangle from an irregular blob.
    quad_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(quad_mask, np.round(quad).astype(np.int32), 255)
    cm = component > 0
    qm = quad_mask > 0
    inter = int(np.count_nonzero(cm & qm))
    union = int(np.count_nonzero(cm | qm))
    iou = inter / max(1, union)
    if iou < config.surface_min_quad_iou:
        return SurfaceInfo(
            mask=component,
            quad=None,
            confidence=0.45,
            touches_frame=touches_frame,
            reason=f"surface contour is too irregular for homography (mask/quad IoU={iou:.3f})",
        )

    overlap_ratio = best[3] / hint_area
    confidence = 0.58 + 0.22 * iou + 0.12 * min(1.0, overlap_ratio) + 0.05 * semantic_score
    if touches_frame:
        confidence -= 0.05  # clipped surface: useful, but less certain than all physical borders
    confidence = float(np.clip(confidence, 0.0, 0.97))
    kind = "clipped visible surface" if touches_frame else "complete visible surface"
    return SurfaceInfo(
        mask=component,
        quad=quad,
        confidence=confidence,
        touches_frame=touches_frame,
        reason=f"{kind}; eps={chosen_eps:.3f}, mask/quad IoU={iou:.3f}",
    )


# ---------------------------------------------------------------------------
# Hough / contour geometry with equation/surface guidance
# ---------------------------------------------------------------------------


def _detect_line_segments(
    image: Array,
    min_line_length_ratio: float = 0.08,
    hough_threshold: int = 50,
    guidance_mask: Optional[Array] = None,
) -> Tuple[List[Line], Array]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    h, w = edges.shape[:2]
    min_len = max(25, int(min_line_length_ratio * min(h, w)))
    raw = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 360.0,
        threshold=hough_threshold,
        minLineLength=min_len,
        maxLineGap=max(15, int(0.02 * min(h, w))),
    )
    if raw is None:
        return [], edges

    # OpenCV returns either (N, 1, 4) or (N, 4), depending on the build.
    # Normalise both forms before iterating so the fallback perspective path
    # works on the same machine as the production inference environment.
    raw = np.asarray(raw).reshape(-1, 4)
    lines: List[Line] = [tuple(map(float, line)) for line in raw]
    if guidance_mask is not None and guidance_mask.shape[:2] == (h, w):
        # Dilate the surface mask so its true borders and nearby supporting lines remain.
        radius = max(5, int(round(0.055 * min(h, w))))
        k = 2 * radius + 1
        guide = cv2.dilate((guidance_mask > 0).astype(np.uint8), np.ones((k, k), np.uint8))
        filtered: List[Line] = []
        for line in lines:
            x1, y1, x2, y2 = line
            pts = [
                (int(round(x1)), int(round(y1))),
                (int(round(x2)), int(round(y2))),
                (int(round((x1 + x2) / 2.0)), int(round((y1 + y2) / 2.0))),
            ]
            keep = False
            for x, y in pts:
                if 0 <= x < w and 0 <= y < h and guide[y, x] > 0:
                    keep = True
                    break
            if keep:
                filtered.append(line)
        if len(filtered) >= 4:
            lines = filtered
    return lines, edges


def _cluster_two_orientations(
    lines: Sequence[Line],
    min_family_size: int = 2,
    min_separation_deg: float = 20.0,
) -> Optional[Tuple[List[Line], List[Line], float, float]]:
    if len(lines) < 2 * min_family_size:
        return None
    angles = np.asarray([_line_angle(l) for l in lines], dtype=np.float32)
    samples = np.column_stack((np.cos(2 * angles), np.sin(2 * angles))).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
    _comp, labels, centers = cv2.kmeans(samples, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(-1)
    fam0 = [lines[i] for i in range(len(lines)) if labels[i] == 0]
    fam1 = [lines[i] for i in range(len(lines)) if labels[i] == 1]
    if len(fam0) < min_family_size or len(fam1) < min_family_size:
        return None
    t0 = 0.5 * math.atan2(float(centers[0, 1]), float(centers[0, 0])) % math.pi
    t1 = 0.5 * math.atan2(float(centers[1, 1]), float(centers[1, 0])) % math.pi
    if math.degrees(_angle_distance(t0, t1)) < min_separation_deg:
        return None
    return fam0, fam1, t0, t1


def _select_outer_boundary_lines(family: Sequence[Line], theta: float) -> Optional[Tuple[Line, Line]]:
    if len(family) < 2:
        return None
    normal = np.array([-math.sin(theta), math.cos(theta)], dtype=np.float64)
    rows = []
    for line in family:
        x1, y1, x2, y2 = line
        mid = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
        rows.append((float(mid @ normal), _line_length(line), line))
    rows.sort(key=lambda r: r[0])
    k = max(1, int(math.ceil(0.20 * len(rows))))
    low = max(rows[:k], key=lambda r: r[1])[2]
    high = max(rows[-k:], key=lambda r: r[1])[2]
    return low, high


def _corners_via_hough(image: Array, config: PreprocessConfig, guidance_mask: Optional[Array] = None) -> Optional[Array]:
    lines, _ = _detect_line_segments(image, guidance_mask=guidance_mask)
    clustered = _cluster_two_orientations(lines, min_family_size=2, min_separation_deg=25.0)
    if clustered is None:
        return None
    fam0, fam1, t0, t1 = clustered
    b0 = _select_outer_boundary_lines(fam0, t0)
    b1 = _select_outer_boundary_lines(fam1, t1)
    if b0 is None or b1 is None:
        return None
    pts: List[Tuple[float, float]] = []
    for a in b0:
        for b in b1:
            p = _line_intersection(a, b)
            if p is None:
                return None
            pts.append(p)
    corners = _order_points(pts)
    valid, _ = _validate_quad(corners, image.shape, config)
    return corners if valid else None


def _corners_via_contour(image: Array, config: PreprocessConfig) -> Optional[Array]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape[:2]
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:16]:
        if cv2.contourArea(contour) < config.min_quad_area_ratio * h * w:
            continue
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        corners = _order_points(approx.reshape(4, 2))
        valid, _ = _validate_quad(corners, image.shape, config)
        if valid:
            return corners
    return None


# ---------------------------------------------------------------------------
# Vanishing-point rectification for incomplete/non-quad surfaces
# ---------------------------------------------------------------------------


def _weighted_orientation_mean(lines: Sequence[Line]) -> float:
    weights = np.asarray([max(1.0, _line_length(l)) for l in lines], dtype=np.float64)
    angles = np.asarray([_line_angle(l) for l in lines], dtype=np.float64)
    s = float(np.sum(weights * np.sin(2.0 * angles)))
    c = float(np.sum(weights * np.cos(2.0 * angles)))
    return (0.5 * math.atan2(s, c)) % math.pi


def _dominant_orientation_family(lines: Sequence[Line], image_shape: Sequence[int], tolerance_deg: float = 8.0) -> Optional[Tuple[List[Line], float, float]]:
    if len(lines) < 3:
        return None
    bins = 36
    hist = np.zeros(bins, dtype=np.float64)
    for line in lines:
        theta = _line_angle(line)
        hist[min(bins - 1, int(theta / math.pi * bins))] += _line_length(line)
    peak = int(np.argmax(hist))
    peak_theta = (peak + 0.5) / bins * math.pi
    broad = [l for l in lines if _angle_distance(_line_angle(l), peak_theta) <= math.radians(12)]
    if len(broad) < 3:
        return None
    theta = _weighted_orientation_mean(broad)
    family = [l for l in lines if _angle_distance(_line_angle(l), theta) <= math.radians(tolerance_deg)]
    if len(family) < 3:
        return None
    h, w = image_shape[:2]
    diag = math.hypot(w, h)
    total = sum(_line_length(l) for l in family)
    if total < 0.45 * diag:
        return None
    support = min(1.0, total / max(diag, 1.0) / 2.0)
    return family, theta, support


def _fit_vanishing_point(lines: Sequence[Line]) -> Optional[Array]:
    if len(lines) < 2:
        return None
    base = np.vstack([_homogeneous_line(l) for l in lines]).astype(np.float64)
    lengths = np.asarray([max(1.0, _line_length(l)) for l in lines], dtype=np.float64)
    weights = np.sqrt(lengths / np.median(lengths))
    v: Optional[Array] = None
    for _ in range(4):
        try:
            _u, _s, vh = np.linalg.svd(base * weights[:, None], full_matrices=False)
        except np.linalg.LinAlgError:
            return None
        v = vh[-1]
        n = float(np.linalg.norm(v))
        if n < 1e-12 or not np.all(np.isfinite(v)):
            return None
        v = v / n
        residual = np.abs(base @ v)
        med = float(np.median(residual)) + 1e-9
        robust = 1.0 / np.maximum(1.0, residual / (2.5 * med))
        weights = np.sqrt(lengths / np.median(lengths)) * robust
    return v


def _rotate_family_horizontal(H: Array, family: Sequence[Line]) -> Array:
    if not family:
        return H
    rep = max(family, key=_line_length)
    p = _transform_points(np.array([[rep[0], rep[1]], [rep[2], rep[3]]], dtype=np.float32), H)
    angle = math.atan2(float(p[1, 1] - p[0, 1]), float(p[1, 0] - p[0, 0]))
    if angle > math.pi / 2:
        angle -= math.pi
    elif angle < -math.pi / 2:
        angle += math.pi
    ca, sa = math.cos(-angle), math.sin(-angle)
    R = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]], dtype=np.float64)
    return R @ H


def _single_vanishing_direction_homography(image_shape: Sequence[int], vp: Array) -> Optional[Array]:
    to_norm, to_pixel = _normalized_pixel_transform(image_shape)
    vn = to_norm @ np.asarray(vp, dtype=np.float64)
    if abs(float(vn[2])) < 1e-7:
        return np.eye(3, dtype=np.float64)
    vn = vn / vn[2]
    vx, vy = float(vn[0]), float(vn[1])
    d2 = vx * vx + vy * vy
    if d2 < 0.55 * 0.55:
        return None
    if d2 > 400.0 * 400.0:
        return np.eye(3, dtype=np.float64)
    Hn = np.array([[1, 0, 0], [0, 1, 0], [-vx / d2, -vy / d2, 1]], dtype=np.float64)
    return to_pixel @ Hn @ to_norm


def _two_vanishing_direction_homography(image_shape: Sequence[int], vp0: Array, vp1: Array) -> Optional[Array]:
    to_norm, to_pixel = _normalized_pixel_transform(image_shape)
    v0 = to_norm @ np.asarray(vp0, dtype=np.float64)
    v1 = to_norm @ np.asarray(vp1, dtype=np.float64)
    line_inf = np.cross(v0, v1)
    if not np.all(np.isfinite(line_inf)) or np.linalg.norm(line_inf[:2]) < 1e-9 or abs(float(line_inf[2])) < 1e-5:
        return None
    line_inf = line_inf / line_inf[2]
    Hn = np.array([[1, 0, 0], [0, 1, 0], [line_inf[0], line_inf[1], 1]], dtype=np.float64)
    return to_pixel @ Hn @ to_norm


def _hint_survives_transform(H: Array, equation_hint: Optional[BBox]) -> bool:
    if equation_hint is None:
        return True
    pts = np.asarray(_bbox_sample_points(equation_hint), dtype=np.float32)
    try:
        dst = _transform_points(pts, H)
    except cv2.error:
        return False
    if not np.all(np.isfinite(dst)):
        return False
    bw = float(dst[:, 0].max() - dst[:, 0].min())
    bh = float(dst[:, 1].max() - dst[:, 1].min())
    src_w = max(1.0, equation_hint[2] - equation_hint[0])
    src_h = max(1.0, equation_hint[3] - equation_hint[1])
    if bw < 0.15 * src_w or bh < 0.15 * src_h:
        return False
    aspect_change = (bw / max(bh, 1e-6)) / (src_w / src_h)
    return 0.25 <= aspect_change <= 4.0


def _try_vanishing_rectification(
    image: Array,
    config: PreprocessConfig,
    guidance_mask: Optional[Array] = None,
    equation_hint: Optional[BBox] = None,
) -> Optional[Tuple[Array, PerspectiveInfo]]:
    lines, _ = _detect_line_segments(image, min_line_length_ratio=0.08, hough_threshold=50, guidance_mask=guidance_mask)
    if len(lines) < 3:
        return None

    clustered = _cluster_two_orientations(lines, min_family_size=3, min_separation_deg=25.0)
    if clustered is not None:
        fam0, fam1, _t0, _t1 = clustered
        vp0, vp1 = _fit_vanishing_point(fam0), _fit_vanishing_point(fam1)
        if vp0 is not None and vp1 is not None:
            H = _two_vanishing_direction_homography(image.shape, vp0, vp1)
            if H is not None:
                H0, H1 = _rotate_family_horizontal(H, fam0), _rotate_family_horizontal(H, fam1)

                def slope_score(Hc: Array) -> float:
                    rep = max(fam0 + fam1, key=_line_length)
                    p = _transform_points(np.array([[rep[0], rep[1]], [rep[2], rep[3]]], np.float32), Hc)
                    return abs(math.atan2(float(p[1, 1] - p[0, 1]), float(p[1, 0] - p[0, 0])))

                H = H0 if slope_score(H0) <= slope_score(H1) else H1
                fitted = _fit_homography_to_canvas(H, image.shape, config.max_rectified_scale)
                if fitted is not None:
                    Hf, size = fitted
                    if _hint_survives_transform(Hf, equation_hint):
                        warped = _warp_white(image, Hf, size)
                        conf = min(0.86, 0.58 + 0.018 * min(len(fam0), len(fam1)))
                        return warped, PerspectiveInfo(
                            "vanishing_points_2d",
                            conf,
                            Hf,
                            "two dominant guided vanishing directions; no 4-corner requirement",
                        )

    if not config.enable_partial_rectification:
        return None
    dominant = _dominant_orientation_family(lines, image.shape)
    if dominant is None:
        return None
    family, _theta, support = dominant
    vp = _fit_vanishing_point(family)
    if vp is None:
        return None
    H = _single_vanishing_direction_homography(image.shape, vp)
    if H is None:
        return None
    H = _rotate_family_horizontal(H, family)
    fitted = _fit_homography_to_canvas(H, image.shape, config.max_rectified_scale)
    if fitted is None:
        return None
    Hf, size = fitted
    if not _hint_survives_transform(Hf, equation_hint):
        return None
    warped = _warp_white(image, Hf, size)
    conf = min(0.74, 0.43 + 0.22 * support + 0.004 * min(20, len(family)))
    return warped, PerspectiveInfo(
        "vanishing_point_1d_partial",
        conf,
        Hf,
        "only one reliable guided line family was observable; corrected that convergence only",
    )


# ---------------------------------------------------------------------------
# Learned-rectifier hook
# ---------------------------------------------------------------------------


def _run_learned_rectifier(image: Array, rectifier: LearnedRectifier) -> Optional[Tuple[Array, PerspectiveInfo]]:
    try:
        result = rectifier(image)
    except Exception:
        return None
    confidence = 0.75
    if isinstance(result, tuple):
        rectified, confidence = result
    else:
        rectified = result
    if not isinstance(rectified, np.ndarray) or rectified.size == 0 or rectified.ndim not in (2, 3):
        return None
    if rectified.dtype != np.uint8:
        rectified = np.clip(rectified, 0, 255).astype(np.uint8)
    return rectified, PerspectiveInfo(
        "learned_rectifier",
        float(np.clip(confidence, 0.0, 1.0)),
        None,
        "output supplied by optional learned rectifier",
    )


# ---------------------------------------------------------------------------
# Perspective correction public entry point
# ---------------------------------------------------------------------------


def perspective_correct(
    image: Array,
    output_size: Optional[Tuple[int, int]] = None,
    *,
    config: Optional[PreprocessConfig] = None,
    learned_rectifier: Optional[LearnedRectifier] = None,
    prefer_learned: bool = False,
    equation_hint: Optional[BBox] = None,
    surface_info: Optional[SurfaceInfo] = None,
    return_info: bool = False,
) -> Union[Array, Tuple[Array, PerspectiveInfo]]:
    config = config or PreprocessConfig()

    def done(img: Array, info: PerspectiveInfo):
        return (img, info) if return_info else img

    if not config.enable_perspective:
        return done(image, PerspectiveInfo("identity", 1.0, None, "perspective correction disabled"))

    if _is_clean_ink_canvas(image):
        return done(
            image,
            PerspectiveInfo(
                "identity_clean_canvas",
                1.0,
                None,
                "clean MathWriting-style ink canvas has no camera perspective to estimate",
            ),
        )

    if prefer_learned and learned_rectifier is not None:
        learned = _run_learned_rectifier(image, learned_rectifier)
        if learned is not None:
            return done(*learned)

    # Highest-priority classical path: a smooth equation-bearing surface whose actual
    # segmented contour is already a high-IoU quadrilateral.  If one physical edge is
    # outside the image, the frame-clipped visible portion can still provide a useful
    # four-sided region, which rectifies the pixels we actually have rather than failing.
    if surface_info is not None and surface_info.quad is not None and surface_info.confidence >= 0.70:
        result = _warp_from_quad(image, surface_info.quad, output_size=output_size)
        if result is not None:
            warped, H = result
            method = "surface_quad_clipped" if surface_info.touches_frame else "surface_quad"
            return done(warped, PerspectiveInfo(method, surface_info.confidence, H, surface_info.reason))

    guidance_mask = surface_info.mask if surface_info is not None else None

    # Generic Hough/contour quads are now gated semantically.  Geometry alone is not
    # enough: the wrong table rectangle in test2_cut must be rejected because it does
    # not contain the equation.
    corners = _corners_via_hough(image, config, guidance_mask=guidance_mask)
    if corners is not None:
        ok, semantic_score, reason = _quad_equation_score(corners, equation_hint)
        if ok:
            result = _warp_from_quad(image, corners, output_size=output_size)
            if result is not None:
                warped, H = result
                return done(warped, PerspectiveInfo("hough_quad_guided", semantic_score, H, reason))

    corners = _corners_via_contour(image, config)
    if corners is not None:
        ok, semantic_score, reason = _quad_equation_score(corners, equation_hint)
        if ok:
            result = _warp_from_quad(image, corners, output_size=output_size)
            if result is not None:
                warped, H = result
                return done(warped, PerspectiveInfo("contour_quad_guided", min(0.82, semantic_score), H, reason))

    vanishing = _try_vanishing_rectification(
        image,
        config,
        guidance_mask=guidance_mask,
        equation_hint=equation_hint,
    )
    if vanishing is not None:
        return done(*vanishing)

    if learned_rectifier is not None:
        learned = _run_learned_rectifier(image, learned_rectifier)
        if learned is not None:
            return done(*learned)

    return done(
        image,
        PerspectiveInfo(
            "identity",
            0.0,
            None,
            "insufficient reliable equation-bearing geometry; original image preserved",
        ),
    )


# ---------------------------------------------------------------------------
# Binarization / resize / MobileNet formatting
# ---------------------------------------------------------------------------


def binarize(image: Array, method: str = "otsu") -> Array:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    if method == "otsu":
        _thr, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "adaptive":
        block = int(max(21, min(51, (min(gray.shape[:2]) // 8) | 1)))
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, 10)
    else:
        raise ValueError(f"Unknown binarization method: {method!r}")
    # Ink should always be black, background white.
    if np.count_nonzero(binary == 255) < np.count_nonzero(binary == 0):
        binary = cv2.bitwise_not(binary)
    return binary


def resize_to_height(image: Array, target_height: int = 64) -> Array:
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError("Image has zero width or height")
    if target_height <= 0:
        raise ValueError("target_height must be positive")
    scale = float(target_height) / float(h)
    new_w = max(1, int(round(w * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(image, (new_w, target_height), interpolation=interp)


def letterbox_resize(image: Array, target_height: int = 64, pad_value: int = 255) -> Array:
    del pad_value
    out = resize_to_height(image, target_height)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR) if out.ndim == 2 else out


def to_mobilenet_input(image: Array, *, imagenet_normalize: bool = False) -> Array:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected grayscale or 3-channel image, got {image.shape}")
    img = image.astype(np.float32) / 255.0
    if imagenet_normalize:
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0)


def pad_mobilenet_batch(
    tensors: Sequence[Array],
    *,
    pad_to_multiple: int = 1,
    imagenet_normalize: bool = False,
) -> Tuple[Array, List[int]]:
    if not tensors:
        raise ValueError("tensors must not be empty")
    normalized: List[Array] = []
    for tensor in tensors:
        t = np.asarray(tensor, dtype=np.float32)
        if t.ndim == 4:
            if t.shape[0] != 1:
                raise ValueError("Each 4D tensor must have batch size 1")
            t = t[0]
        if t.ndim != 3:
            raise ValueError(f"Expected (C,H,W) or (1,C,H,W), got {t.shape}")
        normalized.append(t)
    c0, h0 = normalized[0].shape[:2]
    if any(t.shape[0] != c0 or t.shape[1] != h0 for t in normalized):
        raise ValueError("All tensors must have the same channel count and height")
    widths = [int(t.shape[2]) for t in normalized]
    max_w = max(widths)
    if pad_to_multiple > 1:
        max_w = int(math.ceil(max_w / pad_to_multiple) * pad_to_multiple)
    if imagenet_normalize:
        white = ((np.ones(3, dtype=np.float32) - np.asarray([0.485, 0.456, 0.406], np.float32)) /
                 np.asarray([0.229, 0.224, 0.225], np.float32))
        batch = np.empty((len(normalized), c0, h0, max_w), dtype=np.float32)
        for ch in range(3):
            batch[:, ch, :, :] = white[ch]
    else:
        batch = np.ones((len(normalized), c0, h0, max_w), dtype=np.float32)
    for i, t in enumerate(normalized):
        batch[i, :, :, :t.shape[2]] = t
    return batch, widths


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------


def _draw_bbox(image: Array, bbox: Optional[BBox], color: Tuple[int, int, int] = (0, 255, 0)) -> Array:
    if image.ndim == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        cv2.rectangle(vis, (x0, y0), (x1, y1), color, max(2, int(round(min(vis.shape[:2]) * 0.002))))
    return vis


def preprocess_image(
    image: Array,
    *,
    config: Optional[PreprocessConfig] = None,
    learned_rectifier: Optional[LearnedRectifier] = None,
    prefer_learned: bool = False,
    return_debug: bool = False,
) -> Union[Array, Tuple[Array, Dict[str, Any]]]:
    config = config or PreprocessConfig()
    if image is None or image.size == 0:
        raise ValueError("Input image is empty")

    # Stage A: coarse semantic hint before geometry.
    coarse_info, coarse_mask = locate_equation_region(image, config=config, return_mask=True)
    equation_hint = coarse_info.bbox_xyxy if coarse_info.applied else None

    # Stage B: appearance-guided surface estimate.  It may produce a trustworthy
    # quadrilateral or merely a mask used to guide Hough/vanishing lines.
    if config.enable_surface_guidance and equation_hint is not None:
        surface_info = _surface_mask_from_equation(image, equation_hint, config)
    else:
        surface_info = SurfaceInfo(reason="surface guidance unavailable because no coarse equation hint was found")

    # Stage C: safe perspective correction.
    corrected, perspective_info = perspective_correct(
        image,
        config=config,
        learned_rectifier=learned_rectifier,
        prefer_learned=prefer_learned,
        equation_hint=equation_hint,
        surface_info=surface_info,
        return_info=True,
    )

    # Stage D: track the coarse equation hint through the accepted homography, then
    # compare it with a fresh detector on the rectified image.  Tracking prevents a
    # second detector pass from accidentally returning only a subset such as "ax+b"
    # while dropping "=10" after a perspective warp.
    detected_info = locate_equation_region(corrected, config=config)

    tracked_bbox: Optional[BBox] = None
    if equation_hint is not None:
        if perspective_info.homography is None:
            if perspective_info.method == "identity":
                tracked_bbox = equation_hint
        else:
            x0, y0, x1, y1 = equation_hint
            hint_pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
            try:
                t = _transform_points(hint_pts, perspective_info.homography)
                tx0, ty0 = np.floor(t.min(axis=0)).astype(int)
                tx1, ty1 = np.ceil(t.max(axis=0)).astype(int)
                ch, cw = corrected.shape[:2]
                tx0, ty0 = max(0, tx0), max(0, ty0)
                tx1, ty1 = min(cw, tx1), min(ch, ty1)
                if tx1 > tx0 + 3 and ty1 > ty0 + 3:
                    tracked_bbox = (tx0, ty0, tx1, ty1)
            except cv2.error:
                tracked_bbox = None

    chosen_bbox: Optional[BBox] = None
    chosen_conf = 0.0
    chosen_reason = ""
    detected_bbox = detected_info.bbox_xyxy if detected_info.applied else None

    if tracked_bbox is not None and coarse_info.confidence >= 0.65:
        chosen_bbox = tracked_bbox
        chosen_conf = max(0.55, coarse_info.confidence * 0.92)
        chosen_reason = "tracked the high-confidence coarse equation hint through perspective correction"

        # Fresh detection may add a nearby dot/superscript that the coarse pass missed.
        # Merge only when the union remains compact; never let an unrelated texture
        # group make the final crop explode back to the full photograph.
        if detected_bbox is not None:
            ax0, ay0, ax1, ay1 = tracked_bbox
            bx0, by0, bx1, by1 = detected_bbox
            ix0, iy0 = max(ax0, bx0), max(ay0, by0)
            ix1, iy1 = min(ax1, bx1), min(ay1, by1)
            inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
            union_bbox = _bbox_union(tracked_bbox, detected_bbox)
            compact = _bbox_area(union_bbox) <= 1.85 * max(_bbox_area(tracked_bbox), 1.0)
            related = inter > 0 or (
                abs(_bbox_center(tracked_bbox)[0] - _bbox_center(detected_bbox)[0]) <= 0.45 * max(ax1 - ax0, bx1 - bx0)
                and abs(_bbox_center(tracked_bbox)[1] - _bbox_center(detected_bbox)[1]) <= 0.60 * max(ay1 - ay0, by1 - by0)
            )
            if related and compact:
                chosen_bbox = union_bbox
                chosen_conf = max(chosen_conf, detected_info.confidence)
                chosen_reason = "merged tracked coarse hint with a consistent post-rectification detection"
    elif detected_bbox is not None:
        chosen_bbox = detected_bbox
        chosen_conf = detected_info.confidence
        chosen_reason = detected_info.reason

    if chosen_bbox is not None:
        ch, cw = corrected.shape[:2]
        x0, y0, x1, y1 = chosen_bbox
        ph = max(1, y1 - y0)
        pad = max(config.crop_min_padding_px, int(round(config.crop_padding_ratio * ph)))
        x0, y0 = int(max(0, x0 - pad)), int(max(0, y0 - pad))
        x1, y1 = int(min(cw, x1 + pad)), int(min(ch, y1 + pad))
        cropped = corrected[y0:y1, x0:x1]
        crop_info = CropInfo(True, (x0, y0, x1, y1), chosen_conf, chosen_reason)
    else:
        cropped = corrected
        crop_info = CropInfo(False, None, 0.0, "no reliable equation crop after perspective correction")

    binary = binarize(cropped, method=config.binarize_method)
    resized = resize_to_height(binary, target_height=config.target_height)
    model_input = to_mobilenet_input(resized, imagenet_normalize=config.imagenet_normalize)

    if not return_debug:
        return model_input

    debug: Dict[str, Any] = {
        "original": image,
        "coarse_equation_overlay": _draw_bbox(image, equation_hint),
        "coarse_ink_mask": coarse_mask,
        "surface_mask": surface_info.mask if surface_info.mask is not None else np.zeros(image.shape[:2], dtype=np.uint8),
        "surface_quad_overlay": image.copy(),
        "corrected": corrected,
        "cropped": cropped,
        "binary": binary,
        "resized": resized,
        "coarse_equation_info": coarse_info,
        "surface_info": surface_info,
        "perspective_info": perspective_info,
        "crop_info": crop_info,
        "output_shape": tuple(model_input.shape),
    }
    if surface_info.quad is not None:
        cv2.polylines(
            debug["surface_quad_overlay"],
            [np.round(_order_points(surface_info.quad)).astype(np.int32).reshape(-1, 1, 2)],
            True,
            (0, 0, 255),
            max(2, int(round(min(image.shape[:2]) * 0.002))),
        )
        if equation_hint is not None:
            x0, y0, x1, y1 = equation_hint
            cv2.rectangle(debug["surface_quad_overlay"], (x0, y0), (x1, y1), (0, 255, 0), 3)
    return model_input, debug


def preprocess_pipeline(
    image_path: str,
    target_height: int = 64,
    binarize_method: str = "otsu",
    *,
    learned_rectifier: Optional[LearnedRectifier] = None,
    prefer_learned: bool = False,
    imagenet_normalize: bool = False,
    return_debug: bool = False,
) -> Union[Array, Tuple[Array, Dict[str, Any]]]:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    config = PreprocessConfig(
        target_height=target_height,
        binarize_method=binarize_method,
        imagenet_normalize=imagenet_normalize,
    )
    return preprocess_image(
        image,
        config=config,
        learned_rectifier=learned_rectifier,
        prefer_learned=prefer_learned,
        return_debug=return_debug,
    )


def _save_debug_images(debug: Dict[str, Any], debug_dir: str) -> None:
    os.makedirs(debug_dir, exist_ok=True)
    stages = {
        "01_original.jpg": debug["original"],
        "02_coarse_equation_hint.jpg": debug["coarse_equation_overlay"],
        "03_coarse_ink_mask.png": debug["coarse_ink_mask"],
        "04_surface_mask.png": debug["surface_mask"],
        "05_surface_quad.jpg": debug["surface_quad_overlay"],
        "06_corrected.jpg": debug["corrected"],
        "07_equation_crop.jpg": debug["cropped"],
        "08_binary.png": debug["binary"],
        "09_height64.png": debug["resized"],
    }
    for filename, image in stages.items():
        cv2.imwrite(os.path.join(debug_dir, filename), image)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess photographed handwritten math for MobileNet")
    parser.add_argument("image", help="input image path")
    parser.add_argument("--height", type=int, default=64, help="output raster height (default: 64)")
    parser.add_argument("--binarize", choices=["otsu", "adaptive"], default="otsu")
    parser.add_argument("--no-perspective", action="store_true", help="disable all perspective correction")
    parser.add_argument("--no-partial", action="store_true", help="disable one-direction partial rectification")
    parser.add_argument("--no-surface-guidance", action="store_true", help="disable equation-guided smooth-surface detection")
    parser.add_argument("--imagenet-normalize", action="store_true")
    parser.add_argument("--debug-dir", default=None, help="save intermediate images")
    args = parser.parse_args()

    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")
    config = PreprocessConfig(
        target_height=args.height,
        binarize_method=args.binarize,
        enable_perspective=not args.no_perspective,
        enable_partial_rectification=not args.no_partial,
        enable_surface_guidance=not args.no_surface_guidance,
        imagenet_normalize=args.imagenet_normalize,
    )
    tensor, debug = preprocess_image(image, config=config, return_debug=True)

    coarse: CropInfo = debug["coarse_equation_info"]
    surface: SurfaceInfo = debug["surface_info"]
    pinfo: PerspectiveInfo = debug["perspective_info"]
    cinfo: CropInfo = debug["crop_info"]
    print("Coarse equation bbox:", coarse.bbox_xyxy)
    print("Coarse equation confidence:", f"{coarse.confidence:.3f}")
    print("Surface confidence:", f"{surface.confidence:.3f}")
    print("Surface note:", surface.reason)
    print("Perspective method:", pinfo.method)
    print("Perspective confidence:", f"{pinfo.confidence:.3f}")
    print("Perspective note:", pinfo.reason)
    print("Crop applied:", cinfo.applied)
    print("Crop bbox:", cinfo.bbox_xyxy)
    print("Crop confidence:", f"{cinfo.confidence:.3f}")
    print("Output tensor shape:", tensor.shape)

    if args.debug_dir:
        _save_debug_images(debug, args.debug_dir)
        print("Saved debug stages to:", args.debug_dir)


if __name__ == "__main__":
    _main()

