"""Re-render an arbitrary image the way mathwriting_pipeline renders InkML.

The model has only ever seen InkML renders: uniform STROKE_WIDTH_PX strokes,
ink scaled to TARGET_HEIGHT - 2*MARGIN_PX, MARGIN_PX of white around it, dark
on white. A photo or a drawing-app export matches none of that -- stroke width
varies with pen and pressure, and is usually far heavier relative to glyph size
than a 2.5px render. That mismatch is a domain gap, not a recognition problem,
and it is what makes thick handwriting decode badly.

This closes the gap by throwing away stroke thickness entirely and keeping only
the centreline, then re-rendering that centreline at exactly the width the
training data uses:

    binarise -> Zhang-Suen thin to 1px -> scale centreline to the ink height
    -> redraw at STROKE_WIDTH_PX with supersampled anti-aliasing

Scaling happens on the centreline coordinates rather than on pixels, because
downsampling a 1px skeleton destroys it. Redrawing afterwards is also what
makes the output stroke width correct at the FINAL resolution -- thinning after
scaling would leave width dependent on the input image's size.

Zhang-Suen is implemented here rather than pulled from skimage or
cv2.ximgproc: requirements.txt is deliberately small, and this is ~30 lines of
array arithmetic.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from mathwriting_pipeline import TARGET_HEIGHT, MARGIN_PX, STROKE_WIDTH_PX, SUPERSAMPLE

_NEIGHBOUR_OFFSETS = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                      (1, 0), (1, -1), (0, -1), (-1, -1)]


def _neighbours(padded):
    """The 8 neighbours of every pixel, in Zhang-Suen's P2..P9 order."""
    return [padded[1 + dy: padded.shape[0] - 1 + dy,
                   1 + dx: padded.shape[1] - 1 + dx]
            for dy, dx in _NEIGHBOUR_OFFSETS]


def thin(mask: np.ndarray, max_iters: int = 100) -> np.ndarray:
    """Zhang-Suen thinning. mask: bool array, True = ink. Returns 1px centrelines.

    max_iters bounds the loop; the algorithm converges in far fewer passes for
    normal handwriting, but a pathological input should not hang inference.
    """
    img = mask.astype(bool).copy()
    for _ in range(max_iters):
        changed = False
        for step in (0, 1):
            padded = np.pad(img, 1, mode="constant", constant_values=False)
            p = _neighbours(padded)
            p2, p3, p4, p5, p6, p7, p8, p9 = p

            # B = number of ink neighbours; A = 0->1 transitions around the ring
            B = sum(x.astype(np.uint8) for x in p)
            ring = p + [p2]
            A = sum(((~ring[i]) & ring[i + 1]).astype(np.uint8) for i in range(8))

            if step == 0:
                c1, c2 = ~(p2 & p4 & p6), ~(p4 & p6 & p8)
            else:
                c1, c2 = ~(p2 & p4 & p8), ~(p2 & p6 & p8)

            doomed = img & (B >= 2) & (B <= 6) & (A == 1) & c1 & c2
            if doomed.any():
                img[doomed] = False
                changed = True
        if not changed:
            break
    return img


def to_inkml_render(image, target_height: int = TARGET_HEIGHT,
                    margin: int = MARGIN_PX,
                    stroke_width: float = STROKE_WIDTH_PX,
                    supersample: int = SUPERSAMPLE) -> Image.Image:
    """Any PIL image -> a render matching mathwriting_pipeline's output.

    Returns an "L" mode image, dark ink on white, exactly target_height tall.
    """
    grey = np.asarray(image.convert("L"), dtype=np.uint8)

    # Otsu, so the threshold adapts to the image rather than assuming a level.
    hist = np.bincount(grey.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    omega = np.cumsum(hist) / total
    mu = np.cumsum(hist * np.arange(256)) / total
    mu_t = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega))
    threshold = int(np.nanargmax(sigma_b))
    ink = grey <= threshold                       # dark pixels are ink

    if not ink.any():
        raise ValueError("no ink found -- the image looks blank after binarising")

    ys, xs = np.nonzero(thin(ink))
    if len(xs) == 0:
        raise ValueError("thinning removed every pixel")

    # Scale the centreline so the ink spans target_height - 2*margin, matching
    # rescale_to_height + fit_canvas. Width follows from the aspect ratio.
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    ink_h = max(y1 - y0, 1)
    scale = (target_height - 2 * margin) / ink_h
    out_w = max(1, int(round((x1 - x0) * scale)) + 2 * margin)

    ss = supersample
    canvas = Image.new("L", (out_w * ss, target_height * ss), color=255)
    draw = ImageDraw.Draw(canvas)
    r = stroke_width * ss / 2.0
    px = (xs - x0) * scale * ss + margin * ss
    py = (ys - y0) * scale * ss + margin * ss
    for cx, cy in zip(px, py):
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=0)

    return canvas.resize((out_w, target_height), Image.LANCZOS)
