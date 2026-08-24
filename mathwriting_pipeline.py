"""Shared InkML/rendering/augmentation/tokenizer code.

Extracted so `mathwriting_preprocessing.ipynb` and `dataset.py` (the
training-time PyTorch Dataset) call the exact same rendering and
augmentation logic instead of maintaining two copies that could drift --
in particular, `render_with_augmentation` and `AUGMENTATION_CONFIG` are
meant to be identical between preprocessing-time QA and training-time use.

The notebook remains the source of truth for *running* the pipeline
(config, per-split orchestration, sanity checks, visual QA); this module
just holds the reusable primitives.
"""
import math
import re
import dataclasses

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

try:
    import cairo
    CAIRO_AVAILABLE = True
except ImportError:
    CAIRO_AVAILABLE = False

# --- Image normalization ---
TARGET_HEIGHT = 64        # pixels, final canvas height (fixed; only width varies)
STROKE_WIDTH_PX = 2.5     # constant rendered stroke width, in pixels
MARGIN_PX = 4             # blank margin added around the ink on each side
SUPERSAMPLE = 4           # Pillow fallback supersample factor

# The ink itself is normalized to TARGET_HEIGHT - 2*MARGIN_PX so that once
# fit_canvas adds `margin` on the top and bottom, the canvas comes out to
# exactly TARGET_HEIGHT.
_NORMALIZED_INK_HEIGHT = TARGET_HEIGHT - 2 * MARGIN_PX

# A handful of single-stroke glyphs (e.g. `\overline`, `\underline`, minus
# signs) are near-flat marks whose raw bounding-box height is just pen jitter
# (~1-2 raw units, vs 80+ for every other observed glyph in this excerpt).
# Scaling those up to TARGET_HEIGHT demands a 25x-50x factor, which blows the
# *width* up proportionally (observed up to ~1850px on this excerpt, vs a
# normal p99 of ~450px) since x and y share one scale factor. Capping how
# much any ink can be upscaled bounds this; every legitimately-small glyph in
# this excerpt needs under 2x, so this only catches genuinely degenerate ones.
MAX_UPSCALE = 5.0

RNG_SEED = 0

AUGMENTATION_CONFIG = {
    "rotation_deg":   {"prob": 0.5, "range": (-6, 6),    "dtype": float},
    "shear_deg":      {"prob": 0.4, "range": (-8, 8),    "dtype": float},
    "blur_radius":    {"prob": 0.3, "range": (0.3, 1.0), "dtype": float},
}

_COMMAND_RE = re.compile(r'\\(mathbb{[a-zA-Z]}|begin{[a-z]+}|end{[a-z]+}|operatorname\*|[a-zA-Z]+|.)')


@dataclasses.dataclass
class Ink:
    """A single ink: its strokes and InkML annotations."""
    strokes: list  # each element is an array of shape (2, num_points): rows are (x, y)
    annotations: dict


def read_inkml_file(filename) -> Ink:
    """Parses a single MathWriting InkML file."""
    from xml.etree import ElementTree

    with open(filename, "r", encoding="utf-8") as f:
        root = ElementTree.fromstring(f.read())

    strokes = []
    annotations = {}

    for element in root:
        tag_name = element.tag.split("}")[-1]
        if tag_name == "annotation":
            annotations[element.attrib.get("type")] = element.text
        elif tag_name == "trace":
            xs, ys = [], []
            for point in element.text.split(","):
                x, y, _t = point.split(" ")
                xs.append(float(x))
                ys.append(float(y))
            strokes.append(np.array((xs, ys)))

    return Ink(strokes=strokes, annotations=annotations)


def rescale_to_height(ink: Ink, target_height: float = _NORMALIZED_INK_HEIGHT) -> Ink:
    """Uniformly scales (x and y by the same factor) so the ink's bounding-box
    height equals target_height. Does not reposition the ink.

    The scale factor is capped at MAX_UPSCALE (see above) -- without this, a
    near-flat single-stroke ink (bounding-box height close to 0) would demand
    an enormous scale factor, which blows up its width just as much since x
    and y share one factor.
    """
    ys = np.concatenate([s[1] for s in ink.strokes])
    ink_height = max(ys.max() - ys.min(), 1e-6)  # guards against a single-point/flat ink
    scale = min(target_height / ink_height, MAX_UPSCALE)
    return Ink(strokes=[s * scale for s in ink.strokes], annotations=ink.annotations)


def fit_canvas(ink: Ink, margin: int = MARGIN_PX):
    """Shifts the ink so its bounding box starts at (margin, margin), and
    measures the canvas width that snugly fits it (width is meant to vary --
    that's the whole point of not forcing a square canvas).

    Height is *not* measured from the bbox: every caller normalizes the
    ink's height to `TARGET_HEIGHT - 2*margin` via rescale_to_height
    immediately before calling this, so the post-margin height is exactly
    TARGET_HEIGHT by construction. Computing it from the bbox via ceil()
    (as width still is) would introduce spurious +-1px jitter from float
    rounding, defeating the point of a fixed-height pipeline.

    Returns (shifted_ink, canvas_width, canvas_height).
    """
    xs = np.concatenate([s[0] for s in ink.strokes])
    ys = np.concatenate([s[1] for s in ink.strokes])
    x_min, y_min = xs.min(), ys.min()

    new_strokes = [
        np.stack([s[0] - x_min + margin, s[1] - y_min + margin])
        for s in ink.strokes
    ]

    width = int(math.ceil(max(s[0].max() for s in new_strokes) + margin))
    height = TARGET_HEIGHT

    return Ink(strokes=new_strokes, annotations=ink.annotations), width, height


def _render_with_cairo(ink: Ink, width: int, height: int, stroke_width: float) -> Image.Image:
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    ctx = cairo.Context(surface)
    ctx.set_source_rgb(1, 1, 1)
    ctx.paint()
    ctx.set_source_rgb(0, 0, 0)
    ctx.set_line_width(stroke_width)
    ctx.set_line_cap(cairo.LineCap.ROUND)
    ctx.set_line_join(cairo.LineJoin.ROUND)

    for stroke in ink.strokes:
        if stroke.shape[1] == 1:
            x, y = stroke[0, 0], stroke[1, 0]
            ctx.arc(x, y, stroke_width / 2, 0, 2 * math.pi)
            ctx.fill()
        else:
            ctx.move_to(stroke[0, 0], stroke[1, 0])
            for x, y in stroke[:, 1:].T:
                ctx.line_to(x, y)
            ctx.stroke()

    size = (surface.get_width(), surface.get_height())
    stride = surface.get_stride()
    with surface.get_data() as memory:
        rgb_image = Image.frombuffer("RGB", size, memory.tobytes(), "raw", "BGRX", stride)
    return rgb_image.convert("L")


def _render_with_pillow(ink: Ink, width: int, height: int, stroke_width: float,
                         supersample: int = SUPERSAMPLE) -> Image.Image:
    """Anti-alias-by-supersampling fallback; needs no system libraries."""
    big = Image.new("L", (width * supersample, height * supersample), color=255)
    draw = ImageDraw.Draw(big)
    line_width = max(1, round(stroke_width * supersample))
    radius = line_width / 2

    for stroke in ink.strokes:
        pts = list(zip(stroke[0] * supersample, stroke[1] * supersample))
        if len(pts) == 1:
            x, y = pts[0]
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=0)
        else:
            draw.line(pts, fill=0, width=line_width, joint="curve")
            for x, y in pts:
                draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=0)

    return big.resize((width, height), Image.LANCZOS)


def render_ink(ink: Ink, width: int, height: int, *, stroke_width: float = STROKE_WIDTH_PX) -> Image.Image:
    if CAIRO_AVAILABLE:
        return _render_with_cairo(ink, width, height, stroke_width)
    return _render_with_pillow(ink, width, height, stroke_width)


def apply_affine_ink(ink: Ink, *, angle_deg: float = 0.0, shear_deg: float = 0.0) -> Ink:
    """Rotates/shears strokes around the ink's own bounding-box center."""
    xs = np.concatenate([s[0] for s in ink.strokes])
    ys = np.concatenate([s[1] for s in ink.strokes])
    center = np.array([[(xs.min() + xs.max()) / 2], [(ys.min() + ys.max()) / 2]])

    theta = math.radians(angle_deg)
    shear = math.radians(shear_deg)
    rotation = np.array([[math.cos(theta), -math.sin(theta)],
                         [math.sin(theta), math.cos(theta)]])
    shear_mat = np.array([[1.0, math.tan(shear)],
                          [0.0, 1.0]])
    transform = rotation @ shear_mat

    new_strokes = [transform @ (s - center) + center for s in ink.strokes]
    return Ink(strokes=new_strokes, annotations=ink.annotations)


def gaussian_blur(image: Image.Image, radius: float) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius))


def _roll(cfg: dict, rng: np.random.Generator):
    """Returns a sampled parameter value if this transform fires this draw, else None."""
    if rng.random() >= cfg["prob"]:
        return None
    low, high = cfg["range"]
    value = rng.uniform(low, high)
    return int(round(value)) if cfg["dtype"] is int else value


def render_with_augmentation(normalized_ink: Ink, rng: np.random.Generator):
    """Renders one (already height-normalized) ink with a fresh random draw of
    the configured augmentations. Returns (image, width, height, applied), where
    `applied` records which transforms fired and with what parameter -- useful
    for debugging or logging what a given training step actually saw."""
    applied = {}
    ink = normalized_ink

    angle = _roll(AUGMENTATION_CONFIG["rotation_deg"], rng)
    shear = _roll(AUGMENTATION_CONFIG["shear_deg"], rng)
    if angle is not None or shear is not None:
        ink = apply_affine_ink(ink, angle_deg=angle or 0.0, shear_deg=shear or 0.0)
        # Rotating/shearing changes the ink's own bounding-box height; re-normalize
        # back to a fixed height so only width absorbs the geometric change.
        ink = rescale_to_height(ink)
        if angle is not None:
            applied["rotation_deg"] = angle
        if shear is not None:
            applied["shear_deg"] = shear

    canvas_ink, width, height = fit_canvas(ink)
    image = render_ink(canvas_ink, width, height)

    blur_radius = _roll(AUGMENTATION_CONFIG["blur_radius"], rng)
    if blur_radius is not None:
        image = gaussian_blur(image, blur_radius)
        applied["blur_radius"] = blur_radius

    return image, width, height, applied


def tokenize_expression(s: str) -> list:
    r"""Splits a LaTeX math string into tokens, e.g. r'\frac{1}{2}' ->
    ['\\frac', '{', '1', '}', '{', '2', '}']."""
    tokens = []
    while s:
        if s[0] == "\\":
            tokens.append(_COMMAND_RE.match(s).group(0))
        else:
            tokens.append(s[0])
        s = s[len(tokens[-1]):]
    return tokens


def get_label_text(annotations: dict) -> str:
    """Prefer normalizedLabel; symbols/ inks only provide 'label'."""
    return annotations.get("normalizedLabel") or annotations["label"]
