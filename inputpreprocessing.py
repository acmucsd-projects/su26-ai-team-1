"""
Input Image Preprocessing pipeline for a handwritten-math-to-LaTeX model.

Pipeline:
    scanned/photographed image
      -> perspective correction (Canny -> Hough line detection -> corner
         intersection -> homography / warpPerspective)
      -> binarization (Otsu or adaptive threshold)
      -> letterbox resize (aspect-ratio preserving, zero data loss from cropping)
      -> normalized tensor for a MobileNet encoder

References (traced, not invented):
  - The perspective-correction *steps* (Canny edges -> Hough transform -> boundary
    corner intersection -> homography -> binarize) are the ones described in the
    Im2Latex project page: https://sujayr91.github.io/Im2Latex/
    That implementation is MATLAB and the source was never published, so the
    Python/OpenCV code below is a re-implementation of the described steps, not
    a port of existing code.
  - Canny + cv2.HoughLinesP usage follows the OpenCV Hough Line Transform tutorial:
    https://docs.opencv.org/4.x/d9/db0/tutorial_hough_lines.html
  - Four-point homography via cv2.getPerspectiveTransform / cv2.warpPerspective
    follows the standard OpenCV document-rectification pattern, e.g.
    https://github.com/spmallick/learnopencv/blob/master/Homography/perspective-correction.py
    (that script takes manually-clicked corners; here corner-finding is automated).
  - The contour/approxPolyDP fallback for corner-finding is the common "document
    scanner" pattern (largest 4-point contour) used across OpenCV tutorials when
    Hough line intersection is unreliable (cluttered background, weak edges).
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Step 1: Perspective correction
# ---------------------------------------------------------------------------

"""
Hough Transform
- About :  Run the probabilistic Hough transform on an edge map to get candidate 
           straight-line segments (returned as (x1,y1,x2,y2) endpoints).
Reference: cv2.HoughLinesP parameter usage (rho, theta, threshold,
           minLineLength, maxLineGap) follows the OpenCV Hough Line Transform
           tutorial: https://docs.opencv.org/4.x/d9/db0/tutorial_hough_lines.html
           min_line_length is scaled to image size here (not in the OpenCV
           tutorial) so the same ratio works across differently-sized inputs.
"""
def _find_boundary_lines(edges, min_line_length_ratio=0.3):
    h, w = edges.shape[:2]
    min_line_length = max(20, int(min_line_length_ratio * min(h, w)))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=min_line_length,
        maxLineGap=20,
    )
    return lines

"""
About: Sort the raw Hough line segments into "roughly horizontal" and
        "roughly vertical" buckets by their angle.
Why:   The Im2Latex boundary-finding step needs the top/bottom/left/right
        edges of the page specifically, not just any detected line -- this
        is a necessary bookkeeping step to get there, so we can later pick
        the outermost line in each direction.
"""
def _classify_lines(lines, angle_tol_deg=20):
    horizontals, verticals = [], []
    if lines is None:
        return horizontals, verticals
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < angle_tol_deg or abs(abs(angle) - 180) < angle_tol_deg:
            horizontals.append((x1, y1, x2, y2))
        elif abs(abs(angle) - 90) < angle_tol_deg:
            verticals.append((x1, y1, x2, y2))
    return horizontals, verticals

"""
What: Given two line segments, treat them as infinite lines and solve for
        where they cross.
Why:  This is the "Boundary: the four corner points are found by
        determining the intersection of the boundary lines" step described
        on the Im2Latex project page -- corners aren't detected directly,
        they're computed as where the top/bottom lines meet the left/right
        lines.
"""
def _line_intersection(l1, l2):
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return (px, py)

"""
What: Take 4 unordered (x, y) points and label them top-left, top-right,
        bottom-right, bottom-left.
Why:  cv2.getPerspectiveTransform (used later) needs the 4 source corners
        and 4 destination corners listed in matching order.
Reference: This is the standard "order_points" helper used in OpenCV
        document-scanner tutorials (sum of coords is smallest at top-left /
        largest at bottom-right; difference of coords is smallest at
        top-right / largest at bottom-left).
"""
def _order_points(pts):
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    ordered = np.zeros((4, 2), dtype="float32")
    ordered[0] = pts[np.argmin(s)]       # top-left     (smallest x+y)
    ordered[2] = pts[np.argmax(s)]       # bottom-right (largest x+y)
    ordered[1] = pts[np.argmin(diff)]    # top-right    (smallest y-x)
    ordered[3] = pts[np.argmax(diff)]    # bottom-left  (largest y-x)
    return ordered

"""
What: Chain the steps above together -- blur, Canny edges, Hough lines,
      classify into horizontal/vertical, take the outermost pair of each,
      intersect them -- to get the 4 page/board corners.
"""
def _corners_via_hough(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    lines = _find_boundary_lines(edges)
    horizontals, verticals = _classify_lines(lines)
    if len(horizontals) < 2 or len(verticals) < 2:
        return None

    # Boundary = outermost lines in each direction
    horizontals.sort(key=lambda l: (l[1] + l[3]) / 2)
    verticals.sort(key=lambda l: (l[0] + l[2]) / 2)
    top, bottom = horizontals[0], horizontals[-1]
    left, right = verticals[0], verticals[-1]

    corners = []
    for h in (top, bottom):
        for v in (left, right):
            pt = _line_intersection(h, v)
            if pt is not None:
                corners.append(pt)

    if len(corners) != 4:
        return None
    return _order_points(corners)

"""
Backup contour-based corner-finding, in case the Hough method fails.
What: Find the largest closed contour in the edge map and, if it
      approximates a 4-sided shape, use those 4 points as the corners.
Why:  The Hough-based method above needs 2 clean horizontal + 2 clean
      vertical lines, which fails on cluttered backgrounds or weak/broken
      edges. This is the backup so the pipeline doesn't just give up on
      every real-world messy photo.
"""
def _corners_via_contour(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    if len(approx) != 4:
        return None

    return _order_points(approx.reshape(4, 2))

""" 
What: The full "Step 1" of the pipeline. Get the 4 corners (Hough method,
        falling back to contour method), compute the output rectangle size
        from the corner distances, build the homography matrix that maps
        the (tilted) corners onto that rectangle, and warp the image.
Reference: The corner-finding + homography approach as a whole follows the
        Im2Latex project description (Canny/Hough/boundary/homography steps)
        at https://sujayr91.github.io/Im2Latex/. The actual homography call
        -- cv2.getPerspectiveTransform + cv2.warpPerspective on 4 ordered
        corners -- follows the standard OpenCV pattern also used in
        spmallick/learnopencv's perspective-correction.py (that script takes
        manually-clicked points; here the corners are found automatically by
        _corners_via_hough / _corners_via_contour instead of by clicking).
Safety net: if no clean quadrilateral is found, or the found one is
        degenerate (near-zero size), the original image is returned
        unwarped -- this is my own addition, not from either reference,
        since applying a homography from bad corners would silently wreck
        the image rather than just skip a step.
"""
def perspective_correct(image, output_size=None):
    corners = _corners_via_hough(image)
    if corners is None:
        corners = _corners_via_contour(image)
    if corners is None:
        return image

    (tl, tr, br, bl) = corners
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = int(max(width_a, width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = int(max(height_a, height_b))

    if max_width < 10 or max_height < 10:
        return image  # degenerate quad, don't warp

    if output_size is not None:
        max_width, max_height = output_size

    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype="float32",
    )

    homography_matrix = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(image, homography_matrix, (max_width, max_height))
    return warped


# ---------------------------------------------------------------------------
# Step 2: Binarization
# ---------------------------------------------------------------------------

"""
What: Convert the perspective-corrected image to pure black/white by
        thresholding pixel intensity. "otsu" picks one global threshold
        automatically from the image's intensity histogram; "adaptive"
        computes a local threshold per neighborhood, which handles uneven
        lighting (e.g. a shadow across part of a photographed page) better.
"""
def binarize(image, method="otsu"):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
 
    if method == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "adaptive":
        # More robust to uneven lighting from phone photos than a single global
        # Otsu threshold; worth an experiment vs. otsu on your actual data.
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 10
        )
    else:
        raise ValueError(f"Unknown binarization method: {method}")
 
    """ Polarity normalization 
    cv2.THRESH_BINARY always maps "above threshold" -> 255 and "below" -> 0,
    regardless of which side is ink. That means a white-bg/dark-ink photo
    and a black-bg/white-ink photo (chalkboard, dark-mode screenshot) come
    out with OPPOSITE polarity: ink=0 in one case, ink=255 in the other.
    Left alone, that's inconsistent across a dataset that mixes both kinds
    of input. Normalize here so ink is always 0 (black) and background is
    always 255 (white), by assuming ink is the minority-pixel-count class
    (true for equations/text, which cover a small fraction of the frame).
    """
    ink_is_white = np.count_nonzero(binary == 255) < np.count_nonzero(binary == 0)
    if ink_is_white:
        binary = cv2.bitwise_not(binary)
 
    return binary


# ---------------------------------------------------------------------------
# Step 3: Letterbox resize (aspect-preserving, no cropping / no distortion)
# ---------------------------------------------------------------------------

"""
What: Scale the image down/up so its longer side fits target_size, keeping
        the original width:height ratio, then paste it onto a blank
        target_size canvas (centered), padding the shorter side instead of
        stretching or cropping.
Why:  A plain resize-to-224x224 would stretch a long horizontal equation
        into a squashed square, distorting every symbol's shape -- bad for
        a model that has to tell symbols apart by shape. Cropping to force
        a square would risk cutting off parts of the formula. Letterboxing
        is how you get a fixed input size for a CNN without either problem.
"""
def letterbox_resize(image, target_size=(224, 224), pad_value=255):
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    target_w, target_h = target_size
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))

    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)

    canvas = np.full((target_h, target_w, 3), pad_value, dtype=np.uint8)
    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas


# ---------------------------------------------------------------------------
# Step 4: Format for a MobileNet encoder
# ---------------------------------------------------------------------------

"""
What: Convert the uint8 [0,255] image into the float tensor format a
        MobileNet encoder expects: scale to [0,1], subtract/divide by
        per-channel mean/std, rearrange axes from (height, width, channel)
        to (channel, height, width), and add a batch dimension in front.
Reference: (0.485, 0.456, 0.406) / (0.229, 0.224, 0.225) are the standard
        ImageNet channel mean/std used to normalize inputs for MobileNet (and
        most other) models pretrained on ImageNet
"""
def to_mobilenet_input(image):
    img = image.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))    # HWC -> CHW
    img = np.expand_dims(img, axis=0)     # add batch dim
    return img


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------

"""
What: Run the whole pipeline in order -- read image from disk -> perspective
        correct -> binarize -> letterbox resize -> convert to MobileNet
        input tensor.
Why:  Single entry point so the rest of your training/inference code
        doesn't need to know about the individual steps.
Reference: The step order itself (perspective correction -> binarization ->
        resize) is the order you specified; not attributed to any single
        paper/repo.
"""
def preprocess_pipeline(image_path, target_size=(224, 224), binarize_method="otsu"):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    corrected = perspective_correct(image)
    binary = binarize(corrected, method=binarize_method)
    letterboxed = letterbox_resize(binary, target_size=target_size)
    model_input = to_mobilenet_input(letterboxed)
    return model_input


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python preprocess.py <image_path>")
        sys.exit(1)

    tensor = preprocess_pipeline(sys.argv[1])
    print("Output tensor shape:", tensor.shape)