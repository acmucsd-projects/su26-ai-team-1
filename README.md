# Handwritten Math Input Image Preprocessing Pipeline

`inputpreprocessing.py` is a preprocessing pipeline for input images in our handwritten-math-to-LaTeX model.
Takes a scanned or camera-photographed image of a handwritten equation and produces a normalized tensor ready for a MobileNet encoder.

## Pipeline

```
input image (scan / photo)

-> find the location of equations in the image

-> identifying the edges of writing surface(paper, whiteboard, post-it note, etc.)
   (surface quadrilateral when available; otherwise vanishing-point partial
   rectification when only one direction of page edges is reliable)

-> crop to the equation's ink extent by removing unncessary surrounding blank space

-> binarization
   (Otsu or adaptive threshold, polarity-normalized)

-> resize to `64 x W`
   (height is always 64; width is proportional and remains variable)

-> MobileNet input tensor
   (normalized, CHW, batched)
```

| Step                   | Problem it solves                                                                                                                                                                                       |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Perspective correction | Corrects photographed paper when reliable page/surface geometry exists. Clean MathWriting-style white canvases are detected and deliberately left unwarped, since their pen strokes are not page edges. |
| Ink crop               | Removes photo/page whitespace while retaining disconnected symbols in one expression.                                                                                                                   |
| Binarization           | Removes paper texture, lighting gradients, and camera noise/color, leaving just the ink/marker strokes.                                                                                                 |
| Height-only resize     | A fixed square would stretch wide equations and distort symbols. The pipeline uses `64 x W`; use `pad_mobilenet_batch` only when batching samples with different widths.                                |
| MobileNet formatting   | Converts the image array into the float tensor shape a MobileNet encoder expects.                                                                                                                       |

## References

### Perspective correction

- **Source of the _approach_:** the [Im2Latex project page](https://sujayr91.github.io/Im2Latex/)
  describes correcting perspective distortion: Canny edge detection → Hough transform to find
  the clipboard/page boundary lines → intersecting those lines for the 4
  corners → homography to warp the corners into a rectangle → binarize.

- **Canny + `cv2.HoughLinesP` usage:** follows the
  [OpenCV Hough Line Transform tutorial](https://docs.opencv.org/4.x/d9/db0/tutorial_hough_lines.html).

- **Four-point homography (`cv2.getPerspectiveTransform` + `cv2.warpPerspective`):**
  standard OpenCV document-rectification pattern, e.g.
  [learnopencv's perspective-correction.py](https://github.com/spmallick/learnopencv/blob/master/Homography/perspective-correction.py)

### Binarization

- **Otsu (`cv2.THRESH_OTSU`) and adaptive thresholding
  (`cv2.ADAPTIVE_THRESH_GAUSSIAN_C`):** standard OpenCV binarization methods,
  in general use across OCR preprocessing.
