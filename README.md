# Handwritten Math Input Image Preprocessing Pipeline

`inputpreprocessing.py` is a preprocessing pipeline for input images in our handwritten-math-to-LaTeX model.
Takes a scanned or camera-photographed image of a handwritten equation and produces a normalized tensor ready for a MobileNet encoder.

## Pipeline

```
input image (scan / photo)

-> perspective correction
   (Canny -> Hough -> corner intersection -> homography)

-> binarization
   (Otsu or adaptive threshold, polarity-normalized)

-> letterbox resize
   (aspect-ratio preserving, padded to 224x224)

-> MobileNet input tensor
   (normalized, CHW, batched)
```

| Step                   | Problem it solves                                                                                                                                                                                    |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Perspective correction | Correcting images taken at non-flat angle distorts equations, which would confuse an encoder trained on straight-on images.                                                                          |
| Binarization           | Removes paper texture, lighting gradients, and camera noise/color, leaving just the ink/marker strokes.                                                                                              |
| Letterbox resize       | A plain resize to a fixed square would stretch a wide equation and distort symbol shapes; cropping risks cutting off part of the formula. Letterboxing gets a fixed input size with neither problem. |
| MobileNet formatting   | Converts the image array into the float tensor shape a MobileNet encoder expects.                                                                                                                    |

## References

### Perspective correction

- **Source of the _approach_:** the [Im2Latex project page](https://sujayr91.github.io/Im2Latex/)
  (sujayr91) describes correcting perspective distortion: Canny edge detection → Hough transform to find
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
