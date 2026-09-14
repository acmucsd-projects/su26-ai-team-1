# Handwritten Math Image Preprocessing

`test_inputpreprocessing_64.py` and `test_inputpreprocessing_96.py` are image-output scripts for visually evaluating preprocessing of photographed or scanned handwritten equations. Their purpose is to produce viewable, cropped, orientation-corrected, binarized, and resized images before integrating preprocessing with the handwritten-math-to-LaTeX encoder.

## Output

| Image-output script             | Default image size (height × width) | Python return value                             |
| ------------------------------- | ----------------------------------- | ----------------------------------------------- |
| `test_inputpreprocessing_64.py` | `64 × W`                            | Binary `uint8` image array with shape `(64, W)` |
| `test_inputpreprocessing_96.py` | `96 × W`                            | Binary `uint8` image array with shape `(96, W)` |

Width scales proportionally to the processed crop's height, preserving its aspect ratio. The scripts save PNG images by default, with black ink and a white background. They return image arrays rather than normalized, batched MobileNet tensors. Height can be overridden with `--height`.

## Pipeline

```text
Input photo
  → Locate the equation region and group compatible nearby expression rows
  → Estimate the writing surface and apply perspective correction when accepted
  → Crop the equation region with padding
  → Correct sideways orientation or estimate in-plane deskew
  → Binarize using Otsu or adaptive thresholding
  → Resize to height 64 or 96 with proportional width
  → Threshold again to remove gray pixels introduced by resizing
  → Return the processed image and save it to an image file
```

Perspective correction uses surface quadrilaterals, guided page-edge geometry, or vanishing-direction fallbacks. It can leave an image unwarped when no reliable correction is found. Rotation and equation detection are heuristic; output quality must be checked visually.

## Test your own images

### 1. Set up the project

Open a terminal in the `ACM AI Test` folder and install the dependencies:

```bash
cd "/Users/jh0_726/Desktop/Project/ACM AI Test"
python3 -m pip install -r requirements.txt
```

Replace the project path if you saved this folder elsewhere. Python 3 is required.

### 2. Add your images

Copy your photos or scans into `test_images/`. This is the input folder; generated results belong in `test_64/` and `test_96/`. Keep your own filenames, such as `my_equation.jpg`, or use numbered names such as `test13.JPG`. Give new images unique names to preserve the bundled examples.

### 3. Run one image

Replace `my_equation.jpg` with the exact filename you added:

```bash
python3 test_inputpreprocessing_64.py "test_images/my_equation.jpg" --output "test_64/test_13_result.png"
python3 test_inputpreprocessing_96.py "test_images/my_equation.jpg" --output "test_96/test_13_result.png"
```

Choose an unused result number and use the same number for both sizes. Each command processes your image and saves the result, creating the output directory if needed. Rerunning overwrites the chosen output file; the original image remains unchanged.

### 4. Or run all images

Run this entire block from the project folder to process every supported image directly inside `test_images/`, including the bundled examples. It accepts JPG, JPEG, PNG, BMP, TIFF, and WebP files.

```bash
python3 - <<'PY'
from pathlib import Path
import re
import subprocess
import sys

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def sort_key(path):
    return [int(part) if part.isdigit() else part
            for part in re.split(r"(\d+)", path.name.lower())]

images = sorted(
    (path for path in Path("test_images").iterdir()
     if path.is_file() and path.suffix.lower() in extensions),
    key=sort_key,
)
if not images:
    raise SystemExit("Add images to test_images before running this command.")

for number, source in enumerate(images, start=1):
    for height in (64, 96):
        output = Path(f"test_{height}") / f"test_{number}_result.png"
        print(f"{source} -> {output}", flush=True)
        subprocess.run(
            [sys.executable, f"test_inputpreprocessing_{height}.py",
             str(source), "--output", str(output)],
            check=True,
        )
print(f"Done: processed {len(images)} images at both heights.")
PY
```

Images are sorted by filename with numeric parts in numerical order, then numbered starting at 1. The command prints the source-to-result mapping and uses the same number for both output heights. It stops and displays an error if processing fails.

Rerunning overwrites matching numbered results. Adding or removing inputs can change their assigned numbers; older outputs with other numbers remain in the result folders.

### 5. Check the generated images

| Folder         | Contents                                                        |
| -------------- | --------------------------------------------------------------- |
| `test_images/` | Your original photos and scans                                  |
| `test_64/`     | `test_1_result.png`, etc., at height 64 with proportional width |
| `test_96/`     | Matching numbered PNGs at height 96 with proportional width     |

Compare matching outputs with the source image printed by the batch command. Check that every symbol is visible, the crop contains the whole equation, and the writing is correctly oriented. Correct dimensions and binary pixel values alone do not guarantee readable equations.
