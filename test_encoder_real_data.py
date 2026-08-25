"""
Encoder Test on Real MathWriting Data
===============================================================================
GOAL:
  Verify MobileNetEncoder works on actual preprocessed handwritten math images
  from Ryan's dataset, not just random dummy tensors.

WHAT THIS CHECKS:
  1. Real images load and pass through the encoder without shape errors.
  2. Output height is always 4 (64px input / stride 16), confirming the
     stride-16 cutoff behaves as expected on real data.
  3. Output width varies with each image's width, confirming the encoder
     handles variable-width inputs (which is the whole point of the
     fixed-height / variable-width preprocessing design).

NOTE ON CHANNELS:
  Ryan's pipeline outputs GRAYSCALE (1-channel) PNGs, but MobileNetV3 was
  pretrained on RGB (3-channel) photos, so its first layer expects 3 channels.
  We repeat the single gray channel 3x to match. This is standard practice
  when feeding grayscale into an RGB-pretrained backbone.

  Longer term the team may want to discuss whether to keep this repeat trick
  or modify the first conv layer to accept 1 channel directly -- the repeat is
  simpler and keeps the pretrained weights intact, so it's the safer default.
===============================================================================
"""

from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms as T

from mobilenet_encoder import MobileNetEncoder, device

# Point this at wherever the unzipped dataset lives
DATA_DIR = Path(__file__).resolve().parent.parent / "processed" / "images" / "valid"
NUM_SAMPLES = 5  # how many images to test


def load_image(path):
    """
    Load one preprocessed PNG and turn it into a tensor the encoder can read.

    Steps:
      1. Open the image and force grayscale mode ("L" = luminance, 1 channel).
      2. Convert to a tensor -- this also scales pixel values from 0-255
         down to 0.0-1.0, which is what the pretrained model expects.
      3. Repeat the single gray channel 3 times to fake an RGB image,
         since MobileNet's first layer expects 3 channels.
      4. Normalize using ImageNet's mean/std -- MobileNet was pretrained on
         images centered around these specific values, so we match that here.
      5. Add a batch dimension at the front, since the model always expects
         a batch even if it's a batch of one.
    """
    img = Image.open(path).convert("L")
    tensor = T.ToTensor()(img)            # shape: [1, H, W], scaled to 0.0-1.0
    tensor = tensor.repeat(3, 1, 1)       # shape: [3, H, W]
    tensor = T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )(tensor)                             # shape: [3, H, W], normalized
    return tensor.unsqueeze(0)            # shape: [1, 3, H, W]


if __name__ == "__main__":
    print(f"Using device: {device}")
    print(f"Loading images from: {DATA_DIR}\n")

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_DIR}. Update DATA_DIR to point at wherever "
            f"you unzipped Ryan's processed dataset."
        )

    encoder = MobileNetEncoder(d_model=256).to(device)
    encoder.eval()  # not training here, just checking it runs

    image_paths = sorted(DATA_DIR.glob("*.png"))[:NUM_SAMPLES]
    if not image_paths:
        raise FileNotFoundError(f"No .png files found in {DATA_DIR}")

    print(f"{'filename':<24} | {'input shape':<22} | {'output shape'}")
    print("-" * 78)

    with torch.no_grad():  # no training, so skip gradient tracking
        for path in image_paths:
            x = load_image(path).to(device)
            out = encoder(x)
            print(f"{path.name:<24} | {str(tuple(x.shape)):<22} | {tuple(out.shape)}")

    print("\nSanity checks:")
    print("  - Output height should be 4 for every image (64px / stride 16).")
    print("  - Output width should differ per image, since widths vary.")
    print("  - If any input height is NOT 64, flag it to the team -- the")
    print("    preprocessing pipeline is supposed to guarantee a fixed height.")