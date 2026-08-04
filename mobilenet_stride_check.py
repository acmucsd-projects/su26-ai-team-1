"""
MobileNetV3 Feature Map & Stride Verification
===============================================================================

GOAL:
  This script inspects MobileNetV3-Large block-by-block to find the exact layer
  where feature downsampling hits Stride-16 (height drops from 64px to 4px).

PIPELINE CONTEXT:
  - Input Shape: [Batch, 3, 64, Width] 
    (Matches Jaeho's preprocessing pipeline: fixed height of 64px, variable width).
  - Target Stride: Stride-16 (h=4px). Stride-32 (h=2px) compresses the image 
    too much, losing fine math symbol details.

FINDINGS:
  - Blocks 7 through 12 output Stride-16 features (112 channels, height = 4px).
  - Block 13 drops to Stride-32 (height = 2px).
  - CONCLUSION: We truncate MobileNet at index 12 (`features[:13]`).

===============================================================================
"""

import torch
import torchvision.models as models

INPUT_HEIGHT = 64    # CONFIRMED fixed height
INPUT_WIDTH = 256     # just a representative sample width for this check --
                       # real batches vary, this doesn't need to be exact

model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
model.eval()

x = torch.randn(1, 3, INPUT_HEIGHT, INPUT_WIDTH)

print(f"Input: {tuple(x.shape)}\n")
print(f"{'idx':>4} | {'output shape (C,H,W)':<24} | height stride")
print("-" * 55)

with torch.no_grad():
    out = x
    for i, block in enumerate(model.features):
        out = block(out)
        h = out.shape[-2]
        stride = INPUT_HEIGHT / h
        flag = ""
        if abs(stride - 16) < 0.5:
            flag = "  <-- stride 16 (h=4, current plan)"
        elif abs(stride - 32) < 0.5:
            flag = "  <-- stride 32 (h=2, the alternative)"
        print(f"{i:>4} | {str(tuple(out.shape)):<24} | {stride:.1f}{flag}")