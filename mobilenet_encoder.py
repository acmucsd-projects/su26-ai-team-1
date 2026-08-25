"""
MobileNetV3 Visual Encoder
===============================================================================
GOAL:
  Extracts visual feature representations from handwritten math images and converts 
  them into a sequence of feature vectors for the Transformer Decoder.

ARCHITECTURE OVERVIEW:
  1. Backbone: Pre-trained MobileNetV3-Large truncated at Block 12 (Stride-16).
     - Input: Raw image tensor [Batch, 3, 64, Width]
     - Backbone Output: Feature map [Batch, 112, 4, Width/16]
  2. Projection: 1x1 Convolution mapping 112 channels to d_model (default 256).
  3. Output: Returns the raw 2D feature map [Batch, d_model, 4, Width/16] --
     NOT flattened. Adam's decoder applies its own positional encoding and
     flattening (see ImagePositionalEncoding in baseline_decoder.py).

PIPELINE INTEGRATION:
  - Upstream: Accepts preprocessed grayscale/RGB images from Jaeho's pipeline.
  - Downstream: Feeds the 2D feature map into Adam's ImagePositionalEncoding,
    which handles flattening internally.
===============================================================================
"""

import torch
import torch.nn as nn
import torchvision.models as models

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class MobileNetEncoder(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        # 1. Load pre-trained MobileNetV3-Large
        mobilenet = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)

        # 2. Extract features up to index 12 (stride-16 end point)
        self.backbone = nn.Sequential(*mobilenet.features[:13])

        # 3. Figure out how many channels come out of the backbone automatically
        #    instead of hardcoding 112 -- so changing the cutoff index above is
        #    the only edit needed if we want to test stride-32 or a different layer

        self.backbone.eval()
        with torch.no_grad():
            dummy = torch.randn(1, 3, 64, 256)
            out_channels = self.backbone(dummy).shape[1]

        # 4. Project backbone's channel count -> Transformer d_model
        self.projection = nn.Conv2d(in_channels=out_channels, out_channels=d_model, kernel_size=1)


    def forward(self, x):
        # x shape: [batch_size, 3, H, W]; it's the actual image data flowing thru, one batch at a time
        features = self.backbone(x)              # shape: [batch_size, C, H/16, W/16]
        features = self.projection(features)     # shape: [batch_size, d_model, H/16, W/16]
        return features


# Test run 
if __name__ == "__main__":
    print(f"Using device: {device}")                              
    dummy_input = torch.randn(2, 3, 64, 256).to(device)           
    encoder = MobileNetEncoder(d_model=256).to(device)            
    output = encoder(dummy_input)

    print("Input shape: ", dummy_input.shape)
    print("Encoder output shape:", output.shape)
    # Target shape: [2, 256, 4, 16]  --> 2D feature map (d_model 256, height 4, width 16)