"""
MobileNetV3 Visual Encoder
===============================================================================
GOAL:
  Extracts visual feature representations from handwritten math images and converts
  them into a 2D feature map for the Transformer Decoder.

ARCHITECTURE OVERVIEW:
  1. Backbone: Pre-trained MobileNetV3-Large, truncated by `cutoff`.
     - Input: Raw image tensor [Batch, 3, 96, Width]   (hmer_model.IMAGE_HEIGHT)
     - cutoff=13 (default): stride 16 -> [Batch, 112, 6, ceil(Width/16)]
     - cutoff=7:            stride 8  -> [Batch,  40, 12, ceil(Width/8)]
  2. Projection: 1x1 Convolution mapping the backbone channels to d_model (256).
  3. Output: Returns the raw 2D feature map [Batch, d_model, H/stride, W/stride]
     -- NOT flattened. The decoder side applies its own positional encoding and
     flattening (see ImagePositionalEncoding in baseline_decoder.py).

PADDING-AWARE SQUEEZE-EXCITATION:
  Images are padded to the widest sample in the batch. Every one of MobileNetV3's
  Squeeze-Excitation blocks global-average-pools over the WHOLE feature map, so
  padding used to dilute that average and rescale the channels on the real ink --
  making predictions depend on what else happened to be in the batch (ExpRate
  moved 1.7 points between eval batch sizes; see docs/RESULTS.md). Pass
  `true_widths` to forward() and each SE pools over the real columns only.
  Measured on real columns: ~100x less variation across batch padding widths.
  What remains is the convs' receptive field reaching the far end of the padded
  tensor, which is small by comparison.
===============================================================================
"""

import math

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.ops.misc import SqueezeExcitation

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class _PaddingState:
    """Shared by all MaskedSE blocks: this batch's true widths (or None)."""

    def __init__(self):
        self.true_widths = None   # [B] pixel widths before padding
        self.input_width = None   # padded input width in pixels

    def column_mask(self, feat):
        """[B, 1, 1, W'] float, 1 on real columns, for a feature map `feat`."""
        if self.true_widths is None:
            return None
        w = feat.shape[-1]
        # Stride at this depth is a power of two; snap so odd input widths
        # still land on the same column count the decoder's mask uses.
        stride = 2 ** round(math.log2(self.input_width / w))
        valid = torch.ceil(self.true_widths.to(feat.device).float() / stride).clamp(min=1, max=w)
        cols = torch.arange(w, device=feat.device)
        return (cols.unsqueeze(0) < valid.unsqueeze(1)).to(feat.dtype)[:, None, None, :]


class MaskedSqueezeExcitation(nn.Module):
    """Drop-in for torchvision's SqueezeExcitation that pools over real columns only.

    Reuses the original fc1/fc2 modules, so pretrained weights and state_dict
    keys (`...block.2.fc1.weight`, ...) are unchanged.
    """

    def __init__(self, se: SqueezeExcitation, state: _PaddingState):
        super().__init__()
        self.fc1, self.fc2 = se.fc1, se.fc2
        self.activation, self.scale_activation = se.activation, se.scale_activation
        self._state = state

    def forward(self, x):
        mask = self._state.column_mask(x)
        if mask is None:
            pooled = x.mean(dim=(2, 3), keepdim=True)
        else:
            count = (mask.sum(dim=3, keepdim=True) * x.shape[2]).clamp(min=1)
            pooled = (x * mask).sum(dim=(2, 3), keepdim=True) / count
        scale = self.scale_activation(self.fc2(self.activation(self.fc1(pooled))))
        return scale * x


class MobileNetEncoder(nn.Module):
    def __init__(self, d_model=256, cutoff=13, mask_padding=True):
        """
        cutoff: how many of MobileNetV3-Large's blocks to keep (mobilenet.features[:cutoff]).
            13 (default) -> stride-16, feat_h=6, 112 channels -- the original setting.
             7            -> stride-8,  feat_h=12, 40 channels -- doubles vertical
                             resolution from the SAME 96px input, aimed at case/
                             position confusions the stride-16 grid was too coarse
                             to resolve (see mobilenet_stride_check.py for the
                             full per-block shape table this was read off of).
        Changing this changes the encoder's output height, so it is NOT
        checkpoint-compatible across different cutoff values -- feat_h and the
        positional-encoding table size in HMERModel must match.

        mask_padding: pool Squeeze-Excitation over real columns only when
            forward() is given true_widths. No new parameters, so checkpoints
            load either way -- but a model trained with diluted SE was tuned to
            it, so flipping this on an old checkpoint shifts its predictions.
        """
        super().__init__()
        mobilenet = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        self.backbone = nn.Sequential(*mobilenet.features[:cutoff])

        self._padding = _PaddingState()
        self.mask_padding = mask_padding
        if mask_padding:
            for block in self.backbone.modules():
                for name, child in block.named_children():
                    if isinstance(child, SqueezeExcitation):
                        setattr(block, name, MaskedSqueezeExcitation(child, self._padding))

        # Channel count of the last kept block, read off the layer instead of
        # hardcoded (112 at cutoff=13, 40 at cutoff=7).
        out_channels = next(m.out_channels for m in reversed(list(self.backbone.modules()))
                            if isinstance(m, nn.Conv2d))
        self.projection = nn.Conv2d(in_channels=out_channels, out_channels=d_model, kernel_size=1)

    def forward(self, x, true_widths=None):
        # x: [batch, 3, H, W], padded to the batch's widest image.
        # true_widths: [batch] real pixel widths, so SE ignores the padding.
        if self.mask_padding and true_widths is not None:
            self._padding.true_widths = torch.as_tensor(true_widths)
            self._padding.input_width = x.shape[-1]
        try:
            features = self.backbone(x)          # [batch, C, H/stride, W/stride]
        finally:
            self._padding.true_widths = None
        return self.projection(features)         # [batch, d_model, H/stride, W/stride]


# Test run
if __name__ == "__main__":
    print(f"Using device: {device}")
    dummy_input = torch.randn(2, 3, 96, 256).to(device)
    encoder = MobileNetEncoder(d_model=256).to(device)
    output = encoder(dummy_input)

    print("Input shape: ", dummy_input.shape)
    print("Encoder output shape:", output.shape)
    # Target shape: [2, 256, 6, 16]  --> 2D feature map (d_model 256, height 6, width 16)

    # Batch-invariance: a sample's real columns must not depend on how much
    # OTHER padding it shares a batch with (padding itself is constant white).
    encoder.eval()
    ink = torch.rand(1, 3, 96, 96).to(device)
    pad = lambda w: torch.cat([ink, torch.ones(1, 3, 96, w - 96).to(device)], dim=3)
    tw = torch.tensor([96])
    with torch.no_grad():
        n = 96 // 16                          # the real feature columns
        masked = [encoder(pad(w), tw)[..., :n] for w in (128, 800)]
        plain = [encoder(pad(w))[..., :n] for w in (128, 800)]
    print(f"128px vs 800px batch, max diff, real columns, masked: {(masked[0] - masked[1]).abs().max():.2e}")
    print(f"128px vs 800px batch, max diff, real columns, plain:  {(plain[0] - plain[1]).abs().max():.2e}")
