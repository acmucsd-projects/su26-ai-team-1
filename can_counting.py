"""Simplified CAN auxiliary symbol-counting branch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from baseline_decoder import BOS_IDX, EOS_IDX, PAD_IDX


class CountingModule(nn.Module):
    """Predict a non-negative count for every vocabulary token."""

    def __init__(self, d_model, vocab_size, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model, vocab_size),
            nn.Softplus(),
        )

    def forward(self, features, padding_mask=None):
        """Pool valid MobileNet features and return ``[batch, vocab_size]``."""
        if features.dim() == 4:
            features = features.flatten(2).transpose(1, 2)
        if features.dim() != 3:
            raise ValueError(
                f"Expected [B, S, C] or [B, C, H, W], got {tuple(features.shape)}"
            )

        if padding_mask is None:
            pooled = features.mean(dim=1)
        else:
            if padding_mask.shape != features.shape[:2]:
                raise ValueError(
                    "padding_mask must have shape [B, S], got "
                    f"{tuple(padding_mask.shape)} for features {tuple(features.shape)}"
                )
            valid = (~padding_mask).unsqueeze(-1).to(features.dtype)
            pooled = (features * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        return self.net(pooled)


def symbol_count_targets(
    tokens, vocab_size, ignored_indices=(PAD_IDX, BOS_IDX, EOS_IDX)
):
    """Convert ``[B, T]`` token IDs into bag-of-symbol count targets."""
    if tokens.dim() != 2:
        raise ValueError(f"Expected tokens with shape [B, T], got {tuple(tokens.shape)}")
    if tokens.numel() and (tokens.min() < 0 or tokens.max() >= vocab_size):
        raise ValueError("Token ID is outside the configured vocabulary")

    targets = torch.zeros(
        tokens.size(0), vocab_size, device=tokens.device, dtype=torch.float32
    )
    keep = torch.ones_like(tokens, dtype=torch.bool)
    for idx in ignored_indices:
        keep &= tokens.ne(idx)
    targets.scatter_add_(1, tokens, keep.to(targets.dtype))
    return targets


def counting_loss(predictions, tokens):
    """Smooth-L1 loss between predicted and ground-truth symbol counts."""
    if predictions.dim() != 2:
        raise ValueError(
            f"Expected predictions with shape [B, V], got {tuple(predictions.shape)}"
        )
    targets = symbol_count_targets(tokens, predictions.size(1))
    return F.smooth_l1_loss(predictions, targets)


if __name__ == "__main__":
    torch.manual_seed(0)
    vocab_size = 12
    tokens = torch.tensor([
        [BOS_IDX, 4, 4, 5, EOS_IDX, PAD_IDX],
        [BOS_IDX, 6, EOS_IDX, PAD_IDX, PAD_IDX, PAD_IDX],
    ])
    targets = symbol_count_targets(tokens, vocab_size)
    assert targets[0, 4].item() == 2
    assert targets[0, 5].item() == 1
    assert targets[1, 6].item() == 1
    assert targets[:, [PAD_IDX, BOS_IDX, EOS_IDX]].sum().item() == 0

    head = CountingModule(d_model=8, vocab_size=vocab_size, dropout=0)
    features = torch.randn(2, 12, 8, requires_grad=True)
    padding_mask = torch.tensor([
        [False] * 8 + [True] * 4,
        [False] * 12,
    ])
    predictions = head(features, padding_mask)
    assert predictions.shape == targets.shape
    assert torch.all(predictions >= 0)

    loss = counting_loss(predictions, tokens)
    loss.backward()
    assert features.grad is not None
    assert all(parameter.grad is not None for parameter in head.parameters())
    print("CAN counting checks passed")
