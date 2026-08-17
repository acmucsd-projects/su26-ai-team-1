"""PyTorch Dataset + collate_fn for the processed MathWriting data.

Reads `processed/labels/{split}.jsonl` (written by
`mathwriting_preprocessing.ipynb`). Images are variable-width -- aspect
ratio is preserved, so no two samples are guaranteed the same width -- and
height is fixed at `mathwriting_pipeline.TARGET_HEIGHT`. `collate_fn` pads
each batch to its own max width (not a global max) and rounds up to a
multiple of ENCODER_STRIDE, per processed/metadata.json's
`width_padding_note`.

For `train`, `__getitem__` re-renders the sample from its source InkML with
a fresh call to `render_with_augmentation` every time -- it does NOT read
the clean PNG in `processed/images/train/`, matching this project's
"augmentation is applied online, at training time" design (see the
notebook's Augmentation section). Every other split loads its clean PNG
directly, since there's nothing to distinguish "clean" from "augmented"
outside of train.
"""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from mathwriting_pipeline import read_inkml_file, rescale_to_height, render_with_augmentation

# MobileNetV3's stride; must match processed/metadata.json's width-padding convention.
ENCODER_STRIDE = 32


class MathWritingDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, processed_dir="processed", raw_dir=None, seed=0):
        self.split = split
        self.processed_dir = Path(processed_dir)
        self.raw_dir = Path(raw_dir) if raw_dir is not None else None
        if split == "train" and self.raw_dir is None:
            raise ValueError(
                "raw_dir is required for split='train' (source InkML files are "
                "needed to re-render augmentations each call)"
            )

        label_path = self.processed_dir / "labels" / f"{split}.jsonl"
        with open(label_path, "r", encoding="utf-8") as f:
            self.records = [json.loads(line) for line in f]

        # Per-worker RNG for augmentation draws. Seeded here for the
        # num_workers=0 case; with num_workers>0, pass `seed_worker` as
        # DataLoader's worker_init_fn so each worker process gets an
        # independent stream instead of inheriting this exact state at fork.
        self._rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        sample_id = rec["sample_id"]

        if self.split == "train":
            ink = read_inkml_file(self.raw_dir / "train" / f"{sample_id}.inkml")
            normalized_ink = rescale_to_height(ink)
            image, width, _height, _applied = render_with_augmentation(normalized_ink, self._rng)
        else:
            image = Image.open(self.processed_dir / "images" / self.split / f"{sample_id}.png")
            width = rec["width"]

        image_tensor = torch.from_numpy(np.array(image, dtype=np.float32) / 255.0).unsqueeze(0)  # (1, H, W)
        token_ids = torch.tensor(rec["token_ids"], dtype=torch.long)

        return {"image": image_tensor, "width": width, "token_ids": token_ids}


def seed_worker(worker_id):
    """Pass as `DataLoader(..., worker_init_fn=seed_worker)`. Without this,
    every worker process inherits the Dataset's RNG state as of the fork, so
    they'd draw identical/correlated augmentation sequences instead of
    independent ones."""
    worker_info = torch.utils.data.get_worker_info()
    worker_info.dataset._rng = np.random.default_rng(worker_info.seed % 2**32)


def collate_fn(batch, pad_id=0):
    """Pads a batch of variable-width images (and variable-length token
    sequences) to that batch's own max, not a global max.

    Returns a dict:
      images:       (B, 1, H, padded_width) float32 in [0, 1], zero-padded on the right
      widths:       (B,) long, each sample's true (unpadded) pixel width
      feature_mask: (B, padded_width // ENCODER_STRIDE) bool, True = real content
                    at the encoder's downsampled resolution, for cross-attention
                    to ignore padded columns (per metadata.json's width_padding_note)
      token_ids:    (B, max_token_len) long, padded with `pad_id`
      token_lengths:(B,) long, true (unpadded) token sequence lengths
    """
    batch_max_width = max(item["width"] for item in batch)
    padded_width = -(-batch_max_width // ENCODER_STRIDE) * ENCODER_STRIDE  # round up to a multiple of ENCODER_STRIDE

    height = batch[0]["image"].shape[1]
    images = torch.zeros(len(batch), 1, height, padded_width, dtype=torch.float32)
    widths = torch.empty(len(batch), dtype=torch.long)
    for i, item in enumerate(batch):
        assert item["image"].shape[1] == height, (
            f"expected every image to share a fixed height (mathwriting_pipeline.TARGET_HEIGHT); "
            f"got {item['image'].shape[1]} vs batch height {height} -- only width should vary"
        )
        w = item["image"].shape[2]
        images[i, :, :, :w] = item["image"]
        widths[i] = item["width"]

    feature_len = padded_width // ENCODER_STRIDE
    feature_cols = torch.arange(feature_len).unsqueeze(0)                                # (1, feature_len)
    valid_feature_cols = torch.ceil(widths.float() / ENCODER_STRIDE).long().unsqueeze(1)  # (B, 1)
    feature_mask = feature_cols < valid_feature_cols                                      # (B, feature_len) bool

    token_lengths = torch.tensor([len(item["token_ids"]) for item in batch], dtype=torch.long)
    max_tokens = token_lengths.max().item()
    token_ids = torch.full((len(batch), max_tokens), pad_id, dtype=torch.long)
    for i, item in enumerate(batch):
        token_ids[i, :len(item["token_ids"])] = item["token_ids"]

    return {
        "images": images,
        "widths": widths,
        "feature_mask": feature_mask,
        "token_ids": token_ids,
        "token_lengths": token_lengths,
    }
