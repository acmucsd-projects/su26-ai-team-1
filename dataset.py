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
from baseline_decoder import PAD_IDX
from hmer_model import ENCODER_STRIDE  # 16 -- single source of truth, see hmer_model.py


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
        # rec["token_ids"] already carries [BOS, ...ids, EOS] -- the notebook's
        # vocab cell writes them that way, so do NOT re-wrap them here.
        tokens = torch.tensor(rec["token_ids"], dtype=torch.long)

        # Key names are the per-sample half of the contract in hmer_train_step's
        # docstring: "image" / "tokens" / "width".
        return {"image": image_tensor, "tokens": tokens, "width": width}


def seed_worker(worker_id):
    """Pass as `DataLoader(..., worker_init_fn=seed_worker)`. Without this,
    every worker process inherits the Dataset's RNG state as of the fork, so
    they'd draw identical/correlated augmentation sequences instead of
    independent ones."""
    worker_info = torch.utils.data.get_worker_info()
    worker_info.dataset._rng = np.random.default_rng(worker_info.seed % 2**32)


def collate_fn(batch, pad_idx=PAD_IDX, pad_value=1.0):
    """Pads a batch of variable-width images (and variable-length token
    sequences) to that batch's own max, not a global max.

    This is THE collate_fn for the project -- train.py imports it rather than
    defining its own, so the batch contract lives in exactly one place. The
    keys below are what hmer_train_step / validate / model.predict read:

      images:      (B, 1, H, padded_width) float32 in [0, 1]
      tokens:      (B, max_token_len) long, right-padded with `pad_idx`,
                   each row already [BOS, ...ids, EOS]
      true_widths: (B,) long, each sample's real pixel width BEFORE padding.
                   REQUIRED: it is unrecoverable after padding, and the model
                   derives the cross-attention memory mask from it via
                   baseline_decoder.widths_to_memory_padding_mask.

    pad_value=1.0 pads with WHITE, matching the rendered background. Padding
    with black would put a hard fake edge next to the real ink; the decoder
    masks these columns out, but the ENCODER still convolves across them, so
    the edge bleeds into the last few real feature columns.
    """
    batch_max_width = max(item["width"] for item in batch)
    # Round up to a whole number of encoder feature columns so the last column
    # isn't a ragged partial one.
    padded_width = -(-batch_max_width // ENCODER_STRIDE) * ENCODER_STRIDE

    height = batch[0]["image"].shape[1]
    images = torch.full((len(batch), 1, height, padded_width), pad_value, dtype=torch.float32)
    true_widths = torch.empty(len(batch), dtype=torch.long)
    for i, item in enumerate(batch):
        assert item["image"].shape[1] == height, (
            f"expected every image to share a fixed height (mathwriting_pipeline.TARGET_HEIGHT); "
            f"got {item['image'].shape[1]} vs batch height {height} -- only width should vary"
        )
        w = item["image"].shape[2]
        images[i, :, :, :w] = item["image"]
        true_widths[i] = item["width"]

    token_lengths = [item["tokens"].size(0) for item in batch]
    tokens = torch.full((len(batch), max(token_lengths)), pad_idx, dtype=torch.long)
    for i, item in enumerate(batch):
        tokens[i, :token_lengths[i]] = item["tokens"]

    return {"images": images, "tokens": tokens, "true_widths": true_widths}
