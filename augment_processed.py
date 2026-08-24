"""Bake deterministic augmentations into a separate processed dataset.

InkML-backed splits are re-rendered from strokes. If a processed split has no
matching raw directory (currently ``symbols``), its PNG is augmented in image
space. A complete split is staged before it is committed to the output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from mathwriting_pipeline import (AUGMENTATION_CONFIG, MARGIN_PX, RNG_SEED,
    TARGET_HEIGHT, _roll, read_inkml_file, render_with_augmentation,
    rescale_to_height)

ALL_SPLITS = ("train", "valid", "test", "synthetic", "symbols")


def sample_seed(seed: int, split: str, sample_id: str) -> int:
    payload = f"{seed}:{split}:{sample_id}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def _augment_raster(source: Path, rng: np.random.Generator):
    """Image-space fallback for a split whose source InkML is unavailable."""
    image = Image.open(source).convert("L")
    applied = {}
    angle = _roll(AUGMENTATION_CONFIG["rotation_deg"], rng)
    shear = _roll(AUGMENTATION_CONFIG["shear_deg"], rng)
    if angle is not None:
        image = image.rotate(angle, Image.Resampling.BICUBIC, expand=True, fillcolor=255)
        applied["rotation_deg"] = angle
    if shear is not None:
        factor = np.tan(np.deg2rad(shear))
        extra = int(np.ceil(abs(factor) * image.height))
        image = image.transform((image.width + extra, image.height), Image.Transform.AFFINE,
            (1, factor, 0 if factor >= 0 else extra, 0, 1, 0),
            resample=Image.Resampling.BICUBIC, fillcolor=255)
        applied["shear_deg"] = shear
    bbox = Image.eval(image, lambda p: 255 - p).getbbox()
    if bbox:
        image = image.crop(bbox)
    ink_height = TARGET_HEIGHT - 2 * MARGIN_PX
    width = max(1, round(image.width * ink_height / max(image.height, 1)))
    image = image.resize((width, ink_height), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (width + 2 * MARGIN_PX, TARGET_HEIGHT), 255)
    canvas.paste(image, (MARGIN_PX, MARGIN_PX))
    blur = _roll(AUGMENTATION_CONFIG["blur_radius"], rng)
    if blur is not None:
        canvas = canvas.filter(ImageFilter.GaussianBlur(blur))
        applied["blur_radius"] = blur
    return canvas, canvas.width, TARGET_HEIGHT, applied


def _process_one(args):
    split, sample_id, raw_root, source_image, output, seed = args
    rng = np.random.default_rng(sample_seed(seed, split, sample_id))
    raw_path = Path(raw_root) / split / f"{sample_id}.inkml" if raw_root else None
    if raw_path and raw_path.is_file():
        ink = read_inkml_file(raw_path)
        image, width, height, applied = render_with_augmentation(rescale_to_height(ink), rng)
        mode = "inkml"
    else:
        image, width, height, applied = _augment_raster(Path(source_image), rng)
        mode = "raster-fallback"
    image.save(output, format="PNG")
    return sample_id, width, height, applied, mode


def process_split(split, raw_root, source, staging, seed, workers, limit=0):
    label_path = source / "labels" / f"{split}.jsonl"
    with label_path.open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    if limit:
        records = records[:limit]
    output_dir = staging / split
    output_dir.mkdir(parents=True, exist_ok=True)
    has_raw = (raw_root / split).is_dir()
    tasks = [(split, rec["sample_id"], str(raw_root) if has_raw else None,
        str(source / "images" / split / f'{rec["sample_id"]}.png'),
        str(output_dir / f'{rec["sample_id"]}.png'), seed) for rec in records]
    started = time.perf_counter()
    dimensions, counts, modes = {}, Counter(), Counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (sample_id, width, height, applied, mode) in enumerate(pool.map(_process_one, tasks, chunksize=32), 1):
            dimensions[sample_id] = (width, height)
            counts.update(applied.keys() if applied else ("none",))
            modes[mode] += 1
            if i % 5000 == 0 or i == len(tasks):
                elapsed = time.perf_counter() - started
                eta = elapsed / i * (len(tasks) - i)
                print(f"{split}: {i}/{len(tasks)} ({i/elapsed:.1f}/s, ETA {eta/60:.1f} min)", flush=True)
    return records, dimensions, counts, dict(modes), time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("mathwriting-2024"))
    parser.add_argument("--processed-dir", type=Path, default=Path("processed"),
                        help="Read-only baseline dataset")
    parser.add_argument("--output-dir", type=Path, default=Path("processed-augmented"),
                        help="Destination for the complete augmented dataset")
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--splits", nargs="+", choices=ALL_SPLITS, default=list(ALL_SPLITS))
    parser.add_argument("--limit", type=int, default=0, help="QA only: stage N images per split")
    parser.add_argument("--qa-dir", type=Path, default=Path("augmentation_qa"))
    args = parser.parse_args()
    staging = args.qa_dir if args.limit else args.output_dir / ".staging"
    if staging.exists():
        shutil.rmtree(staging)
    if not args.limit:
        (args.output_dir / "images").mkdir(parents=True, exist_ok=True)
        (args.output_dir / "labels").mkdir(parents=True, exist_ok=True)
    summary = {}
    for split in args.splits:
        target_images = args.output_dir / "images" / split
        target_label = args.output_dir / "labels" / f"{split}.jsonl"
        if not args.limit and target_images.is_dir() and target_label.is_file():
            with target_label.open(encoding="utf-8") as f:
                count = sum(1 for _ in f)
            summary[split] = {"count": count, "status": "already-completed"}
            print(f"{split}: already completed ({count} images); skipping", flush=True)
            continue
        records, dimensions, counts, mode, elapsed = process_split(
            split, args.raw_dir, args.processed_dir, staging, args.seed, args.workers, args.limit)
        summary[split] = {"count": len(records), "mode": mode, "seconds": elapsed, "transforms": dict(counts)}
        if args.limit:
            continue
        source_label = args.processed_dir / "labels" / f"{split}.jsonl"
        temp_label = target_label.with_suffix(".jsonl.tmp")
        with source_label.open(encoding="utf-8") as src, temp_label.open("w", encoding="utf-8") as dst:
            for line in src:
                rec = json.loads(line)
                rec["width"], rec["height"] = dimensions[rec["sample_id"]]
                dst.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if target_images.exists():
            shutil.rmtree(target_images)
        shutil.move(str(staging / split), str(target_images))
        shutil.move(str(temp_label), str(target_label))
    if not args.limit:
        source_metadata = args.processed_dir / "metadata.json"
        metadata_path = args.output_dir / "metadata.json"
        metadata = json.loads(source_metadata.read_text(encoding="utf-8"))
        metadata["augmentation_note"] = "Augmentations are baked once into every processed PNG with stable per-sample seeds; training reads these PNGs directly."
        metadata["augmentation_config"] = {
            name: {**cfg, "dtype": cfg["dtype"].__name__}
            for name, cfg in AUGMENTATION_CONFIG.items()
        }
        metadata["augmentation_seed"] = args.seed
        metadata["augmentation_run"] = summary
        temp = metadata_path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temp.replace(metadata_path)
        shutil.copy2(args.processed_dir / "vocab.json", args.output_dir / "vocab.json")
        if staging.exists():
            staging.rmdir()
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
