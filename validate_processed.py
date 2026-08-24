"""Validate PNG readability and fixed height for a processed dataset."""
import argparse
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image


def inspect_png(path):
    try:
        with Image.open(path) as image:
            width, height = image.size
            image.verify()
        return str(path), width, height, None
    except Exception as exc:
        return str(path), None, None, repr(exc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args()
    paths = sorted((args.dataset / "images").glob("*/*.png"))
    failures, heights = [], Counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, (path, width, height, error) in enumerate(pool.map(inspect_png, paths, chunksize=128), 1):
            if error or height != args.height or not width:
                failures.append((path, width, height, error))
            heights[height] += 1
            if i % 25000 == 0 or i == len(paths):
                print(f"validated {i}/{len(paths)} PNGs", flush=True)
    print(f"height counts: {dict(heights)}")
    if failures:
        print(f"failures ({len(failures)}): {failures[:20]}")
        raise SystemExit(1)
    print(f"PASS: all {len(paths)} PNGs are readable and exactly {args.height}px high")


if __name__ == "__main__":
    main()
