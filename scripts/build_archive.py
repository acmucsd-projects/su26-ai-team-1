"""Build a processed archive from raw InkML, on whatever box has the data.

    PYTHONPATH=. python scripts/build_archive.py \
        --raw mathwriting-2024 --out processed-96px-ready \
        --vocab processed/vocab.json \
        --augment train,synthetic,symbols --clean valid,test

Renders at mathwriting_pipeline.TARGET_HEIGHT and writes images + labels +
metadata.json, i.e. everything run_train.py needs. Splits listed in --augment
get one augmentation draw baked in; splits in --clean are rendered through the
clean path only. Validation belongs in --clean: augmented validation cost ~4pp
ExpRate at 64px and is not comparable to a clean-render baseline.

Why the vocab is an input and never rebuilt: token ids are positional in the
vocab file, so rebuilding it from a different split mix silently renumbers
every id and quietly invalidates every existing checkpoint. This reuses the
frozen vocab and reports the OOV rate instead, which is the number that
actually tells you whether the vocab still fits the data.

Re-running skips images that already exist, so an interrupted run resumes
rather than starting over.
"""
import argparse, json, time
from pathlib import Path

from mathwriting_pipeline import (
    TARGET_HEIGHT, STROKE_WIDTH_PX, MARGIN_PX, CAIRO_AVAILABLE,
    read_inkml_file, rescale_to_height, fit_canvas, render_ink,
    render_with_augmentation, tokenize_expression, get_label_text,
    AUGMENTATION_CONFIG, RNG_SEED,
)
import numpy as np

BOS, EOS, UNK = "<BOS>", "<EOS>", "<UNK>"


def find_split_dir(raw: Path, split: str) -> Path:
    """Locate <split>/ under raw, tolerating an extra nesting level.

    A .tgz extracted into a directory of the same name gives
    mathwriting-2024/mathwriting-2024/valid, which is easy to end up with and
    produces a confusing 'no such file' much later if not resolved here.
    """
    for cand in (raw / split, raw / raw.name / split):
        if cand.is_dir():
            return cand
    matches = [p for p in raw.rglob(split) if p.is_dir()
               and any(p.glob("*.inkml"))]
    if len(matches) == 1:
        return matches[0]
    raise SystemExit(
        f"could not find a '{split}' directory with .inkml files under {raw}. "
        f"Looked at {raw/split}, {raw/raw.name/split}, and searched recursively "
        f"(found {len(matches)} candidates). Pass --raw pointing at the "
        f"directory that directly contains the split folders."
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True, help="root of the InkML tree")
    p.add_argument("--out", required=True)
    p.add_argument("--vocab", required=True,
                   help="existing vocab.json -- reused, never rebuilt")
    p.add_argument("--augment", default="", help="comma-separated splits")
    p.add_argument("--clean", default="", help="comma-separated splits")
    p.add_argument("--progress-every", type=int, default=5000)
    a = p.parse_args()

    if not CAIRO_AVAILABLE:
        raise SystemExit(
            "pycairo not importable, so rendering would silently fall back to "
            "the Pillow path and produce images that differ from any "
            "cairo-rendered archive. Install it:\n"
            "  pip install pycairo   (needs system cairo + pkg-config)"
        )

    raw, out = Path(a.raw), Path(a.out)
    aug_splits = [s for s in a.augment.split(",") if s]
    clean_splits = [s for s in a.clean.split(",") if s]
    overlap = set(aug_splits) & set(clean_splits)
    if overlap:
        raise SystemExit(f"{sorted(overlap)} listed as both --augment and --clean")
    if not aug_splits and not clean_splits:
        raise SystemExit("nothing to do: pass --augment and/or --clean")

    vocab = json.loads(Path(a.vocab).read_text())
    print(f"vocab           : {a.vocab} ({len(vocab)} tokens, reused not rebuilt)")
    print(f"target height   : {TARGET_HEIGHT}px   renderer: cairo")

    (out / "labels").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)
    summary = {}

    for split in aug_splits + clean_splits:
        augment = split in aug_splits
        split_dir = find_split_dir(raw, split)
        img_dir = out / "images" / split
        img_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(p for p in split_dir.iterdir() if p.suffix == ".inkml")
        print(f"\n{split}: {len(files)} inks from {split_dir} "
              f"({'augmented' if augment else 'CLEAN'})")

        records, t0, skipped, n_unk, n_tok = [], time.perf_counter(), 0, 0, 0
        for i, path in enumerate(files):
            sid = path.stem
            png = img_dir / f"{sid}.png"
            ink = read_inkml_file(path)
            normalized = rescale_to_height(ink)

            if png.exists():
                # resume: trust the pixels, but width must come from the file
                # itself -- it is what drives the cross-attention padding mask
                from PIL import Image
                with Image.open(png) as im:
                    width, height = im.size
                skipped += 1
            elif augment:
                image, width, height, _ = render_with_augmentation(normalized, rng)
                image.save(png)
            else:
                canvas_ink, width, height = fit_canvas(normalized)
                render_ink(canvas_ink, width, height).save(png)

            toks = tokenize_expression(get_label_text(ink.annotations))
            ids = [vocab.get(t, vocab[UNK]) for t in toks]
            n_unk += sum(1 for x in ids if x == vocab[UNK])
            n_tok += len(ids)

            records.append({
                "sample_id": sid, "split": split,
                "label": ink.annotations.get("label"),
                "normalized_label": ink.annotations.get("normalizedLabel"),
                "tokens": toks,
                "token_ids": [vocab[BOS]] + ids + [vocab[EOS]],
                "width": width, "height": height,
                "num_strokes": len(normalized.strokes),
            })

            if a.progress_every and (i + 1) % a.progress_every == 0:
                el = time.perf_counter() - t0
                rate = (i + 1) / el
                print(f"    {i+1}/{len(files)}  {el:.0f}s  {rate:.0f} ink/s  "
                      f"~{(len(files)-i-1)/rate:.0f}s left", flush=True)

        with open(out / "labels" / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        secs = time.perf_counter() - t0
        oov = n_unk / n_tok if n_tok else 0.0
        summary[split] = {"count": len(records), "augmented": augment,
                          "seconds": secs, "oov_rate": oov,
                          "reused_existing_png": skipped}
        print(f"  {len(records)} records, OOV {oov:.2%}, {secs:.0f}s"
              + (f", reused {skipped} existing PNGs" if skipped else ""))

    (out / "vocab.json").write_text(Path(a.vocab).read_text())
    (out / "metadata.json").write_text(json.dumps({
        "target_height_px": TARGET_HEIGHT,
        "stroke_width_px": STROKE_WIDTH_PX,
        "margin_px": MARGIN_PX,
        "renderer": "cairo",
        "augmented_splits": aug_splits,
        "clean_splits": clean_splits,
        "augmentation_config": {k: {kk: list(vv) if isinstance(vv, tuple) else vv
                                    for kk, vv in v.items() if kk != "dtype"}
                                for k, v in AUGMENTATION_CONFIG.items()},
        "augmentation_note": (
            "Built by scripts/build_archive.py. Splits in 'clean_splits' are "
            "rendered with no augmentation so ExpRate is comparable to a "
            "clean-render baseline."
        ),
        "vocab_size": len(vocab),
        "build": summary,
    }, indent=2))
    print(f"\n{out} ready -- point --processed at it")


if __name__ == "__main__":
    main()
