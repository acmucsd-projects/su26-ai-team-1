"""Re-render splits CLEAN at TARGET_HEIGHT, with no augmentation.

Usage:
    PYTHONPATH=. python scripts/render_clean_splits.py \
        mathwriting-2024 processed-96px processed-96px-clean valid,test

Needs the raw InkML tree (mathwriting-2024) and pycairo, so that the renderer
matches the archive being corrected:
    brew install cairo pkg-config && pip install "pycairo>=1.25,<2"
Without pycairo mathwriting_pipeline silently falls back to a Pillow renderer,
which would make the clean splits differ from the augmented ones by renderer as
well as by augmentation -- the assert below refuses that.

Verified by re-rendering 200 samples at TARGET_HEIGHT=64 and diffing against
the original clean processed/images/valid PNGs: 200/200 byte-identical.

The processed-96px archive bakes one augmentation draw into every split, so
79% of its valid/test images are rotated/sheared/blurred. Validating on those
is not comparable to a clean-render baseline -- the team already measured that
gap at ~4pp when they hit it at 64px (see scripts/ec2_bootstrap.sh, which
exists purely to pair augmented train with clean valid).

This mirrors exactly the clean path in notebooks/mathwriting_preprocessing.ipynb
(process_split): read -> rescale_to_height -> fit_canvas -> render_ink. It does
NOT call render_with_augmentation. Labels/tokens are copied from the existing
archive unchanged and only width/height are recomputed, because augmentation
changes an image's width and the padding mask is driven by that number.
"""
import json, sys, time
from pathlib import Path

from mathwriting_pipeline import (
    TARGET_HEIGHT, read_inkml_file, rescale_to_height, fit_canvas, render_ink,
    CAIRO_AVAILABLE,
)

RAW = Path(sys.argv[1])          # mathwriting-2024 root
SRC = Path(sys.argv[2])          # processed-96px (for labels/tokens)
OUT = Path(sys.argv[3])          # processed-96px-clean
SPLITS = sys.argv[4].split(",")

assert CAIRO_AVAILABLE, "cairo renderer required to match the archive's renderer"

summary = {}
for split in SPLITS:
    img_dir = OUT / "images" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    (OUT / "labels").mkdir(parents=True, exist_ok=True)

    records = [json.loads(l) for l in open(SRC / "labels" / f"{split}.jsonl")]
    t0, changed, out_records = time.perf_counter(), 0, []
    for i, rec in enumerate(records):
        sid = rec["sample_id"]
        ink = read_inkml_file(RAW / split / f"{sid}.inkml")
        canvas_ink, width, height = fit_canvas(rescale_to_height(ink))
        render_ink(canvas_ink, width, height).save(img_dir / f"{sid}.png")

        if rec["width"] != width:
            changed += 1
        rec = dict(rec, width=width, height=height)
        out_records.append(rec)
        if (i + 1) % 2000 == 0:
            print(f"  {split}: {i+1}/{len(records)}  {time.perf_counter()-t0:.0f}s", flush=True)

    with open(OUT / "labels" / f"{split}.jsonl", "w") as f:
        for rec in out_records:
            f.write(json.dumps(rec) + "\n")

    secs = time.perf_counter() - t0
    summary[split] = {"count": len(out_records), "width_changed_vs_augmented": changed,
                      "seconds": secs}
    print(f"{split}: {len(out_records)} rendered clean in {secs:.0f}s "
          f"({changed} widths differ from the augmented render)", flush=True)

meta = json.loads((SRC / "metadata.json").read_text())
meta["target_height_px"] = TARGET_HEIGHT
meta["augmentation_note"] = (
    "CLEAN renders -- no augmentation. Re-rendered from source InkML for the "
    "splits listed in 'clean_render'; the augmented archive this was derived "
    "from bakes one augmentation draw into every split, which is not usable "
    "for validation you want to compare against a clean baseline."
)
meta.pop("augmentation_config", None)
meta["clean_render"] = summary
(OUT / "metadata.json").write_text(json.dumps(meta, indent=2))
(OUT / "vocab.json").write_text((SRC / "vocab.json").read_text())
print("wrote", OUT / "metadata.json", "and vocab.json")
