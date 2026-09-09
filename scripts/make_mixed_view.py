"""Assemble an augmented-train / clean-valid view out of two archives.

    python scripts/make_mixed_view.py processed-96px processed-96px-clean \
        processed-96px-mixed

Training reads augmented images; validation must read CLEAN ones or the
ExpRate is not comparable to a clean-render baseline (~4pp lower at 64px --
see the note in scripts/ec2_bootstrap.sh, which does this same assembly in
shell for the 64px archives).

This exists as a script rather than as symlinks committed anywhere because
symlinks do not survive a zip/tar round-trip through S3 intact, and a mixed
view that silently resolves to the wrong split is exactly the failure this is
meant to prevent. Splits present in the clean archive win; everything else
comes from the augmented one.
"""
import json, os, sys
from pathlib import Path

AUG, CLEAN, OUT = (Path(p) for p in sys.argv[1:4])

def splits_in(root):
    d = root / "labels"
    return {p.stem for p in d.glob("*.jsonl")} if d.is_dir() else set()

clean_splits = splits_in(CLEAN)
aug_splits = splits_in(AUG) - clean_splits
if not clean_splits:
    raise SystemExit(f"{CLEAN} has no labels/*.jsonl -- nothing clean to use")

(OUT / "images").mkdir(parents=True, exist_ok=True)
(OUT / "labels").mkdir(parents=True, exist_ok=True)

def link(src, dst):
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    os.symlink(os.path.relpath(src.resolve(), dst.parent), dst)

for split, root in [(s, AUG) for s in sorted(aug_splits)] + \
                   [(s, CLEAN) for s in sorted(clean_splits)]:
    link(root / "images" / split, OUT / "images" / split)
    link(root / "labels" / f"{split}.jsonl", OUT / "labels" / f"{split}.jsonl")

link(AUG / "vocab.json", OUT / "vocab.json")

aug_meta = json.loads((AUG / "metadata.json").read_text())
clean_meta = json.loads((CLEAN / "metadata.json").read_text())
if aug_meta["target_height_px"] != clean_meta["target_height_px"]:
    raise SystemExit(
        f"{AUG} is {aug_meta['target_height_px']}px but {CLEAN} is "
        f"{clean_meta['target_height_px']}px -- these cannot be mixed."
    )
(OUT / "metadata.json").write_text(json.dumps({
    "target_height_px": aug_meta["target_height_px"],
    "stroke_width_px": aug_meta["stroke_width_px"],
    "margin_px": aug_meta["margin_px"],
    "renderer": aug_meta["renderer"],
    "view_note": (
        f"Assembled view, not a rendered archive. {sorted(aug_splits)} are "
        f"symlinks into {AUG.name} (augmentation baked in); "
        f"{sorted(clean_splits)} are symlinks into {CLEAN.name} (clean, no "
        f"augmentation). Rebuild with scripts/make_mixed_view.py."
    ),
    "augmented_splits": sorted(aug_splits),
    "clean_splits": sorted(clean_splits),
    "special_tokens": aug_meta["special_tokens"],
    "vocab_size": aug_meta["vocab_size"],
}, indent=2))

for split in sorted(aug_splits | clean_splits):
    n = sum(1 for _ in open(OUT / "labels" / f"{split}.jsonl"))
    imgs = len(list((OUT / "images" / split).glob("*.png")))
    src = (OUT / "images" / split).resolve().parent.parent.name
    if n != imgs:
        raise SystemExit(f"{split}: {n} labels but {imgs} images")
    print(f"  {split:10s} {n:6d} samples  <- {src}")
print(f"{OUT} ready")
