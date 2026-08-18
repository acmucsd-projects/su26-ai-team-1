"""Training driver for the merged HMER pipeline.

Usage:
    python run_train.py --epochs 30 --batch-size 32 --device mps
    python run_train.py --smoke        # tiny subset, 2 epochs, sanity only
"""
import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from dataset import MathWritingDataset, collate_fn, seed_worker
from latex_decoder import load_vocab_config
from hmer_model import HMERModel
from train import fit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--processed", default="processed")
    p.add_argument("--raw-dir", default=None,
                   help="root of the source InkML tree; required unless --no-augment")
    p.add_argument("--train-split", default="train")
    p.add_argument("--val-split", default="valid")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="mps")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--freeze-encoder", action="store_true",
                   help="isolate the decoder for a first run")
    p.add_argument("--limit-train", type=int, default=None)
    p.add_argument("--limit-val", type=int, default=None)
    p.add_argument("--val-beam", type=int, default=1)
    p.add_argument("--checkpoint", default="best_model.pt")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke:
        args.epochs, args.limit_train, args.limit_val = 2, 256, 64
        args.batch_size = min(args.batch_size, 16)

    processed = Path(args.processed)
    cfg = load_vocab_config(processed / "vocab.json")
    print(f"vocab_size      : {cfg.vocab_size}")
    print(f"structure tokens: {cfg.structure_tokens}")

    train_ds = MathWritingDataset(args.train_split, processed_dir=processed,
                                  raw_dir=args.raw_dir,
                                  augment=bool(args.raw_dir))
    val_ds = MathWritingDataset(args.val_split, processed_dir=processed)
    print(f"train / val     : {len(train_ds)} / {len(val_ds)}")

    if args.limit_train:
        train_ds = Subset(train_ds, range(min(args.limit_train, len(train_ds))))
    if args.limit_val:
        val_ds = Subset(val_ds, range(min(args.limit_val, len(val_ds))))
        print(f"limited to      : {len(train_ds)} / {len(val_ds)}")

    common = dict(collate_fn=collate_fn, num_workers=args.workers)
    if args.workers > 0:
        common["worker_init_fn"] = seed_worker
        common["persistent_workers"] = True
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **common)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, **common)

    model = HMERModel(cfg.vocab_size, structure_tokens=cfg.structure_tokens)
    if args.freeze_encoder:
        for prm in model.encoder.parameters():
            prm.requires_grad = False
        print("encoder         : FROZEN")

    n = sum(p_.numel() for p_ in model.parameters())
    print(f"parameters      : {n/1e6:.2f}M")
    print(f"device          : {args.device}\n")

    t0 = time.perf_counter()
    history = fit(model, train_loader, val_loader,
                  epochs=args.epochs, device=args.device,
                  checkpoint_path=args.checkpoint,
                  val_beam_width=args.val_beam, log_every=50)
    print(f"\ntotal wall clock: {(time.perf_counter()-t0)/60:.1f} min")

    Path("history.json").write_text(json.dumps(history, indent=2))
    print("history -> history.json")


if __name__ == "__main__":
    main()
