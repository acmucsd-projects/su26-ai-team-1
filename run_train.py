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
from torch.utils.data import ConcatDataset, DataLoader, Subset

from dataset import MathWritingDataset, collate_fn, seed_worker
from latex_decoder import load_vocab_config
from hmer_model import HMERModel
from hmer_model import hmer_posformer_train_step, hmer_train_step
from train import fit


def filter_to_model_capacity(ds, name, max_width, max_tokens):
    """Drop samples the model cannot represent, and say how many.

    ImagePositionalEncoding precomputes a [max_h, max_w] table, so an image
    wider than max_w * ENCODER_STRIDE raises rather than being handled. Same
    for token sequences past the decoder's max_len. `train` fits under both by
    luck (max 1003px); `synthetic` does not -- it reaches 2801px.

    Filtering rather than enlarging the tables is deliberate: a single 2801px
    sample pads its entire batch to 2801px, so those few samples would cost
    more compute than the 98% that remain.
    """
    keep = [i for i, r in enumerate(ds.records)
            if r["width"] <= max_width and len(r["token_ids"]) <= max_tokens]
    dropped = len(ds.records) - len(keep)
    if dropped:
        print(f"  split {name:<10}: {len(keep)} "
              f"(dropped {dropped} over capacity: >{max_width}px or >{max_tokens} tokens)")
        return Subset(ds, keep)
    print(f"  split {name:<10}: {len(keep)}")
    return ds


def load_initial_weights(model, checkpoint_path):
    """Warm-start from an existing checkpoint instead of training from scratch.

    Adapted from Yuki's version on `add-can`, minus the CAN-specific tolerance:
    on this branch the architecture is unchanged, so the state dict must match
    EXACTLY. A silently-skipped layer here would look like a bad run rather
    than a loading bug, which is the expensive kind of mistake.
    """
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state", checkpoint.get("state_dict", checkpoint))

    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(
            f"{checkpoint_path} does not match this model. "
            f"missing={incompatible.missing_keys[:5]} "
            f"unexpected={incompatible.unexpected_keys[:5]}. The usual cause is "
            f"a different vocab.json -- the checkpoint and the dataset must "
            f"agree on vocabulary size and token IDs."
        )

    where = [f"epoch {checkpoint[k]}" if k == "epoch" else f"ExpRate {checkpoint[k]:.4f}"
             for k in ("epoch", "exprate") if k in checkpoint]
    print(f"warm start      : {checkpoint_path}"
          f"{' (' + ', '.join(where) + ')' if where else ''}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--processed", default="processed")
    p.add_argument("--raw-dir", default=None,
                   help="root of the source InkML tree; required unless --no-augment")
    p.add_argument("--train-split", default="train",
                   help="comma-separated list; several splits are concatenated "
                        "(e.g. 'train,synthetic' to pretrain on both)")
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
    p.add_argument("--init-checkpoint", default=None,
                   help="warm-start from this checkpoint instead of random init")
    p.add_argument("--encoder-lr", type=float, default=1e-4)
    p.add_argument("--decoder-lr", type=float, default=3e-4,
                   help="lower both (e.g. 3e-5 / 1e-4) when warm-starting")
    p.add_argument("--aux", action="store_true",
                   help="train the PosFormer auxiliary objective as well as CE; "
                        "needs a model built with use_position_forest=True")
    p.add_argument("--val-max-batches", type=int, default=None,
                   help="cap validation batches per epoch (beam search is slow)")
    p.add_argument("--max-width", type=int, default=1024,
                   help="drop wider samples; must be <= max_w * ENCODER_STRIDE")
    p.add_argument("--max-tokens", type=int, default=200,
                   help="drop longer sequences; must be <= the decoder's max_len")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke:
        args.epochs, args.limit_train, args.limit_val = 2, 256, 64
        args.batch_size = min(args.batch_size, 16)

    processed = Path(args.processed)
    cfg = load_vocab_config(processed / "vocab.json")
    print(f"vocab_size      : {cfg.vocab_size}")
    print(f"structure tokens: {cfg.structure_tokens}")

    split_names = [s.strip() for s in args.train_split.split(",") if s.strip()]
    parts = [filter_to_model_capacity(
                 MathWritingDataset(s, processed_dir=processed, raw_dir=args.raw_dir,
                                    augment=bool(args.raw_dir)),
                 s, args.max_width, args.max_tokens)
             for s in split_names]
    train_ds = parts[0] if len(parts) == 1 else ConcatDataset(parts)
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
    if args.init_checkpoint:
        load_initial_weights(model, args.init_checkpoint)
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
                  encoder_lr=args.encoder_lr, decoder_lr=args.decoder_lr,
                  val_beam_width=args.val_beam,
                  val_max_batches=args.val_max_batches,
                  step_fn=hmer_posformer_train_step if args.aux else hmer_train_step,
                  log_every=50)
    print(f"\ntotal wall clock: {(time.perf_counter()-t0)/60:.1f} min")

    Path("history.json").write_text(json.dumps(history, indent=2))
    print("history -> history.json")


if __name__ == "__main__":
    main()
