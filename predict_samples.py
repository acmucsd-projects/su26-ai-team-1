"""Decode validation samples with a trained checkpoint and print predicted vs
ground-truth LaTeX.

    python predict_samples.py --checkpoint best_model.pt --n 15 --beam 5
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from baseline_decoder import BOS_IDX, EOS_IDX, PAD_IDX
from dataset import MathWritingDataset, collate_fn
from hmer_model import HMERModel
from latex_decoder import load_vocab_config
from train import strip_special, expression_rate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--processed", default="processed")
    p.add_argument("--split", default="valid")
    p.add_argument("--checkpoint", default="best_model.pt")
    p.add_argument("--n", type=int, default=15)
    p.add_argument("--beam", type=int, default=5)
    p.add_argument("--device", default="mps")
    p.add_argument("--max-len", type=int, default=120)
    args = p.parse_args()

    processed = Path(args.processed)
    cfg = load_vocab_config(processed / "vocab.json")
    id_to_tok = {i: t for t, i in cfg.vocab.items()}

    ds = MathWritingDataset(args.split, processed_dir=processed)
    ds = Subset(ds, range(min(args.n, len(ds))))
    loader = DataLoader(ds, batch_size=8, collate_fn=collate_fn)

    model = HMERModel(cfg.vocab_size, structure_tokens=cfg.structure_tokens)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state.get("model_state", state))
    model.to(args.device).eval()
    print(f"loaded {args.checkpoint} "
          f"(epoch {state.get('epoch', '?')}, ExpRate {state.get('exprate', float('nan')):.3f})\n")

    def detok(ids):
        return "".join(id_to_tok.get(i, "?") for i in ids)

    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            out = model.predict(batch["images"], batch["true_widths"],
                                beam_width=args.beam, max_len=args.max_len)
            preds.extend(out)
            targets.extend(strip_special(r) for r in batch["tokens"].tolist())

    hits = 0
    for i, (pr, tg) in enumerate(zip(preds, targets), 1):
        ok = pr == tg
        hits += ok
        print(f"[{i:>3}] {'MATCH' if ok else '  -  '}")
        print(f"      truth: {detok(tg)}")
        print(f"      pred : {detok(pr)}")

    print(f"\nexact matches: {hits}/{len(preds)}")
    print(expression_rate(preds, targets))


if __name__ == "__main__":
    main()
