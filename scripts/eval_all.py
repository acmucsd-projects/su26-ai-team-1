"""Evaluate many checkpoints on one split and emit a comparable results table.

    PYTHONPATH=. python scripts/eval_all.py \
        --processed processed-96px-ready --split valid --height 96 --beam 1 \
        best_model_96px.pt best_model_can_96px.pt

Every checkpoint is scored on the SAME split with the SAME decoding, which is
the only way the numbers can be put in one table. Training-time ExpRate is not
comparable to this: it is usually capped at 1024 samples (~3% sampling error)
on a fixed non-random subset, whereas this runs the full split (~0.75%).

Architecture is read from the weights, not from flags, so a checkpoint cannot
be silently evaluated as something it is not:
  - use_can   <- presence of counting_module.* keys
  - stride    <- encoder.projection input channels (112 -> 16, 40 -> 8)
  - max_w     <- the stored positional-encoding table's width

`--height` is the one thing NOT recoverable from the weights: at stride 16 a
64px and a 96px model have identical parameter shapes. It must match the
archive being evaluated against, so 64px and 96px models need separate runs.
Checkpoints that record image_height are cross-checked against it and refuse
to run on a mismatch.
"""
import argparse, json, time
from pathlib import Path

import torch
import hmer_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--processed", required=True)
    p.add_argument("--split", default="valid")
    p.add_argument("--height", type=int, required=True,
                   help="input height these checkpoints were trained at; must "
                        "match --processed's target_height_px")
    p.add_argument("--beam", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="mps")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--markdown", help="write the results table here")
    a = p.parse_args()

    meta_path = Path(a.processed) / "metadata.json"
    if meta_path.exists():
        rendered = json.loads(meta_path.read_text()).get("target_height_px")
        if rendered is not None and rendered != a.height:
            raise SystemExit(
                f"--height {a.height} but {meta_path} was rendered at "
                f"{rendered}px. These must agree."
            )
        clean = json.loads(meta_path.read_text()).get("clean_splits")
        if clean is not None and a.split not in clean:
            print(f"WARNING: '{a.split}' is not in this archive's clean_splits "
                  f"{clean}. Augmented validation scores lower and is not "
                  f"comparable to a clean-render baseline.")

    # Must be rebound before HMERModel is imported/constructed: feat_h derives
    # from it, and the whole point of this script is evaluating models trained
    # at a height other than the repo's current default.
    hmer_model.IMAGE_HEIGHT = a.height

    from torch.utils.data import DataLoader
    from dataset import MathWritingDataset, collate_fn
    from latex_decoder import load_vocab_config
    from train import validate
    from hmer_model import HMERModel

    cfg = load_vocab_config(f"{a.processed}/vocab.json")
    ds = MathWritingDataset(a.split, processed_dir=a.processed)
    loader = DataLoader(ds, batch_size=a.batch_size, collate_fn=collate_fn,
                        num_workers=a.workers)
    n = len(ds)
    print(f"data: {a.processed}/{a.split}  n={n}  height={a.height}  beam={a.beam}\n")

    rows = []
    for ck_path in a.checkpoints:
        ck = torch.load(ck_path, map_location="cpu", weights_only=True)
        state = ck.get("model_state", ck)

        recorded = ck.get("image_height")
        if recorded is not None and recorded != a.height:
            print(f"SKIP {ck_path}: records image_height={recorded}, not {a.height}")
            continue

        use_can = any(k.startswith("counting_module.") for k in state)
        stride = {112: 16, 40: 8}.get(state["encoder.projection.weight"].shape[1])
        if stride is None:
            print(f"SKIP {ck_path}: unrecognised encoder width")
            continue
        max_w = state["img_pos_enc.pe"].shape[2]

        model = HMERModel(cfg.vocab_size, structure_tokens=cfg.structure_tokens,
                          stride=stride, use_can=use_can, max_w=max_w)
        model.load_state_dict(state)     # strict: a mismatch is a real problem
        model.to(a.device)

        t0 = time.perf_counter()
        r = validate(model, loader, device=a.device, beam_width=a.beam)
        mins = (time.perf_counter() - t0) / 60
        se = (r["exprate"] * (1 - r["exprate"]) / n) ** 0.5
        rows.append({
            "checkpoint": Path(ck_path).name, "exprate": r["exprate"],
            "ci": 1.96 * se, "leq1": r["exprate_leq1"], "leq2": r["exprate_leq2"],
            "val_loss": r["val_loss"], "can": use_can, "stride": stride,
            "epoch": ck.get("epoch"), "minutes": mins,
        })
        print(f"{Path(ck_path).name:38s} ExpRate {r['exprate']:.4f} "
              f"+/-{1.96*se:.4f}  ({round(r['exprate']*n)}/{n})  "
              f"can={use_can} stride={stride}  {mins:.1f}m", flush=True)

    rows.sort(key=lambda x: -x["exprate"])
    head = (f"\n| Checkpoint | ExpRate | <=1 | <=2 | val_loss | CAN | stride | epoch |\n"
            f"|---|---|---|---|---|---|---|---|\n")
    body = "".join(
        f"| `{r['checkpoint']}` | {r['exprate']:.4f} ±{r['ci']:.4f} | {r['leq1']:.4f} "
        f"| {r['leq2']:.4f} | {r['val_loss']:.4f} | {'yes' if r['can'] else 'no'} "
        f"| {r['stride']} | {r['epoch']} |\n" for r in rows)
    caption = (f"\nAll rows: {a.processed}/{a.split}, n={n}, "
               f"{'greedy' if a.beam == 1 else f'beam-{a.beam}'}, height {a.height}px.\n")
    print(head + body + caption)
    if a.markdown:
        Path(a.markdown).write_text(head + body + caption)
        print(f"table -> {a.markdown}")


if __name__ == "__main__":
    main()
