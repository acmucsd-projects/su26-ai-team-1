"""Score a checkpoint on a validation split.

Per-epoch ExpRate during training is capped (--limit-val), where the standard
error near 0.62 is ~1.5pp -- too wide to separate two models a point apart.
Running the whole 15,674-sample split tightens that to ~0.4pp, which is the
difference between "we improved" and "we cannot tell".

Also the place to spend beam search: it is too slow for per-epoch validation
but it is what you want for a number you would actually report.

    python evaluate.py --checkpoint best_model_aws.pt --device cuda --beam 5

The __main__ guard is required -- macOS spawns dataloader workers rather than
forking, so module-level code would re-execute in every worker.
"""
import argparse
import time

import torch
from torch.utils.data import DataLoader, Subset

from dataset import MathWritingDataset, collate_fn
from hmer_model import HMERModel
from latex_decoder import load_vocab_config
from train import validate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--processed", default="processed-mixed")
    p.add_argument("--split", default="valid")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--beam", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="mps")
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()

    cfg = load_vocab_config(f"{a.processed}/vocab.json")
    ds = MathWritingDataset(a.split, processed_dir=a.processed)
    if a.limit:
        ds = Subset(ds, range(min(a.limit, len(ds))))
    loader = DataLoader(ds, batch_size=a.batch_size, collate_fn=collate_fn,
                        num_workers=a.workers)

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    state = ck.get("model_state", ck)
    # Detected from the checkpoint itself rather than a CLI flag, so this
    # keeps working for any future architecture variant without the caller
    # needing to remember which flags produced which checkpoint.
    use_can = any(k.startswith("counting_module.") for k in state)
    # encoder.projection's input-channel count is stride-specific (112 for
    # stride-16, 40 for stride-8 -- see _STRIDE_TO_ENCODER_CUTOFF in
    # hmer_model.py), so it tells us which stride this checkpoint was built
    # with. Same bug class as use_can above: this crashed on the first
    # stride-8 checkpoint because the default (16) was assumed silently.
    channels_to_stride = {112: 16, 40: 8}
    proj_in_channels = state["encoder.projection.weight"].shape[1]
    stride = channels_to_stride.get(proj_in_channels)
    if stride is None:
        raise ValueError(
            f"encoder.projection has {proj_in_channels} input channels, which "
            f"doesn't match any known stride ({channels_to_stride}). Was this "
            f"checkpoint built with a cutoff not yet in _STRIDE_TO_ENCODER_CUTOFF?"
        )
    model = HMERModel(cfg.vocab_size, structure_tokens=cfg.structure_tokens,
                      stride=stride,
                      use_can=use_can)
    model.load_state_dict(state)
    model.to(a.device)

    t0 = time.perf_counter()
    r = validate(model, loader, device=a.device, beam_width=a.beam)
    n = len(ds)
    se = (r["exprate"] * (1 - r["exprate"]) / n) ** 0.5

    print(f"\ncheckpoint : {a.checkpoint}")
    print(f"data       : {a.processed}/{a.split}   n={n}  beam={a.beam}")
    print(f"val_loss   : {r['val_loss']:.4f}")
    print(f"ExpRate    : {r['exprate']:.4f}  +/- {1.96 * se:.4f}  (95% CI)")
    print(f"ExpRate<=1 : {r['exprate_leq1']:.4f}")
    print(f"ExpRate<=2 : {r['exprate_leq2']:.4f}")
    print(f"took       : {(time.perf_counter() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
