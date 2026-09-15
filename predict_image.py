"""Recognise handwriting from your own photos: image file -> LaTeX.

    python predict_image.py --checkpoint best_model_can_96px.pt my_note.jpg
    python predict_image.py --checkpoint best_model_can_96px.pt notes/*.jpg --beam 5

Unlike predict_samples.py, which decodes samples out of a processed archive,
this runs the full inference path a real user hits: photo -> perspective
correction and binarisation (inputpreprocessing.py) -> encoder -> decoder.

The two halves of that path have to agree on image height. inputpreprocessing
renders at hmer_model.IMAGE_HEIGHT and the model asserts on anything else, so a
mismatch fails loudly here rather than producing quiet nonsense.

Pass --save-raster to write out what the model actually saw. When a prediction
looks wrong, that image usually explains it -- a bad crop or a blown-out
binarisation is far more often the cause than the decoder.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from hmer_model import HMERModel, IMAGE_HEIGHT
from inputpreprocessing import preprocess_pipeline
from latex_decoder import load_vocab_config


def to_model_tensor(raster):
    """Preprocessor output -> [1, C, IMAGE_HEIGHT, W] float in [0, 1].

    preprocess_pipeline already returns batched NCHW float32 in [0, 1] with a
    white background, which is the same convention dataset.py produces (PNG /
    255.0), so no renormalisation is needed and none is done -- inverting the
    polarity here would hand the model white ink on black.

    Bare [H, W] and [H, W, C] rasters are also accepted so this keeps working
    if the preprocessor's return shape changes.
    """
    arr = np.asarray(raster).astype(np.float32)
    if arr.ndim == 2:                          # [H, W]
        arr = arr[None, None]
    elif arr.ndim == 3:                        # [H, W, C] -> [1, C, H, W]
        arr = arr.transpose(2, 0, 1)[None]
    elif arr.ndim != 4:
        raise SystemExit(f"unexpected preprocessor output shape {arr.shape}")

    if arr.max() > 1.5:                        # 0-255 -> 0-1
        arr = arr / 255.0
    if arr.shape[2] != IMAGE_HEIGHT:
        raise SystemExit(
            f"preprocessing returned height {arr.shape[2]}, but the model needs "
            f"{IMAGE_HEIGHT}. inputpreprocessing.PreprocessConfig.target_height "
            f"and hmer_model.IMAGE_HEIGHT must agree."
        )
    return torch.from_numpy(arr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("images", nargs="+", help="photo files to recognise")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--processed", default="processed-96px-ready",
                   help="only its vocab.json is read, for the token mapping")
    p.add_argument("--beam", type=int, default=5,
                   help="1 = greedy (fast); 5 = what you would report")
    p.add_argument("--max-len", type=int, default=120)
    p.add_argument("--device", default="mps")
    p.add_argument("--save-raster", metavar="DIR",
                   help="write the preprocessed image the model saw into DIR")
    a = p.parse_args()

    cfg = load_vocab_config(f"{a.processed}/vocab.json")
    id_to_tok = {i: t for t, i in cfg.vocab.items()}

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    state = ck.get("model_state", ck)
    # Architecture comes from the weights, so any checkpoint works without
    # the caller remembering which flags produced it.
    use_can = any(k.startswith("counting_module.") for k in state)
    stride = {112: 16, 40: 8}[state["encoder.projection.weight"].shape[1]]
    max_w = state["img_pos_enc.pe"].shape[2]
    recorded = ck.get("image_height")
    if recorded is not None and recorded != IMAGE_HEIGHT:
        raise SystemExit(
            f"{a.checkpoint} was trained on {recorded}px images but "
            f"IMAGE_HEIGHT is {IMAGE_HEIGHT}. Predictions would be meaningless."
        )

    model = HMERModel(cfg.vocab_size, structure_tokens=cfg.structure_tokens,
                      stride=stride, use_can=use_can, max_w=max_w)
    model.load_state_dict(state)
    model.to(a.device).eval()
    print(f"{a.checkpoint}  (epoch {ck.get('epoch', '?')}, "
          f"ExpRate {ck.get('exprate', float('nan')):.4f})  "
          f"{IMAGE_HEIGHT}px  beam={a.beam}\n")

    if a.save_raster:
        Path(a.save_raster).mkdir(parents=True, exist_ok=True)

    failures = 0
    for path in a.images:
        try:
            raster = preprocess_pipeline(path, target_height=IMAGE_HEIGHT)
            images = to_model_tensor(raster).to(a.device)
            widths = torch.tensor([images.shape[3]], device=a.device)

            with torch.no_grad():
                ids = model.predict(images, widths, beam_width=a.beam,
                                    max_len=a.max_len)[0]
            latex = "".join(id_to_tok.get(i, "?") for i in ids)

            print(f"{Path(path).name}")
            print(f"  {images.shape[3]}x{IMAGE_HEIGHT}px  ->  {latex}")

            if a.save_raster:
                from PIL import Image
                out = Path(a.save_raster) / f"{Path(path).stem}_{IMAGE_HEIGHT}px.png"
                Image.fromarray(
                    (images[0, 0].cpu().numpy() * 255).astype(np.uint8)
                ).save(out)
                print(f"  raster -> {out}")
        except Exception as exc:
            # One unreadable photo should not abandon the rest of the batch.
            failures += 1
            print(f"{Path(path).name}\n  FAILED: {type(exc).__name__}: {exc}")
        print()

    if failures:
        print(f"{failures} of {len(a.images)} failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
