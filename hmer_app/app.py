"""Tiny local web app: drop in an image, get LaTeX back.

    python app.py                      # then open http://localhost:8000
    python app.py --checkpoint ../best_model_96px.pt --port 8080

Standard library only -- no Flask, no Gradio, nothing to install. The model is
loaded once at startup, so the first request is as fast as the rest.

Three preprocessing modes, because the right one depends on where the image
came from and picking wrong changes the answer completely:

  inkml  (default) re-render via centreline thinning so the input matches the
         2.5px uniform strokes the model trained on. Best for drawing-app
         exports, screenshots, and anything with thick or uneven strokes.
  photo  perspective correction + page detection + binarisation. For actual
         photographs of paper. On a clean digital image its region detector has
         nothing to lock onto and will crop to a fragment.
  direct feed the image in as-is, only resized to the model's height. For
         images already rendered in the training convention.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Three layouts, checked in order, so the same file runs everywhere without
# editing paths:
#   HERE                          a flat deploy bundle: modules beside app.py
#   HERE.parent                   inside the repo:      <repo>/hmer_app/
#   HERE.parent/su26-ai-team-1    beside the repo:      ACM_AI/hmer_app/
for _candidate in (HERE, HERE.parent, HERE.parent / "su26-ai-team-1"):
    if (_candidate / "hmer_model.py").exists():
        REPO = _candidate
        break
else:
    raise SystemExit(
        f"cannot find hmer_model.py; looked in {HERE}, {HERE.parent} and "
        f"{HERE.parent / 'su26-ai-team-1'}"
    )
sys.path.insert(0, str(REPO))

import numpy as np                                            # noqa: E402
import torch                                                  # noqa: E402
from PIL import Image                                         # noqa: E402

from hmer_model import HMERModel, IMAGE_HEIGHT                # noqa: E402
from latex_decoder import load_vocab_config                   # noqa: E402
from inkml_normalize import to_inkml_render                   # noqa: E402

STATE: dict = {}


def load_model(checkpoint: str, processed: str, device: str):
    cfg = load_vocab_config(f"{processed}/vocab.json")
    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = ck.get("model_state", ck)

    # Architecture is read from the weights so any checkpoint works without the
    # caller recalling which flags produced it.
    use_can = any(k.startswith("counting_module.") for k in state)
    stride = {112: 16, 40: 8}[state["encoder.projection.weight"].shape[1]]
    max_w = state["img_pos_enc.pe"].shape[2]
    recorded = ck.get("image_height")
    if recorded is not None and recorded != IMAGE_HEIGHT:
        raise SystemExit(
            f"{checkpoint} was trained on {recorded}px images but IMAGE_HEIGHT "
            f"is {IMAGE_HEIGHT}; predictions would be meaningless."
        )

    model = HMERModel(cfg.vocab_size, structure_tokens=cfg.structure_tokens,
                      stride=stride, use_can=use_can, max_w=max_w)
    model.load_state_dict(state)
    model.to(device).eval()

    STATE.update(model=model, device=device,
                 id_to_tok={i: t for t, i in cfg.vocab.items()},
                 name=Path(checkpoint).name,
                 exprate=ck.get("exprate"), epoch=ck.get("epoch"))
    print(f"loaded {Path(checkpoint).name} "
          f"(epoch {ck.get('epoch','?')}, ExpRate {ck.get('exprate', float('nan')):.4f}) "
          f"on {device}, {IMAGE_HEIGHT}px")


def flatten_to_white(im: Image.Image) -> Image.Image:
    """RGBA -> RGB on white. A transparent background would otherwise become
    black, handing the model white ink on black -- the inverse of its training
    data, which produces confident nonsense rather than an obvious error."""
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[3])
        return bg
    return im.convert("RGB")


def rasterise(im: Image.Image, mode: str) -> Image.Image:
    if mode == "inkml":
        return to_inkml_render(im)
    if mode == "photo":
        import tempfile
        from inputpreprocessing import preprocess_pipeline
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fh:
            im.save(fh.name)
            arr = np.asarray(preprocess_pipeline(fh.name, target_height=IMAGE_HEIGHT))
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim == 3:
            arr = arr[0] if arr.shape[0] in (1, 3) else arr.mean(axis=2)
        if arr.max() <= 1.5:
            arr = arr * 255.0
        return Image.fromarray(arr.astype(np.uint8), mode="L")
    # direct
    grey = im.convert("L")
    w = max(1, round(grey.width * IMAGE_HEIGHT / grey.height))
    return grey.resize((w, IMAGE_HEIGHT), Image.LANCZOS)


def predict(im: Image.Image, mode: str, beam: int) -> dict:
    raster = rasterise(flatten_to_white(im), mode)
    if raster.height != IMAGE_HEIGHT:
        raise ValueError(f"preprocessing produced height {raster.height}, "
                         f"expected {IMAGE_HEIGHT}")

    arr = np.asarray(raster, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr)[None, None].to(STATE["device"])
    widths = torch.tensor([x.shape[3]], device=STATE["device"])
    with torch.no_grad():
        ids = STATE["model"].predict(x, widths, beam_width=beam, max_len=120)[0]
    # Tokens carry no separators. The vocab has a space token and the model
    # usually emits it where LaTeX needs one, but "\cdot" followed by "r" with
    # no space is the undefined command "\cdotr". The token boundary is known
    # here and nowhere downstream (a regex on the joined string cannot tell
    # where a command name ends), so this is the one place to insert it: after
    # a letter-named command when the next token starts with a letter.
    toks = [STATE["id_to_tok"].get(i, "?") for i in ids]
    parts = []
    for j, t in enumerate(toks):
        parts.append(t)
        nxt = toks[j + 1] if j + 1 < len(toks) else ""
        if t.startswith("\\") and t[-1].isalpha() and nxt[:1].isalpha():
            parts.append(" ")
    latex = "".join(parts)

    buf = io.BytesIO()
    raster.save(buf, format="PNG")
    return {"latex": latex,
            "raster": base64.b64encode(buf.getvalue()).decode(),
            "width": raster.width, "mode": mode, "beam": beam}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):            # quieter console
        pass

    def _send(self, code, body, ctype="application/json"):
        payload = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/info":
            self._send(200, json.dumps({
                "checkpoint": STATE["name"], "exprate": STATE["exprate"],
                "epoch": STATE["epoch"], "height": IMAGE_HEIGHT,
                "device": STATE["device"]}))
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if self.path != "/predict":
            return self._send(404, json.dumps({"error": "not found"}))
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n))
            raw = base64.b64decode(req["image"].split(",")[-1])
            im = Image.open(io.BytesIO(raw))
            self._send(200, json.dumps(predict(
                im, req.get("mode", "inkml"), int(req.get("beam", 5)))))
        except Exception as exc:
            traceback.print_exc()
            self._send(400, json.dumps({"error": f"{type(exc).__name__}: {exc}"}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(REPO / "best_model_can_96px.pt"))
    p.add_argument("--processed", default=str(REPO / "processed-96px-ready"),
                   help="only its vocab.json is read")
    p.add_argument("--device", default="auto",
                   help="cuda, mps, cpu, or auto (best available; a cloud box "
                        "without a GPU silently gets cpu)")
    p.add_argument("--host", default="127.0.0.1",
                   help="bind address; a container must use 0.0.0.0 or nothing "
                        "outside it can connect")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()

    if a.device == "auto":
        a.device = ("cuda" if torch.cuda.is_available()
                    else "mps" if torch.backends.mps.is_available()
                    else "cpu")

    load_model(a.checkpoint, a.processed, a.device)
    print(f"\n  open  http://localhost:{a.port}\n  stop  Ctrl+C\n")
    try:
        HTTPServer((a.host, a.port), Handler).serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
