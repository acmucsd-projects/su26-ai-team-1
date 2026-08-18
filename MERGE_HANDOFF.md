# Merge handoff — all four branches into one

**Status:** all four feature branches are merged on the local branch `merge-all`, and the full pipeline runs end to end. `main` is still at "Initial commit" and nothing has been pushed yet.

The merge itself was clean — every branch added **disjoint `.py` files**, so there were **zero conflicts in any source file**. Only `README.md` (all four) and `.gitignore` (two) collided.

But wiring the branches together surfaced several bugs that were invisible while each branch was developed alone. Most of them would not have thrown an error — they'd have quietly trained a worse model. Those are listed below by branch so each owner can review their own.

---

## 1. What got merged

| Branch | Owner | Adds |
|---|---|---|
| `data-preprocessing` | r4tangUCSD | `mathwriting_pipeline.py`, `dataset.py`, 2 notebooks, `requirements.txt` |
| `input-preprocessing` | Jaeho Shim | `inputpreprocessing.py` (1533 lines) + test images |
| `model-encoder` | Sunisa Saeli | `mobilenet_encoder.py`, `mobilenet_stride_check.py` |
| `model-decoder` | Adam Connor | `baseline_decoder.py`, `latex_decoder.py`, `hmer_model.py`, `train.py` |

The pieces fit together better than they had any right to. `hmer_model.py` was already written against `mobilenet_encoder`, and `inputpreprocessing.py` ends with `to_mobilenet_input()` / `pad_mobilenet_batch()`. Everything agrees on one contract:

> **Images are exactly 64px tall, width varies.** The encoder's stride of 16 turns that into a feature grid of height `feat_h = 4`.

---

## 2. Bugs found and fixed

### `data-preprocessing` — the notebook would not open

`mathwriting_preprocessing.ipynb` was committed with **unresolved `git stash` conflict markers** in its metadata block:

```
<<<<<<< Updated upstream
   "version": "3.12.6"
=======
   ...
>>>>>>> Stashed changes
```

This made the file invalid JSON, so Jupyter could not open it at all. Since this notebook is what generates `processed/`, training was blocked on it. **Fixed** — kept the fuller side. This was already broken on `origin/data-preprocessing`; it did not come from the merge.

**Please check:** when you `git stash pop` on a notebook, the markers land inside the JSON. Worth re-running the notebook once to confirm nothing else was lost in that stash.

### `data-preprocessing` — three bugs in `dataset.py`'s `collate_fn`

`collate_fn` was defined **twice** — once in `dataset.py` and once in `train.py` — with incompatible signatures. `dataset.py`'s copy had drifted from the contract every consumer reads:

| Problem | Was | Now |
|---|---|---|
| Wrong stride | `ENCODER_STRIDE = 32` | `16` — imported from `hmer_model` |
| Wrong pad color | `torch.zeros` (black) | `1.0` (white) |
| Wrong key names | `token_ids`, `widths`, `feature_mask` | `tokens`, `true_widths` |

- **Stride** — the encoder cuts MobileNet at `features[:13]`, which is stride-**16**, not 32. Width rounding and the feature mask were being computed at half the real resolution.
- **Pad color** — the decoder masks padding out, but the **encoder still convolves across it**. Black padding puts a hard fake edge right next to the real ink, which bleeds into the last few real feature columns. `train.py`'s copy had this right and documented why.
- **Key names** — `hmer_train_step`, `validate`, and `model.predict` all read `tokens` / `true_widths`. Connecting the real dataset would have failed immediately on the missing `true_widths` key.

Also removed `feature_mask` entirely: it was **inverted** relative to `nn.Transformer`'s `key_padding_mask` convention (`True` meant "real content" instead of "ignore this position") and nothing consumed it. The model derives the mask itself from `true_widths` via `widths_to_memory_padding_mask`.

**`dataset.py` now owns the single definition; `train.py` imports it.** Added an assert for a silent-corruption case: if the JSONL's `width` ever disagrees with the actual PNG width, you get an error instead of a quietly misaligned attention mask.

### `data-preprocessing` — `requirements.txt` could not install at all

Three separate problems:

1. **Missing `torchvision`** (the MobileNetV3 backbone) and **`opencv-python`** (all of `inputpreprocessing.py`). Both are hard imports.
2. **`pycairo` was pinned as a hard requirement.** It builds from source and needs system Cairo + `pkg-config`, so `pip install -r requirements.txt` **fails outright on macOS and Windows before installing anything else**. This contradicts the code: `mathwriting_pipeline.py` detects `CAIRO_AVAILABLE` and falls back to a supersampled Pillow renderer precisely because Cairo is expected to be missing. Moved to an optional note with `brew install cairo pkg-config` instructions.
3. A UTF-8 BOM on the first line.

**All fixed.** This one was blocking every teammate on a non-Linux machine.

### `data-preprocessing` — the processed archive alone could not train

`MathWritingDataset` hard-required `raw_dir` for the `train` split, because train re-renders augmentations from source InkML on every `__getitem__`. That means the 2 GB `processed/` archive on Drive was **not sufficient to train** — you also needed the ~2.9 GB raw InkML tree.

Added an explicit `augment=` parameter. The default is unchanged (train augments, and still requires `raw_dir`), but `augment=False` reads the clean PNGs from `processed/images/train/` like every other split. The trade-off is real and documented in the docstring: without online augmentation the model sees each image identically every epoch and will overfit sooner. It's a "get a baseline running" setting, not the one to report numbers from.

### `input-preprocessing` — ~21 MB of generated files committed

`test_images/` (21 MB of JPGs), `__pycache__/*.pyc`, and three `.DS_Store` files were tracked in git. The `.pyc` files go stale immediately and are pure noise in diffs. **Removed from tracking and added to `.gitignore`.** The JPGs are left in place since they're useful fixtures — move them to Drive if the repo size becomes a problem.

### All four — `README.md`

Each branch wrote its own README and two of them replaced the project title with their own `# H1`. **Replaced with one unified README** with an architecture diagram, a file table, and a quickstart; each branch's original notes are preserved as sections underneath.

---

## 3. Still needs a decision — not mine to make

These are real open questions. Nothing is blocked on them for a first training run, but they should be settled.

**The PosFormer auxiliary loss is not actually being trained.** `hmer_train_step` covers the baseline cross-entropy objective only. The aux losses live in `posformer_train_step` (`latex_decoder.py`), which takes a feature map rather than images, so the two don't compose yet. `hmer_model.py:225` already flags unifying them. **Right now the model is built with ARM + position forest enabled but trains without the forest objective.** — *Adam*

**`collate_fn` duplication was resolved in `dataset.py`'s favor.** If that was the wrong call, the contract to change is the one in `hmer_train_step`'s docstring. — *Ryan + Adam*

**The CAN counting module has no branch.** The encoder README's pipeline diagram includes `[CAN Counting Module]` between encoder and decoder, but nothing implements it. Is it still in scope?

**`d_model = 256` is consistent** across `mobilenet_encoder` and `HMERModel` — the encoder README lists this as an open item, but both defaults already agree. Consider it closed unless someone disagrees.

**Stride-16 vs stride-32** is listed as needing team consensus in the encoder README. The code now assumes 16 in three places. If anyone still wants to test 32, change `ENCODER_STRIDE` in `hmer_model.py` — `dataset.py` imports it from there, so they can't drift apart again.

**One preprocessing edge case:** `test_images/test3_full.JPG` comes out **33 px wide**. Perspective correction falls back to `identity` at 0.0 confidence ("only 2/5 equation hint points lie inside quadrilateral"), then a tall crop gets squeezed to 64px height. That's almost certainly too narrow to decode. — *Jaeho*

### Two performance problems worth fixing before any long run

**Padding wastes roughly 2.3× of the compute.** Measured on the real `train` split: median image width is 134 px, but a random batch of 32 pads to the batch maximum, which averages **330 px**. Every sample in the batch then gets convolved across that full padded width.

```
width  : min 9   p50 134   p90 247   p99 355   max 824
tokens : min 3   p50 18    p90 32    p99 47    max 99
```

A length-bucketed sampler (group similar widths into the same batch) is close to a free 2× speedup and is the single highest-value optimization available. — *Ryan + Adam*

**`fit()` never caps validation.** It calls `validate(model, val_loader, ...)` without passing `max_batches`, so every epoch runs autoregressive generation over the **entire** validation split. `validate()` already supports `max_batches` — `fit()` just doesn't expose it. With the full 15,674-sample `valid` split that's hours per epoch spent on validation alone, dwarfing the training time. Either thread `max_batches` through `fit()`, or pass a `Subset` as the val loader. — *Adam*

---

## 4. How to train the model

### Step 1 — Environment

⚠️ **Do not use the system `python3`.** On macOS with Homebrew that's now Python 3.14, which has **no PyTorch wheels**. Use 3.12:

```bash
cd su26-ai-team-1
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 2 — Build the dataset

Run `mathwriting_preprocessing.ipynb` top to bottom. It writes `processed/vocab.json`, `processed/labels/*.jsonl`, and `processed/images/`.

⚠️ **Check the download cell first.** The notebook is set to the **full** dataset (`mathwriting-2024.tgz`, **2.88 GB** compressed, considerably more once extracted), but the README describes an *excerpt*. They disagree.

For a first smoke test, use the excerpt — it's **1.5 MB**:

```python
DATASET_URL = "https://storage.googleapis.com/mathwriting_data/mathwriting-2024-excerpt.tgz"
ROOT_DIR = Path("mathwriting-2024-excerpt")
```

Switch to the full dataset once the loop is confirmed working.

### Step 3 — Train

The pieces are wired; this is the driver you need:

```python
import torch
from torch.utils.data import DataLoader
from dataset import MathWritingDataset, collate_fn, seed_worker
from latex_decoder import load_vocab_config
from hmer_model import HMERModel
from train import fit

cfg = load_vocab_config("processed/vocab.json")

# raw_dir is REQUIRED for train: it re-renders augmentations from source
# InkML every epoch rather than reading a fixed augmented copy.
train_ds = MathWritingDataset("train", raw_dir="mathwriting-2024")
val_ds   = MathWritingDataset("valid")

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                          collate_fn=collate_fn,
                          num_workers=4, worker_init_fn=seed_worker)
val_loader   = DataLoader(val_ds, batch_size=32,
                          collate_fn=collate_fn, num_workers=4)

model = HMERModel(cfg.vocab_size, structure_tokens=cfg.structure_tokens)

fit(model, train_loader, val_loader,
    epochs=30, device="mps",              # "cuda" on a GPU box, "cpu" as fallback
    checkpoint_path="best_model.pt", log_every=50)
```

**`device="mps"` works** — Apple GPU training is verified end to end on this stack.

Always pass `worker_init_fn=seed_worker` when `num_workers > 0`. Without it every worker process inherits the same RNG state at fork and draws **identical augmentations**.

### Step 4 — Reading the output

- **ExpRate stays 0.000 for a long time.** It's exact-match on free-running generation with no teacher forcing — the honest metric, and a harsh one. Watch `train_loss` and `token_acc` for early signs of life; ExpRate only becomes informative later.
- **Checkpoints save on best ExpRate, not best val_loss.** They don't always move together.
- **Validation uses greedy decoding by default** (`val_beam_width=1`) because generation dominates validation time. Use `beam_width=5` only for a number you'd actually report.

### Tips

**Freeze the encoder for your first real run.** It isolates whether the decoder works before adding MobileNet gradients to the mix, and it's much faster to debug. `hmer_model.py:199` documents this:

```python
for p in model.encoder.parameters():
    p.requires_grad = False
```

**Calling the decoder directly?** `greedy_decode` and `beam_search_batch` need `memory_height=4` passed through — they forward `**model_kwargs` straight to the decoder, which needs it to un-flatten memory back into a spatial grid.

**Don't re-wrap tokens with BOS/EOS.** `rec["token_ids"]` from the notebook already contains `[BOS, ...ids, EOS]`.

---

## 5. What was actually verified

Every self-test passes on the merged branch:

```
PASS  mobilenet_encoder.py     PASS  hmer_model.py
PASS  baseline_decoder.py      PASS  train.py
PASS  latex_decoder.py         PASS  inputpreprocessing.py (real photo)
```

Plus the two integration paths that no single branch could have tested alone:

- **Photo → tokens.** `test2_full.JPG` → preprocessing `(1,3,64,149)` → encoder → memory `(1,40,256)` → PosFormer → tokens. Perspective correction picked `surface_quad` at 0.94 confidence.
- **InkML → encoder.** A synthetic `\frac{a}{b}^{2}` → 11 tokens → rendered at `113×64` → memory `(1,32,256)`.

And most importantly, **the real `MathWritingDataset` → `collate_fn` → `fit()` path was run against a small synthetic `processed/` directory. It doesn't just execute — it learns.** Loss 2.99 → 0.87, ExpRate 0.000 → 1.000 on a trivially-learnable task. That's the signal that the encoder, the padding mask, and the decoder are genuinely aligned rather than merely running without crashing.

---

## 6. Landing it

Five commits sit on local `merge-all`. `main` is untouched and nothing has been pushed.

```bash
git checkout main && git merge merge-all
git push origin main
```

Given this touches three people's code and fixes real bugs in it, pushing `merge-all` and opening a PR is probably the better call:

```bash
git push -u origin merge-all
```
