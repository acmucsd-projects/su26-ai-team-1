# Merge handoff — all four branches into one

**Status:** all four feature branches are merged on the local branch `merge-all`, the full pipeline runs end to end, and **the model has been trained on the real MathWriting data** — 13.5% ExpRate after 15 minutes on a subset (see §6). `main` is still at "Initial commit" and nothing has been pushed yet.

The merge itself was clean — every branch added **disjoint `.py` files**, so there were **zero conflicts in any source file**. Only `README.md` (all four) and `.gitignore` (two) collided.

But wiring the branches together surfaced **16 changes worth knowing about** — 11 genuine bugs plus 5 pieces of hardening. Most of the bugs would not have thrown an error; they'd have quietly trained a worse model. §2 lists every one of them with what changed and why, and ends with a per-owner breakdown so each person can review only their own.

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

## 2. Every change I made

Complete list. Grouped by whether it was **broken** (would have cost you correctness or blocked someone) or **hardening** (not a bug, but needed to run).

### Bugs fixed

| # | File | What changed | Why |
|---|---|---|---|
| 1 | `mathwriting_preprocessing.ipynb` | Removed unresolved `git stash` conflict markers from the metadata block | The markers made it invalid JSON, so **Jupyter could not open it at all** — and this notebook is what generates `processed/`. Pre-existing on `origin/data-preprocessing`, not caused by the merge. |
| 2 | `dataset.py` | `ENCODER_STRIDE` 32 → 16 (now imported from `hmer_model`) | The encoder cuts MobileNet at `features[:13]`, which is stride-**16**. Width rounding and the feature mask were being computed at **half the real resolution**. |
| 3 | `dataset.py` | Image padding `0.0` (black) → `1.0` (white) | The decoder masks padding out, but the **encoder still convolves across it**. Black padding puts a hard fake edge next to the ink that bleeds into the last real feature columns. |
| 4 | `dataset.py` | Batch keys `token_ids`/`widths` → `tokens`/`true_widths` | `hmer_train_step`, `validate`, and `model.predict` all read `tokens`/`true_widths`. Connecting the real dataset would have **failed immediately** on the missing `true_widths`. |
| 5 | `dataset.py` | Deleted `feature_mask` | It was **inverted** vs `nn.Transformer`'s `key_padding_mask` convention (`True` meant "real content", not "ignore") and nothing used it. The model derives the mask from `true_widths` itself. |
| 6 | `train.py` | Deleted its duplicate `collate_fn`; imports the one from `dataset.py` | `collate_fn` was defined **twice** with incompatible signatures. Now one definition, so the dummy self-test exercises the same padding path real training does. |
| 7 | `requirements.txt` | Added `torchvision` and `opencv-python` | Both are hard imports (MobileNetV3 backbone; all of `inputpreprocessing.py`). Nothing ran from a clean install. |
| 8 | `requirements.txt` | Removed the hard `pycairo` pin (kept as an optional note) | It builds from source and needs system Cairo + `pkg-config`, so `pip install -r requirements.txt` **died before installing anything** on macOS and Windows. The code already falls back to Pillow when Cairo is absent, so the pin contradicted it. **This blocked every non-Linux teammate.** |
| 9 | `requirements.txt` | Stripped a UTF-8 BOM from line 1 | Corrupted the first package name for some parsers. |
| 10 | repo | Untracked `__pycache__/*.pyc` and three `.DS_Store` files | Generated files — the `.pyc`s go stale immediately and are pure diff noise. |
| 11 | `README.md` | Merged four competing READMEs into one | Each branch wrote its own and two replaced the project title with their own `# H1`. Original notes are preserved as sections underneath. |

### Hardening — not bugs, but needed to actually run

| # | File | What changed | Why |
|---|---|---|---|
| 12 | `dataset.py` | Assert that the label record's `width` matches the actual image width | `true_widths` drives the cross-attention mask. If the JSONL and the PNG ever disagree, the mask silently misaligns instead of erroring — the worst kind of bug. |
| 13 | `dataset.py` | Added `augment=` parameter (default unchanged) | The `train` split hard-required `raw_dir`, so the 2 GB `processed/` archive on Drive was **not enough to train** — you also needed the ~2.9 GB raw InkML tree. `augment=False` reads the clean train PNGs instead. Trade-off: no online augmentation means overfitting sooner, so it's a "get a baseline running" setting. |
| 14 | `.gitignore` | Ignore `.venv/`, `*.pt`, `*.pth`, `history.json`, `*.zip`, `.DS_Store` | Without these, a 29 MB checkpoint and a 2 GB dataset zip get committed on the next `git add -A`. |
| 15 | *(new)* `run_train.py` | CLI training driver | Makes runs reproducible instead of pasted snippets. Has `--smoke` for a fast sanity check. |
| 16 | *(new)* `predict_samples.py` | Decodes a checkpoint's predictions vs ground truth | Loss numbers don't tell you *how* the model is wrong. This is what revealed that errors are symbol-identity, not structural. |

### Who should look at what

- **r4tangUCSD** (`data-preprocessing`) — #1–5, 7–9, 12, 13. Most of the list; the notebook and `dataset.py` had the substantive issues.
- **Adam Connor** (`model-decoder`) — #6, plus the two open items in §3 (aux loss, uncapped validation).
- **Jaeho Shim** (`input-preprocessing`) — #10, plus the `test3_full.JPG` edge case in §3.
- **Sunisa Saeli** (`model-encoder`) — nothing broken. The stride-16 decision is now load-bearing in three places, so it's worth confirming it's final.

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

## 6. First real training run — it works

Trained on the real MathWriting data (the `processed/` archive from Drive). **Scoped deliberately**: 20,000 of 229,864 train samples, 512 validation samples, 6 epochs, batch 32, on an Apple M-series GPU (`mps`). **15.1 minutes total.**

Two caveats on these numbers: augmentation was **off** (`augment=False`, since the raw InkML tree isn't available locally), and the **PosFormer auxiliary loss is not being trained** (see §3). So this is a floor, not a ceiling.

| Epoch | train_loss | val_loss | ExpRate | ExpRate ≤1 | secs |
|------:|-----------:|---------:|--------:|-----------:|-----:|
| 1 | 2.6075 | 2.0040 | 0.004 | 0.035 | 235 |
| 2 | 1.5906 | 1.3828 | 0.029 | 0.117 | 180 |
| 3 | 1.2060 | 1.0512 | 0.074 | 0.166 | 134 |
| 4 | 0.9922 | 0.8977 | 0.105 | 0.256 | 124 |
| 5 | 0.8773 | 0.8367 | **0.135** | 0.273 | 116 |
| 6 | 0.8283 | 0.8254 | 0.133 | 0.281 | 116 |

ExpRate is exact-match on free-running autoregressive generation — no teacher forcing — so **13.5% of validation expressions came out character-perfect** after 15 minutes on under 9% of the data. Val loss stayed below train loss the whole way: no overfitting yet, even without augmentation.

**Beam search barely helps at this stage.** On the same 512 samples: greedy 0.1348, beam-5 0.1484 (+1.4pp). Not worth the validation-time cost yet — keep `val_beam_width=1` until the model is stronger.

### What the predictions actually look like

```
MATCH  truth: \int_{0}^{\infty}\frac{sin(x)}{x}dx
       pred : \int_{0}^{\infty}\frac{sin(x)}{x}dx

MATCH  truth: \frac{d^{2}x}{dt^{2}}
       pred : \frac{d^{2}x}{dt^{2}}

MATCH  truth: T=(\begin{matrix}1&0\\ 1&0\end{matrix})
       pred : T=(\begin{matrix}1&0\\ 1&0\end{matrix})

  -    truth: R=\int_{0}^{\tau}X(s)ds
       pred : R=\int_{0}^{T}X(s)ds          <- tau read as T

  -    truth: \frac{dP}{d\tau}
       pred : \frac{dP}{dT}                 <- same confusion

  -    truth: o_{n}(R)
       pred : \sigma_{n}(R)                 <- symbol identity, structure right
```

**The error pattern is informative.** Nested fractions, integrals with both limits, matrix environments, and sub/superscript nesting all come out structurally correct. The failures are overwhelmingly **symbol identity** — `\tau`→`T`, `o`→`\sigma`, `\tilde`→`\overline`. That is exactly the profile of a model that has learned layout but hasn't yet had enough data to disambiguate similar glyphs, and it's the profile that more training data and augmentation fix.

Reproduce with:

```bash
python run_train.py --limit-train 20000 --limit-val 512 --epochs 6 \
                    --batch-size 32 --device mps --workers 4
python predict_samples.py --checkpoint best_model.pt --n 24 --beam 5
```

### What a full run costs

Measured throughput: **3.2 steps/s at batch 32 (~102 samples/s)** on `mps`. A full pass over all 229,864 training samples is **~37 min/epoch**, so 30 epochs is roughly **18–20 hours** — feasible on the Mac overnight, though a CUDA box would be far better. Fix the 2.3× padding waste first (§3) and that could drop to ~10 hours.

## 7. Landing it

Five commits sit on local `merge-all`. `main` is untouched and nothing has been pushed.

```bash
git checkout main && git merge merge-all
git push origin main
```

Given this touches three people's code and fixes real bugs in it, pushing `merge-all` and opening a PR is probably the better call:

```bash
git push -u origin merge-all
```
