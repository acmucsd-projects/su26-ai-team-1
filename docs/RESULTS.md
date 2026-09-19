# Experimental Results

Handwritten mathematical expression recognition on MathWriting. This records
what each configuration scored, how it was measured, and which comparisons are
controlled enough to support a conclusion.

Every figure below was measured by re-running the checkpoint through a single
evaluation path, on the full validation split, against clean renders, at a
fixed batch size. Numbers quoted from training logs are not used.

Last updated: 2026-09-14.

## Summary

1. **Input resolution 64px → 96px: +5.8 points ExpRate.** Controlled
   comparison — identical recipe, only resolution differing. This is the change
   that mattered.
2. **CAN counting objective: no measurable effect.** Matched ablation at 96px:
   baseline 0.7071, CAN 0.7054. A 27-sample difference out of 15,674.
3. **Warm-start quality carries through.** Two runs identical but for their
   starting checkpoint finished ~0.5 points apart.
4. **ExpRate depends on evaluation batch size** — up to 1.7 points. A
   measurement artifact of Squeeze-Excitation pooling over padding, and a
   correctness issue in its own right. See
   [batch-size sensitivity](#batch-size-sensitivity).

Best model: **`best_model_can_96px.pt`, 71.15% ExpRate.**

## 1. Setup

| | |
|---|---|
| Encoder | MobileNetV3-Large, cut at `features[:13]` (stride 16) |
| Decoder | PosFormer, 3 layers, d_model 256, 8 heads |
| Parameters | 7.37M; 8.27M with the CAN counting head |
| Vocabulary | 258 tokens, frozen across every run |
| Train split | 229,864 real samples (`train`); `synthetic` adds 388,735 |
| Validation | 15,674 samples (`valid`), clean renders |
| Hardware | Apple silicon, 48 GB unified memory (MPS); GPU instance for later runs |

**Metric.** ExpRate is exact-match accuracy over the whole expression;
ExpRate≤1 and ≤2 allow one and two token errors. Confidence intervals are 95%
normal-approximation on the binomial.

**Evaluation protocol, applied uniformly.** Full 15,674-sample split, clean
renders, greedy decoding, **batch size 16**, `scripts/eval_all.py`.
Architecture is read from the weights (`use_can` from the counting keys,
stride from encoder channel count, PE width from the stored table) so no
checkpoint can be evaluated as something it is not.

## 2. Measurement methodology

Three properties of this setup make naive comparisons wrong. All three
produced an incorrect conclusion at some point during the project.

### Subset vs full split

Training-time validation was capped at 1,024 samples for speed (±3% sampling
error, fixed non-random subset). Those numbers are **not** comparable to
full-split figures (±0.75%). The long-standing "62.9%" project baseline is one
such number — 644/1024 exactly. Nothing in this document uses subset metrics.

### Clean vs augmented validation

The 96px archive as delivered bakes augmentation into every split, validation
included: 79% of `valid` images are rotated, sheared or blurred. Measured
penalty, same checkpoint both ways:

| | augmented | clean | Δ |
|---|---|---|---|
| `best_model_can_96px.pt` | 0.7054 | 0.7115 | +0.61 |
| `best_model_baseline_96px.pt` | 0.7011 | 0.7071 | +0.60 |

A consistent ~0.6 points. Validation must read clean renders.

A related trap: augmentation changes image **width**, and width drives the
cross-attention padding mask. 9,403 of 15,674 validation widths differ between
clean and augmented renders, so label files must be regenerated alongside the
images, never reused.

### Batch-size sensitivity

ExpRate is **not** invariant to evaluation batch size. Same checkpoint
(`best_model_96px.pt`), same data, greedy decoding:

| batch | ExpRate | correct |
|---|---|---|
| 32 | 0.6356 | 9,963 |
| 16 | 0.6434 | 10,084 |
| 8 | 0.6522 | 10,222 |

Monotonic, 1.66 points end to end. The cause is architectural: MobileNetV3
contains five Squeeze-Excitation blocks, each of which global-average-pools
across the entire feature map — **including padding**. Images are padded to
their batch's widest member, so a larger batch means more padding, a more
diluted SE pooling average, and different channel scaling applied to the real
ink. The encoder is therefore not translation-local, and padding is not free.

Consequences:

- Numbers are comparable only at equal batch size. Everything here uses 16.
- `--bucket` (grouping similar widths) improves **accuracy**, not only
  throughput, because it reduces padding. It is documented as a speed
  optimisation only.
- `MobileNetEncoder` now masks SE pooling to each sample's true width
  (`mask_padding=True`, default). Real-column features vary ~100x less across
  padding widths on random input. NOT yet re-evaluated on a trained checkpoint,
  and existing checkpoints were trained with diluted SE, so expect their
  numbers to shift until retrained/fine-tuned.

## 3. Results

All rows: `valid`, n = 15,674, clean renders, greedy, batch 16.

### 64px models

| Checkpoint | Configuration | ExpRate | ≤1 | ≤2 | val_loss | ep |
|---|---|---|---|---|---|---|
| `best_model_scratch_aux.pt` | train-only, scratch + aux | 0.5855 ±0.0077 | 0.7297 | 0.8019 | 0.1911 | 26 |
| `best_model_full.pt` | + synthetic + fine-tune | 0.6374 ±0.0075 | 0.7699 | 0.8376 | 0.1577 | 25 |
| `best_model_can.pt` | + CAN, 30 epochs | 0.6629 ±0.0074 | 0.7910 | 0.8526 | 0.1566 | 30 |

### 96px models

| Checkpoint | Configuration | ExpRate | ≤1 | ≤2 | val_loss | ep |
|---|---|---|---|---|---|---|
| `best_model_96px_synth.pt` | +1 epoch subsampled synthetic | 0.6284 ±0.0076 | 0.7714 | 0.8385 | 0.1470 | 1 |
| `best_model_96px.pt` | train-only, scratch + aux | 0.6434 ±0.0075 | 0.7749 | 0.8424 | 0.1575 | 30 |
| `best_model_can_96px_from_full.pt` | CAN, warm from `full` | 0.7054 ±0.0071 | 0.8214 | 0.8782 | 0.1317 | 28 |
| `best_model_baseline_96px.pt` | baseline, warm from `full` | 0.7071 ±0.0071 | 0.8212 | 0.8799 | 0.1302 | 27 |
| **`best_model_can_96px.pt`** | **CAN, warm from `can`** | **0.7115 ±0.0071** | **0.8257** | **0.8815** | 0.1353 | 26 |

Two independent reproductions confirm the evaluation path is deterministic:
`best_model_can_96px.pt` returned 0.7115 on two separate runs, and evaluating
it on the augmented split returned 11,057/15,674 = 0.7054, matching the value
stored in its own checkpoint exactly.

## 4. Controlled comparison: input resolution

`best_model_scratch_aux.pt` and `best_model_96px.pt` differ **only** in input
height. Both are train-only, from scratch, with the auxiliary objective,
width-bucketed at 9,578 batches/epoch, same validation cap, same vocabulary,
same evaluation.

| | 64px | 96px | Δ |
|---|---|---|---|
| ExpRate | 0.5855 | 0.6434 | **+5.79** |
| ExpRate≤1 | 0.7297 | 0.7749 | +4.52 |
| ExpRate≤2 | 0.8019 | 0.8424 | +4.05 |
| val_loss | 0.1911 | 0.1575 | −0.0336 |

Confidence intervals do not overlap; the gap is roughly 7× the margin of
error, and every metric moves the same direction. This supports the hypothesis
from error analysis that the 64px input limited the model's ability to resolve
fine detail.

**Cost.** 96px raises inference time ~40% and training memory ~2.4×: the
feature map grows from 4 to 6 rows and images run ~1.57× wider at the same
aspect ratio.

## 5. CAN ablation

Two 96px runs from the same warm start (`best_model_full.pt`), same schedule,
same data, differing only in whether the counting objective was active:

| | ExpRate | correct | val_loss |
|---|---|---|---|
| baseline (no CAN) | 0.7071 ±0.0071 | 11,083 / 15,674 | 0.1302 |
| + CAN | 0.7054 ±0.0071 | 11,056 / 15,674 | 0.1317 |

**A 27-sample difference out of 15,674** — 0.17 points, with confidence
intervals almost entirely overlapping. CAN neither helps nor hurts under a
matched comparison.

Note the "baseline" checkpoint does contain four `counting_module.*` tensors:
the head was constructed but its loss was never backpropagated. Its counting
loss stayed at 0.2601 versus 0.0131 for the CAN run, confirming the head was
untrained. Functionally it is a no-CAN model carrying dead parameters.

This supersedes two earlier readings. A first CAN run scored 0.7115 and
appeared to prove CAN's value, but it warm-started from a CAN checkpoint and
so carried extra training history. An earlier isolation attempt
(`run_can_isolated.log`) is also uninformative: CAN warm-started from a 62.5%
model fell to 58.2% over four epochs, but the aux-only control behaved
identically (57.6%), so that experiment measured insufficient fine-tuning
rather than either objective.

## 6. Warm-start quality carries through

The three 96px runs above differ mainly in where they started:

| Run | Started from | Start ExpRate | Final ExpRate |
|---|---|---|---|
| `best_model_can_96px.pt` | `best_model_can.pt` | 0.6629 | **0.7115** |
| `best_model_baseline_96px.pt` | `best_model_full.pt` | 0.6374 | 0.7071 |
| `best_model_can_96px_from_full.pt` | `best_model_full.pt` | 0.6374 | 0.7054 |

A 2.55-point better starting checkpoint produced a ~0.5-point better finish
after ~27 epochs. The advantage is damped but not erased, which is the more
useful finding from the experiment that was designed to test CAN.

## 7. Secondary observations

- **Convergence.** The 96px train-only run reached 0.588 (subset metric) by
  epoch 15 and 0.605 by epoch 30 — the last ten epochs bought about half a
  point. Roughly 20 epochs is sufficient; 30 is not economical.
- **Encoder stride.** Stride 8 at 64px (4 → 8 feature rows) reached 0.504
  after 10 epochs, below the stride-16 trajectory, at ~2.2× the per-epoch
  cost. Raising input resolution is the better way to buy vertical detail.
- **Synthetic data at 96px is untested.** The one 96px synthetic run is a
  single epoch on synthetic subsampled to 38%, and scores below the train-only
  model. Three attempts to train on the full 618,599-sample set failed on a
  48 GB machine — one killed by the OS at 46 GB swap, the others running 5–8×
  below normal throughput. This configuration needs a GPU instance.
- **Lower loss did not always mean higher ExpRate.** `best_model_96px_synth.pt`
  has a better val_loss than `best_model_96px.pt` (0.1470 vs 0.1575) but scores
  1.5 points lower on exact match: better average token confidence, more
  sequences missed by one or two tokens.

## 8. Reproduction

Build the data from raw InkML, with clean validation:

```bash
PYTHONPATH=. python scripts/build_archive.py \
  --raw mathwriting-2024 --out processed-96px-ready \
  --vocab processed/vocab.json \
  --augment train,synthetic,symbols --clean valid,test
```

Train:

```bash
python run_train.py --processed processed-96px-ready \
  --train-split train --val-split valid --aux --bucket \
  --epochs 20 --batch-size 24 --device cuda --workers 6 \
  --limit-val 1024 --checkpoint best_model_96px.pt
```

Reproduce any table above:

```bash
PYTHONPATH=. python scripts/eval_all.py \
  --processed processed-96px-ready --split valid --height 96 \
  --beam 1 --batch-size 16 *.pt
```

`--height` is the one property not recoverable from the weights — at stride 16
a 64px and a 96px model have identical tensor shapes — so 64px and 96px models
must be evaluated in separate runs against their own archives. Checkpoints
trained outside this branch may use a different positional-encoding width
(`best_model_can_96px.pt` has `max_w=320` against this branch's 100); the
script reads it from the checkpoint.

## 9. Open questions

1. **Mask the Squeeze-Excitation pooling to each sample's true width.** The
   batch-size result implies padding is degrading real predictions. This is
   the most concrete accuracy improvement identified here, and it is untested.
2. **Does synthetic pretraining help at 96px?** Untested at full scale;
   requires a GPU instance.
3. **Does beam search still pay at 96px?** Every figure here is greedy. Beam-5
   historically added a few points and is the cheapest remaining gain.
4. **Is 96px the optimum, or does 128px go further?** The 64→96 gain was large
   enough that the curve has not obviously flattened.
