## Status — decoder

All four files below are complete and reviewed, but **have never been trained**.
Verification is dummy tensors, shape/guard checks, and an equivalence test of
the position-forest parser against the official PosFormer implementation
(500/500 on random expressions). There is no loss curve yet.

**Four files, deliberately split:**

- `baseline_decoder.py` — plain transformer decoder, training loop, greedy +
  beam search. Runs standalone. This is what we ship if we run out of time.
- `latex_decoder.py` — imports the baseline and adds PosFormer's position
  forest and attention correction behind two independent toggles. With both
  off it is bit-identical to the baseline, so turning them off is a real
  fallback, not a rewrite.
- `hmer_model.py` — the encoder and decoder as one module, plus one optimizer
  covering both (encoder at a lower LR, since it starts from ImageNet weights
  and the decoder starts from random). The only file that knows both halves
  exist; it also handles the grayscale → RGB expansion and builds the padding
  mask. Verified end to end on dummy images with real MobileNetV3 weights.
- `train.py` — epoch and validation loops, checkpointing on best ExpRate
  (exact-match rate, the metric HMER is actually scored on — not the token
  accuracy the training loop prints). The dummy `Dataset` + `collate_fn` at the
  bottom are a runnable spec of the batch contract, for whoever writes the real
  one.

`mobilenet_encoder.py` is a temporary copy from `model-encoder`; replace it
with a proper branch merge once the 2D output and BatchNorm fix land.

**Expect a cleanup pass.** These files carry guards and dual-path handling for
things that aren't settled yet — encoder output layout, real `vocab_size`,
`max_len`, image sizing. Once the input/output shapes and model sizes are
final, a lot of that becomes dead weight and should be deleted rather than
maintained. Not doing it now because the checks are what catch interface drift
between our branches while they're still moving.

**Encoder interface is settled** (checked against `model-encoder`): stride 16,
so the feature grid is 4 × (W/16), flattened row-major to `[batch, H*W, 256]`.
Image height is fixed at 64px, which makes `feat_h=4` a constant; width varies
per batch and is derived from the sequence length, never assumed. The encoder
returns no padding mask, so the decoder builds one from each sample's true
pixel width.

**Blocking a first training run, in order:**

1. Run the preprocessing notebook. `processed/vocab.json` doesn't exist yet, so
   `vocab_size` is still unknown. `load_vocab_config()` reads it and also
   resolves the structure-token ids (`^`, `\frac`, `{` …) the position forest
   needs — no ids are hardcoded.
2. Write the `Dataset`/collate against `processed/labels/*.jsonl`. It must pass
   through each sample's true pixel width; cross-attention masking depends on
   it, and a missing mask degrades training silently rather than erroring.

**Needs an owner:** preprocessing writes 1-channel grayscale PNGs, but the
encoder expects 3-channel RGB (`[B, 3, 64, W]`) for its ImageNet weights.
Something has to expand 1 → 3, and neither branch does it today.
.

## Decoder — research notes & context

The `baseline_decoder` files is a WIP and contains documentation summarizing and explaining next steps

### Where the decoder fits

```
raw image -> preprocessing -> MobileNet encoder
          -> [counting module] -> decoder
          -> LaTeX sequence
```

### Transformer decoder, background

Each decoder layer runs three steps in order: masked self-attention (blocks
seeing future tokens), cross-attention (looks at the encoder's output), then
a feed-forward network — each wrapped in a residual connection + layer norm.
Layers are stacked (we use 3; more on why below).

### PosFormer — what it adds on top of a plain decoder
(source: [arXiv:2407.07764](https://arxiv.org/abs/2407.07764),
[SJTU-DeepVisionLab/PosFormer](https://github.com/SJTU-DeepVisionLab/PosFormer))

PosFormer builds on **CoMER**, an earlier sequence-based HMER decoder — it
doesn't reinvent the decoder, it adds two things:
- **Position forest**: encodes each LaTeX sequence into a forest structure
  where every symbol gets a position identifier (root/left/right-ish),
  trained as an auxiliary task alongside normal token prediction. Training-
  time only — removed at inference, so it costs nothing at test time.
- **Implicit attention correction**: refines attention weights using the
  model's own past attention, to reduce a common failure mode (attending to
  the wrong region, or repeating/skipping a symbol).
- PosFormer's own encoder is DenseNet, not MobileNet — the decoder-side
  ideas should transfer, but their specific hyperparameters don't
  necessarily apply to our setup.

### Next Steps

Done: stride resolved (16), PAD/BOS/EOS/UNK confirmed as 0/1/2/3, training
loop + loss + greedy/beam search added, PosFormer additions complete.

Remaining:

1. Run the preprocessing notebook to generate `processed/` — unblocks the real
   `vocab_size`.
2. Wire up a `Dataset`/`DataLoader` against `processed/labels/*.jsonl`,
   carrying each sample's true pixel width.
3. Resolve the grayscale → RGB channel mismatch between preprocessing and the
   encoder.
4. First end-to-end training run; set `max_len` from real label-length
   percentiles instead of the current arbitrary 200.
5. Tune the PosFormer auxiliary loss weights — currently the paper's
   0.25 / 0.25, untuned on our data.
