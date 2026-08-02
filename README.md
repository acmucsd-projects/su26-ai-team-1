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

1. Resolve the stride-16 vs. stride-32.
2. Get the real `vocab_size` and finalize the PAD/BOS/EOS/UNK ids.
3. Add training loop, loss function, and inference (greedy + beam search).
4. Wire up a real `Dataset`/`DataLoader` against `processed/labels/*.jsonl`.
5. Complete the PosFormer additions.
