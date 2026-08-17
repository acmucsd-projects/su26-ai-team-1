# su26-ai-team-1

https://drive.google.com/file/d/13TUkaJ0AaIobDGX0DzM3Dtjv0d9ndU4P/view?usp=drive_link

Link to processed dataset as a .zip on Drive

## Data pipeline

`mathwriting_preprocessing.ipynb` is the full MathWriting data pipeline: it downloads the dataset (if not already present locally), converts InkML strokes into normalized grayscale PNGs, tokenizes labels into a frozen vocabulary, and defines the training-time augmentation utility (rotation, shear, stroke thinning, Gaussian blur).

Run the notebook top to bottom. It's self-contained:
- Requires `pillow`, `numpy`, `matplotlib`; the notebook installs any that are missing. `pycairo` gives higher-fidelity rendering but is optional — if it isn't available (common on Windows without a system Cairo library), the notebook falls back to a supersampled Pillow renderer automatically.
- On first run it downloads the public MathWriting excerpt dataset into `mathwriting-2024-excerpt/`. Re-running skips the download if that directory already exists. To use the full dataset instead of the excerpt, edit `DATASET_URL`/`ROOT_DIR` in the download cell.

**Outputs** (all git-ignored, regenerate by re-running the notebook):
- `processed/images/{split}/{sample_id}.png` — clean (unaugmented) grayscale renders, one per split (`train`/`valid`/`test`/`synthetic`/`symbols`).
- `processed/labels/{split}.jsonl` — one record per sample: tokens, token ids, image width/height, etc.
- `processed/vocab.json` — token vocabulary, built from `train` + `synthetic` only.
- `processed/metadata.json` — the config a run used (rendering settings, augmentation policy), for reproducibility.
- `augmented_demo/` — a few `train` samples rendered with the augmentation utility, for visual QA only (not training data).

**Augmentation policy:** applied online at training time (e.g. from a `Dataset.__getitem__`) via `render_with_augmentation`, defined in the notebook — never baked into the PNGs in `processed/`. Each of the four transforms (rotation, shear, thinning, blur) rolls independently with its own probability and parameter range, so a training `Dataset` should call this fresh per `train` sample per epoch rather than reading from a fixed augmented copy. See the "Augmentation: policy" section in the notebook for the exact config and rationale.

`mathwriting_code_examples.ipynb` is the unmodified official MathWriting example notebook, kept for reference only.
# MobileNet Visual Encoder (`mobilenet_encoder.py`)

The visual encoder component of the handwritten-math-to-LaTeX pipeline. It extracts visual features from preprocessed handwritten math images and converts them into sequence tokens for cross-attention in the Transformer decoder.

---

## Repository Files

* `mobilenet_encoder.py`: Core encoder class (`MobileNetEncoder`) that outputs visual tokens for the Transformer decoder.
* `mobilenet_stride_check.py`: Diagnostic helper script used to verify layer output shapes and confirm the Stride-16 cutoff index.

---

## Pipeline Context

```text
Raw Image ──► Preprocessing ──► MobileNet Encoder ──► [CAN Counting Module] ──► Transformer Decoder ──► LaTeX Output
```

## Running the Stride Verification Script

Before building the encoder, run `mobilenet_stride_check.py` to inspect the spatial downsampling across MobileNetV3 blocks:

```bash
python3 mobilenet_stride_check.py
```

What it does:

* Loops through `MobileNetV3-Large.features` block-by-block using a sample input shaped 64px tall, 256px wide.
* Prints the image height after each block, so you can see exactly where it shrinks.
* Confirms that block 12 is the last block where the height is 4px (stride-16) — block 13 shrinks it further to 2px (stride-32).

## Architectural Decisions

* **Pretrained backbone:** Loads `MobileNetV3-Large` with ImageNet weights instead of training from scratch. General visual features (edges, strokes, loops) transfer well to handwritten math symbols, saving a lot of training time.

* **Stride-16 cutoff:** Cuts the backbone off at block index 12 (`features[:13]`). This keeps more spatial detail than going further to stride-32, which would compress the image too much and risk losing fine stroke details.

* **Dynamic projection:** Converts the backbone's 112 output channels to `d_model` using a 1x1 convolution. This channel count is calculated automatically from a test pass instead of hardcoded, so if the cutoff index ever changes, this part doesn't need manual updates.

* **Sequence formatting:** Flattens the 2D grid of features (height × width) into a 1D sequence, which is the format `nn.TransformerDecoder` expects as input.

## Confirmed Specifications & Verification

* **Input size:** Fixed height of 64px, variable width (padded per batch) — matches Jaeho's `input-preprocessing` branch.
* **Layer output shape:** Verified with a sample input of shape (1, 3, 64, 256): backbone output is (1, 112, 4, 16) — meaning 112 channels, height shrunk from 64px to 4px (stride-16), width shrunk from 256px to 16px.
* **End-to-end encoder test:** Verified with a dummy batch of shape (2, 3, 64, 256): output is (2, 64, d_model) — batch size 2, sequence length 64 (4 × 16 flattened), and each token sized to match `d_model`.

## Open Dependencies

> [!NOTE]
> **`d_model` status: resolved for now, pending final confirmation**
>
> * Encoder default: `d_model = 256`
> * Decoder default (`baseline_decoder.py`): `d_model = 256`

> [!NOTE]
> **Stride consensus**
>
> * Confirm with the team that stride-16 remains the final target design relative to stride-32.

## Next Steps

1. **Confirm with Adam:** double-check `d_model = 256` is the agreed final value
2. **End-to-end test:** pass `MobileNetEncoder`'s output directly into `LatexDecoder.forward()` via the `memory` argument.
3. **CAN integration:** coordinate with Hsin-Yu to connect the counting module's feature map between the encoder and decoder.

## References

* [PyTorch MobileNetV3 Documentation](https://pytorch.org/vision/main/models/mobilenetv3.html)
* [MobileNetV2 Autoencoder for Feature Extraction](https://medium.com/@abbesnessim/mobilenetv2-autoencoder-an-efficient-approach-for-feature-extraction-and-image-reconstruction-9c70ba58947a)
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
# Handwritten Math Input Image Preprocessing Pipeline

`inputpreprocessing.py` is a preprocessing pipeline for input images in our handwritten-math-to-LaTeX model.
Takes a scanned or camera-photographed image of a handwritten equation and produces a normalized tensor ready for a MobileNet encoder.

## Pipeline

```
input image (scan / photo)

-> find the location of equations in the image

-> identifying the edges of writing surface(paper, whiteboard, post-it note, etc.)
   (surface quadrilateral when available; otherwise vanishing-point partial
   rectification when only one direction of page edges is reliable)

-> crop to the equation's ink extent by removing unncessary surrounding blank space

-> binarization
   (Otsu or adaptive threshold, polarity-normalized)

-> resize to `64 x W`
   (height is always 64; width is proportional and remains variable)

-> MobileNet input tensor
   (normalized, CHW, batched)
```

| Step                   | Problem it solves                                                                                                                                                                                       |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Perspective correction | Corrects photographed paper when reliable page/surface geometry exists. Clean MathWriting-style white canvases are detected and deliberately left unwarped, since their pen strokes are not page edges. |
| Ink crop               | Removes photo/page whitespace while retaining disconnected symbols in one expression.                                                                                                                   |
| Binarization           | Removes paper texture, lighting gradients, and camera noise/color, leaving just the ink/marker strokes.                                                                                                 |
| Height-only resize     | A fixed square would stretch wide equations and distort symbols. The pipeline uses `64 x W`; use `pad_mobilenet_batch` only when batching samples with different widths.                                |
| MobileNet formatting   | Converts the image array into the float tensor shape a MobileNet encoder expects.                                                                                                                       |

## References

### Perspective correction

- **Source of the _approach_:** the [Im2Latex project page](https://sujayr91.github.io/Im2Latex/)
  describes correcting perspective distortion: Canny edge detection → Hough transform to find
  the clipboard/page boundary lines → intersecting those lines for the 4
  corners → homography to warp the corners into a rectangle → binarize.

- **Canny + `cv2.HoughLinesP` usage:** follows the
  [OpenCV Hough Line Transform tutorial](https://docs.opencv.org/4.x/d9/db0/tutorial_hough_lines.html).

- **Four-point homography (`cv2.getPerspectiveTransform` + `cv2.warpPerspective`):**
  standard OpenCV document-rectification pattern, e.g.
  [learnopencv's perspective-correction.py](https://github.com/spmallick/learnopencv/blob/master/Homography/perspective-correction.py)

### Binarization

- **Otsu (`cv2.THRESH_OTSU`) and adaptive thresholding
  (`cv2.ADAPTIVE_THRESH_GAUSSIAN_C`):** standard OpenCV binarization methods,
  in general use across OCR preprocessing.
