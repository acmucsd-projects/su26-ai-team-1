# su26-ai-team-1

Handwritten math → LaTeX. Photo or InkML strokes in, LaTeX tokens out.

The inference path is preprocessing → MobileNet encoder → 2D positional encoding
→ Transformer decoder → LaTeX. During training, an auxiliary counting head
reads the encoder features in parallel with the decoder. Count predictions
are not fed into the decoder.

The decoder includes PosFormer-inspired components. The current training
objective optimizes sequence loss and CAN counting loss; it does **not**
include the position-forest auxiliary objective.

Every stage agrees on one contract: **images are exactly 96px tall, width varies**, and the
encoder's stride of 16 turns that into a feature grid of height `feat_h = 6`.

## Layout

| File | Role |
|---|---|
| `inputpreprocessing.py` | Inference input: photo → perspective-corrected, binarized 96px raster |
| `mathwriting_pipeline.py` | Training input: MathWriting InkML → normalized 96px render + augmentation |
| `dataset.py` | `MathWritingDataset` + collate over the rendered training data |
| `mobilenet_encoder.py` | MobileNetV3-Large truncated at stride-16 → visual tokens |
| `can_counting.py` | Auxiliary CAN head, symbol-count targets, and counting loss |
| `mobilenet_stride_check.py` | Diagnostic: prints per-block strides to justify the cutoff |
| `baseline_decoder.py` | Plain Transformer decoder, positional encodings, greedy/beam search |
| `latex_decoder.py` | PosFormer: attention refinement + position-forest aux task |
| `hmer_model.py` | Wires encoder → 2D pos-enc → decoder as one `HMERModel` |
| `train.py` | Training/validation loop, ExpRate metrics, checkpointing |
| `run_train.py` | Real-data training CLI, checkpoint initialization, and metrics export |
| `predict_samples.py` | Decode dataset samples and compare predictions with labels |

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# self-tests — each file runs standalone
python mobilenet_encoder.py     # encoder shapes
python baseline_decoder.py      # decoder, train step, beam search
python latex_decoder.py         # PosFormer, all four flag combinations
python hmer_model.py            # encoder+decoder end to end, gradient flow
python train.py                 # 3 epochs on dummy data

# preprocess a photo
python inputpreprocessing.py test_images/test2_full.JPG --debug-dir /tmp/dbg
```

## CAN auxiliary counting loss (`add-can`)

The MobileNet feature map now feeds two branches during training:

- Decoder branch: encoder features → positional encoding → decoder → sequence loss.
- Counting branch: encoder features → counting module → symbol-count loss.

Both losses update the shared encoder. `HMERModel.predict()` generates LaTeX
without running the counting head.

The combined objective is:

```text
total_loss = sequence_loss + counting_weight * counting_loss
```

`counting_weight` defaults to `0.1` and can be changed from the training CLI:

```bash
python run_train.py --smoke --counting-weight 0.1
```

Use `--counting-weight 0` to disable the counting loss contribution. The current
model still contains the counting head, and the training step still computes its
outputs. This is an objective ablation, not a head-free model architecture.

For the full 96px run on an AWS GPU instance, copy `processed-96px/` (including
its images, labels, and vocabulary) alongside the code and use a CUDA-enabled
PyTorch environment. Run:

```bash
python run_train.py --processed processed-96px --device cuda --epochs 30 --patience 0
```

These are also the driver defaults. This uses every record in the `train` and
`valid` splits, with no sample limits, and stops after epoch 30. Do not pass
`--smoke` or `--limit-train` / `--limit-val` for this run. The test split remains
reserved for evaluation. The best validation checkpoint is saved to
`best_model_96px_full.pt`, and the completed run's metrics to
`history_96px_full.json`. Add `--raw-dir /path/to/mathwriting` to enable online
InkML augmentation; otherwise the saved 96px PNGs are used.

To fine-tune a sequence-only checkpoint with the CAN objective, initialize the
encoder and decoder from it while leaving the new counting head random:

```bash
python run_train.py --processed processed-96px --device cuda \
  --init-checkpoint best_model_full.pt --counting-weight 0.1 \
  --checkpoint best_model_can_96px_from_full.pt \
  --history history_can_96px_from_full.json
```

The checkpoint and the selected dataset's `vocab.json` must use identical
token IDs and vocabulary size. Initialization from a CAN checkpoint also loads
its counting head; a legacy checkpoint without that head leaves it randomly
initialized. The loader rebuilds the deterministic image positional-encoding
buffer if its shape differs. This initializes a new training run; it does not
resume optimizer or scheduler state.

The default output name `best_model_96px_full.pt` does not mean CAN is disabled:
the default counting weight is 0.1. Set explicit checkpoint and history names
for each experiment to avoid overwriting earlier runs.

Building the real model needs the vocab produced by the data pipeline:

```python
from hmer_model import HMERModel
from latex_decoder import load_vocab_config

cfg = load_vocab_config("processed-96px/vocab.json")
model = HMERModel(cfg.vocab_size, structure_tokens=cfg.structure_tokens)
```

When calling `greedy_decode` / `beam_search_batch` directly, pass `memory_height=6`
(they forward `**model_kwargs` to the decoder, which needs it to un-flatten memory).


## Reported training results

The following values come from project training logs and checkpoint summaries.
They are validation results, not held-out test scores or a fresh evaluation of
this branch.

| Input height | Objective | Initialization | Reported validation ExpRate | Checkpoint |
|---|---|---|---|---|---|
| 64px | Sequence only | Not recorded here | 62.89% | `best_model_full.pt` |
| 64px | Sequence + CAN | 64px sequence-only checkpoint | 66.29% | `best_model_can.pt` |
| 96px | Sequence + CAN | 64px CAN checkpoint | 70.62% (epoch 30 log) | `best_model_can_96px.pt` |
| 96px | Sequence + CAN | 64px sequence-only checkpoint | Approximately 69.8% (epoch 28 checkpoint) | `best_model_can_96px_from_full.pt` |

ExpRate is the fraction of complete predicted token sequences that exactly
match the reference after special-token removal. It differs from teacher-forced
token accuracy. The reported 96px CAN runs used greedy decoding
(`beam_width=1`) and a counting weight of 0.1. Historical 64px decoding
settings should be checked against the original run configuration before a
controlled comparison.

The reported full-data runs used 229,864 training examples, 15,674 validation
examples, and a vocabulary of 258 tokens. Read the actual split counts and
vocabulary from the selected dataset when reproducing an experiment.

These runs differ in initialization and training history. The 64px-to-96px
comparison does not isolate resolution alone, and a completed comparable 96px
sequence-only baseline is not reported here. Checkpoints and processed data
must be obtained separately; they are not included with the source code.

## Decode validation samples

With a compatible 96px CAN checkpoint and its matching vocabulary:

```bash
python predict_samples.py --processed processed-96px --split valid \
  --checkpoint best_model_can_96px.pt --device cuda --beam 1 --n 15
```

This prints predictions, reference labels, and sample-level exact matches.
It evaluates only the requested sample subset, not the full validation set.
The script defaults to a decoding limit of 120 tokens; set `--max-len`
explicitly when matching another evaluation. Use `--beam 5` for beam search
and report it separately from greedy results.

`predict_samples.py` loads model weights strictly. Legacy 64px checkpoints
can have a different positional-encoding buffer and may lack CAN tensors;
they are not directly interchangeable with the current inference model.
Use the training driver's initialization path for the documented 64px-to-96px
fine-tuning workflow.

The photo command in Quickstart performs preprocessing only. Custom-image
prediction scripts such as `predict_image.py` and the handwriting/iPad
preprocessors are not included in this branch at the time of this update;
their commands are therefore not part of this quickstart.

---

# Data pipeline (MathWriting → training tensors)


`mathwriting_preprocessing.ipynb` is the full MathWriting data pipeline: it downloads the dataset (if not already present locally), converts InkML strokes into normalized grayscale PNGs, tokenizes labels into a frozen vocabulary, and defines the training-time augmentation utility (rotation, shear, stroke thinning, Gaussian blur).

The notebook examples below use `processed/`, while `run_train.py` defaults
to `processed-96px/`. Pass `--processed /path/to/dataset` to select the actual
output directory. Confirm the rendered images are 96px tall; renaming a 64px
dataset directory does not convert its images.

Run the notebook top to bottom. It's self-contained:
- Requires `pillow`, `numpy`, `matplotlib`; the notebook installs any that are missing. `pycairo` gives higher-fidelity rendering but is optional — if it isn't available (common on Windows without a system Cairo library), the notebook falls back to a supersampled Pillow renderer automatically.
- On first run it downloads the public MathWriting excerpt dataset into `mathwriting-2024-excerpt/`. Re-running skips the download if that directory already exists. To use the full dataset instead of the excerpt, edit `DATASET_URL`/`ROOT_DIR` in the download cell.

**Outputs** (all git-ignored, regenerate by re-running the notebook):
- `processed/images/{split}/{sample_id}.png` — clean (unaugmented) grayscale renders, one per split (`train`/`valid`/`test`/`synthetic`/`symbols`).
- `processed/labels/{split}.jsonl` — one record per sample: tokens, token ids, image width/height, etc.
- `processed/vocab.json` — token vocabulary, built from `train` + `synthetic` only.
- `processed/metadata.json` — the config a run used (rendering settings, augmentation policy), for reproducibility.
- `augmented_demo/` — a few `train` samples rendered with the augmentation utility, for visual QA only (not training data).

**Augmentation policy:** when `run_train.py --raw-dir /path/to/mathwriting` is
provided, augmentation is applied online at training time (e.g. from a `Dataset.__getitem__`) via `render_with_augmentation`, defined in the notebook — never baked into the PNGs in `processed/`. Each of the four transforms (rotation, shear, thinning, blur) rolls independently with its own probability and parameter range, so a training `Dataset` should call this fresh per `train` sample per epoch rather than reading from a fixed augmented copy. See the "Augmentation: policy" section in the notebook for the exact config and rationale.

`mathwriting_code_examples.ipynb` is the unmodified official MathWriting example notebook, kept for reference only.

---

# Input preprocessing (photo → model input)


`inputpreprocessing.py` is a preprocessing pipeline for input images in our handwritten-math-to-LaTeX model.
Takes a scanned or camera-photographed image of a handwritten equation and produces a normalized tensor ready for a MobileNet encoder.

### Pipeline

```
input image (scan / photo)

-> find the location of equations in the image

-> identifying the edges of writing surface(paper, whiteboard, post-it note, etc.)
   (surface quadrilateral when available; otherwise vanishing-point partial
   rectification when only one direction of page edges is reliable)

-> crop to the equation's ink extent by removing unncessary surrounding blank space

-> binarization
   (Otsu or adaptive threshold, polarity-normalized)

-> resize to `96 x W`
   (height is always 96; width is proportional and remains variable)

-> MobileNet input tensor
   (normalized, CHW, batched)
```

| Step                   | Problem it solves                                                                                                                                                                                       |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Perspective correction | Corrects photographed paper when reliable page/surface geometry exists. Clean MathWriting-style white canvases are detected and deliberately left unwarped, since their pen strokes are not page edges. |
| Ink crop               | Removes photo/page whitespace while retaining disconnected symbols in one expression.                                                                                                                   |
| Binarization           | Removes paper texture, lighting gradients, and camera noise/color, leaving just the ink/marker strokes.                                                                                                 |
| Height-only resize     | A fixed square would stretch wide equations and distort symbols. The pipeline uses `96 x W`; use `pad_mobilenet_batch` only when batching samples with different widths.                                |
| MobileNet formatting   | Converts the image array into the float tensor shape a MobileNet encoder expects.                                                                                                                       |

### References

#### Perspective correction

- **Source of the _approach_:** the [Im2Latex project page](https://sujayr91.github.io/Im2Latex/)
  describes correcting perspective distortion: Canny edge detection → Hough transform to find
  the clipboard/page boundary lines → intersecting those lines for the 4
  corners → homography to warp the corners into a rectangle → binarize.

- **Canny + `cv2.HoughLinesP` usage:** follows the
  [OpenCV Hough Line Transform tutorial](https://docs.opencv.org/4.x/d9/db0/tutorial_hough_lines.html).

- **Four-point homography (`cv2.getPerspectiveTransform` + `cv2.warpPerspective`):**
  standard OpenCV document-rectification pattern, e.g.
  [learnopencv's perspective-correction.py](https://github.com/spmallick/learnopencv/blob/master/Homography/perspective-correction.py)

#### Binarization

- **Otsu (`cv2.THRESH_OTSU`) and adaptive thresholding
  (`cv2.ADAPTIVE_THRESH_GAUSSIAN_C`):** standard OpenCV binarization methods,
  in general use across OCR preprocessing.

---

# Encoder


The visual encoder component of the handwritten-math-to-LaTeX pipeline. It extracts visual features from preprocessed handwritten math images and converts them into sequence tokens for cross-attention in the Transformer decoder.

---

### Repository Files

* `mobilenet_encoder.py`: Core encoder class (`MobileNetEncoder`) that outputs visual tokens for the Transformer decoder.
* `mobilenet_stride_check.py`: Diagnostic helper script used to verify layer output shapes and confirm the Stride-16 cutoff index.

---

### Pipeline Context

The encoder feeds visual tokens to the decoder through 2D positional encoding.
The CAN head reads the same encoder features as a separate training branch.

### Running the Stride Verification Script

Before building the encoder, run `mobilenet_stride_check.py` to inspect the spatial downsampling across MobileNetV3 blocks:

```bash
python3 mobilenet_stride_check.py
```

What it does:

* Loops through `MobileNetV3-Large.features` block-by-block using a sample input shaped 96px tall, 256px wide.
* Prints the image height after each block, so you can see exactly where it shrinks.
* Confirms that block 12 is the last block where the height is 6px (stride-16) — block 13 shrinks it further to 3px (stride-32).

### Architectural Decisions

* **Pretrained backbone:** Loads `MobileNetV3-Large` with ImageNet weights instead of training from scratch. General visual features (edges, strokes, loops) transfer well to handwritten math symbols, saving a lot of training time.

* **Stride-16 cutoff:** Cuts the backbone off at block index 12 (`features[:13]`). This keeps more spatial detail than going further to stride-32, which would compress the image too much and risk losing fine stroke details.

* **Dynamic projection:** Converts the backbone's 112 output channels to `d_model` using a 1x1 convolution. This channel count is calculated automatically from a test pass instead of hardcoded, so if the cutoff index ever changes, this part doesn't need manual updates.

* **Sequence formatting:** Flattens the 2D grid of features (height × width) into a 1D sequence, which is the format `nn.TransformerDecoder` expects as input.

### Confirmed Specifications & Verification

* **Input size:** Fixed height of 96px, variable width (padded per batch).
* **Layer output shape:** With an input of shape (1, 3, 96, 256), the backbone output is (1, 112, 6, 16): stride-16 in both spatial dimensions.
* **End-to-end encoder test:** A dummy batch shaped (2, 3, 96, 256) produces (2, 96, d_model): 6 × 16 spatial tokens per image.

### Integration status

The current model uses `d_model=256`, stride 16, and a six-row feature grid
for 96px inputs. Encoder/decoder wiring, grayscale-to-RGB expansion, padding
masks, and the auxiliary CAN branch are implemented in `hmer_model.py`.
The positional-encoding buffer reserves at least eight rows; that capacity
does not change the actual feature height of six.

### References

* [PyTorch MobileNetV3 Documentation](https://pytorch.org/vision/main/models/mobilenetv3.html)
* [MobileNetV2 Autoencoder for Feature Extraction](https://medium.com/@abbesnessim/mobilenetv2-autoencoder-an-efficient-approach-for-feature-extraction-and-image-reconstruction-9c70ba58947a)

---

# Decoder

The encoder/decoder pipeline has been trained on real data; see the reported
results above. The implementation is split across:

- `baseline_decoder.py`: plain Transformer components, positional encodings,
  and greedy/beam decoding.
- `latex_decoder.py`: PosFormer-inspired attention refinement and position-forest
  components, controlled by `use_arm` and `use_position_forest`.
- `hmer_model.py`: encoder/decoder integration, grayscale-to-RGB expansion,
  cross-attention padding masks, CAN counting, and the training step.
- `train.py`: epoch loops, validation metrics, checkpoint selection, and early stopping.
- `dataset.py`: the real dataset and collate function; dummy data in
  `train.py` is only for smoke tests.

The encoder returns flattened visual features of shape `[B, 6 * W_feat, 256]`.
`HMERModel` constructs the padding mask from each sample's true pixel width
and passes the actual feature height to the decoder. Batch collation pads
images with white pixels to a width divisible by 16.

Each Transformer layer uses masked self-attention, cross-attention to the
image features, and a feed-forward network. The default model has three layers.

### PosFormer components and the active objective

The implementation draws on [PosFormer](https://arxiv.org/abs/2407.07764)
and its [reference code](https://github.com/SJTU-DeepVisionLab/PosFormer).
Attention refinement and position-forest components are available in the
decoder. However, `hmer_train_step()` currently optimizes only:

```text
total_loss = sequence_loss + counting_weight * counting_loss
```

The position-forest auxiliary loss is not included in this training step.
Enabling its module does not mean that auxiliary objective was trained.
Setting both decoder toggles to false selects the plain decoder path;
setting `--counting-weight 0` only changes the CAN loss contribution.

### Evaluation and remaining work

- Complete a comparable 96px sequence-only run with matched initialization,
  training budget, dataset, and decoding settings.
- Evaluate checkpoints on the held-out test split separately from validation.
- Record augmentation settings, maximum decoding length, and beam width with
  each reported result.
- Add the custom-image inference scripts and their usage instructions once
  they are available on this branch.
- Investigate errors on external handwriting, including uppercase/lowercase
  ambiguity and complex expression structure. Validation ExpRate does not
  guarantee the same accuracy on photographed or iPad-written formulas.

