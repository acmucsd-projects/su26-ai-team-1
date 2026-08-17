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
