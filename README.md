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