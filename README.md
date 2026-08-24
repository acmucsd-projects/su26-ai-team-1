# su26-ai-team-1

https://drive.google.com/file/d/13TUkaJ0AaIobDGX0DzM3Dtjv0d9ndU4P/view?usp=drive_link

Link to processed dataset as a .zip on Drive

## Data pipeline

`mathwriting_preprocessing.ipynb` creates the baseline dataset. `augment_processed.py` then bakes rotation, shear, and Gaussian blur into the processed PNGs. Stroke thinning is intentionally excluded because it can erase fine symbols.

Run the notebook top to bottom. It's self-contained:
- Requires `pillow`, `numpy`, `matplotlib`; the notebook installs any that are missing. `pycairo` gives higher-fidelity rendering but is optional — if it isn't available (common on Windows without a system Cairo library), the notebook falls back to a supersampled Pillow renderer automatically.
- On first run it downloads the public MathWriting excerpt dataset into `mathwriting-2024-excerpt/`. Re-running skips the download if that directory already exists. To use the full dataset instead of the excerpt, edit `DATASET_URL`/`ROOT_DIR` in the download cell.

**Outputs** (all git-ignored, regenerate by re-running the notebook):
- `processed/images/{split}/{sample_id}.png` — deterministically augmented grayscale renders after running `augment_processed.py`.
- `processed/labels/{split}.jsonl` — one record per sample: tokens, token ids, image width/height, etc.
- `processed/vocab.json` — token vocabulary, built from `train` + `synthetic` only.
- `processed/metadata.json` — the config a run used (rendering settings, augmentation policy), for reproducibility.
- `augmented_demo/` — a few `train` samples rendered with the augmentation utility, for visual QA only (not training data).

**Augmentation policy:** each transform rolls independently using the config in `mathwriting_pipeline.py`. Seeds are derived from the global seed, split, and sample ID, so output is reproducible regardless of worker count. Run a non-destructive preflight with `python augment_processed.py --limit 2`, inspect `augmentation_qa/`, then run `python augment_processed.py`. The baseline `processed/` directory remains unchanged; the complete augmented dataset is written to `processed-augmented/`. Every split is staged completely before its images and updated label widths are committed. Samples with a matching local InkML are re-rendered from strokes; any missing source uses the documented image-space fallback and the metadata records the counts of each mode. Point `MathWritingDataset(processed_dir=...)` at the desired dataset.

Validate every output PNG is readable and exactly 64px high with `python validate_processed.py processed-augmented`.

`mathwriting_code_examples.ipynb` is the unmodified official MathWriting example notebook, kept for reference only.
