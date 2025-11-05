# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# Processed Dataset Pipeline Guide

## Purpose

- Document how `data/processed/*.parquet` files are generated and validated after the November 2025 refactor.
- Clarify the difference between temporary synthetic fixtures (used for tests) and production datasets assembled from collection runs.
- Provide reproducible commands for rebuilding processed datasets when onboarding a new environment or after cleaning the `data/` directory.

## Key Artifacts

| File                                                | Description                                                                                                                        |
| --------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `data/processed/all_runs_combined.parquet`          | Baseline combined dataset produced from `data/runs/` via `combine_runs.py`. Required by STMGT smoke tests and exploratory scripts. |
| `data/processed/all_runs_augmented.parquet`         | Output of `augment_data_advanced.py` (balanced augmentation)                                                                       |
| `data/processed/all_runs_extreme_augmented.parquet` | Output of `augment_extreme.py` (max-coverage augmentation)                                                                         |

Synthetic fixtures generated during hardening are clearly marked in the changelog. Replace them with production data as soon as the collection pipeline runs on the target machine.

## Regenerating Combined Runs

```bash
# 1. Ensure raw collections are present (data/runs/run_*/)
# 2. Combine individual runs into the baseline parquet
conda run -n dsp python scripts/data/combine_runs.py \
    --runs-dir data/runs \
    --output-file data/processed/all_runs_combined.parquet \
    --validate
```

Arguments `--runs-dir` and `--output-file` are optional; defaults match the project layout. Passing `--validate` triggers `traffic_forecast/data/dataset_validation.py`, printing record counts, missing columns, and duplicate statistics before saving the parquet file.

## Validating Processed Datasets

Use the shared validator to confirm schema and data health before launching training jobs or dashboards:

```bash
conda run -n dsp python scripts/data/validate_processed_dataset.py \
    --dataset data/processed/all_runs_combined.parquet
```

- Verifies required columns (`run_id`, `timestamp`, node identifiers, speed, distance, duration).
- Detects missing values and duplicate edge samples within each run.
- Returns exit code `0` on success, `1` on failure—making it CI-friendly.

To enforce extra columns (for augmented datasets):

```bash
conda run -n dsp python scripts/data/validate_processed_dataset.py \
    --dataset data/processed/all_runs_extreme_augmented.parquet \
    --require precipitation_mm wind_speed_kmh temperature_c
```

## Linking With Training & Dashboard

1. **Training configs** reference parquet filenames through `training.data_source`. Place the file under `data/processed/` and run the validator before training.
2. **Dashboard** dataset selectors (e.g., `2_Data_Overview.py`) inspect the file system. After regenerating datasets, rerun the validator and refresh the dashboard page.
3. **Automation**: integrate the validator into collection/augmentation workflows to catch schema drift early.

## Next Steps

- Integrate validation into CI once real datasets are restored to the repo environment.
- Extend the validator to support row-level heuristic checks (e.g., speed bounds, timestamp monotonicity) when production data is available again.
