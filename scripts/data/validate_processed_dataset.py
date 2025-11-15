"""Validate processed traffic datasets used across training and dashboards.

Example usage
-------------

Validate the default combined parquet file:

```
python scripts/data/validate_processed_dataset.py
```

Validate a specific file with custom required columns:

```
python scripts/data/validate_processed_dataset.py \
    --dataset data/processed/all_runs_gapfilled_week.parquet \
    --require run_id timestamp node_a_id node_b_id speed_kmh mae
```
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from traffic_forecast.data.dataset_validation import (
    DEFAULT_REQUIRED_COLUMNS,
    DatasetValidationResult,
    validate_processed_dataset,
)


def _format_list(values: List[str]) -> str:
    return ", ".join(values) if values else "-"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate processed parquet datasets before training or analysis.",
    )
    parser.add_argument(
    "--dataset",
    type=Path,
    default=Path("data/processed/all_runs_gapfilled_week.parquet"),
    help="Path to the parquet dataset to validate (default: data/processed/all_runs_gapfilled_week.parquet).",
    )
    parser.add_argument(
        "--require",
        nargs="*",
        default=None,
        help="Additional column names that must be present (space separated).",
    )
    return parser


def _print_report(result: DatasetValidationResult) -> None:
    print("== Dataset Validation Report ==")
    print(f"Path: {result.path}")
    print(f"Exists: {result.exists}")
    if not result.exists:
        print("Errors:")
        for error in result.errors:
            print(f"  - {error}")
        return

    print(f"Rows: {result.rows}")
    print(f"Columns: {_format_list(result.columns)}")
    print(f"Missing Columns: {_format_list(result.missing_columns)}")
    print(f"Columns With Nulls: {_format_list(result.null_columns)}")
    print(f"Duplicate Rows (run_id/node pair): {result.duplicate_rows}")
    if result.errors:
        print("Errors:")
        for error in result.errors:
            print(f"  - {error}")


def main(argv: List[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    required = list(DEFAULT_REQUIRED_COLUMNS)
    if args.require:
        required.extend(args.require)

    result = validate_processed_dataset(args.dataset, required)
    _print_report(result)
    return 0 if result.is_valid else 1


if __name__ == "__main__":
    sys.exit(main())
