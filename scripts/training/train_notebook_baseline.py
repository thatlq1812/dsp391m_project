"""Run the ASTGCN workflow to reproduce the original analysis."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from traffic_forecast.models.astgcn import (
    NotebookBaselineConfig,
    NotebookBaselineRunner,
)

LOGGER = logging.getLogger("astgcn")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute the notebook-style baseline analysis and store outputs.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/processed/all_runs_combined.parquet"),
        help="Input dataset path. Supports CSV or Parquet.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/astgcn"),
        help="Directory where artifacts will be written.",
    )
    parser.add_argument(
        "--congestion-threshold",
        type=float,
        default=20.0,
        help="Speed threshold (km/h) for congestion detection.",
    )
    return parser


def configure_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)


def main() -> None:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args()

    config = NotebookBaselineConfig(
        data_path=args.data_path,
        output_root=args.output_root,
        congestion_threshold=args.congestion_threshold,
    )

    runner = NotebookBaselineRunner(config)
    LOGGER.info("Starting ASTGCN run")
    outputs = runner.run()
    for name, path in outputs.items():
        LOGGER.info("%s -> %s", name, path)
    LOGGER.info("ASTGCN run complete")


if __name__ == "__main__":
    main()
