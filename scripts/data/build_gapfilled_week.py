#!/usr/bin/env python3
"""Create a gap-filled weekly dataset on a strict 15-minute cadence."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from traffic_forecast.data.gap_fill import GapFillAugmentor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fill temporal gaps and extend coverage to a single week",
    )
    parser.add_argument(
        "--input",
        default="data/processed/all_runs_combined.parquet",
        help="Input parquet with real collection runs",
    )
    parser.add_argument(
        "--output",
        default="data/processed/all_runs_gapfilled_week.parquet",
        help="Destination parquet for the gap-filled dataset",
    )
    parser.add_argument(
        "--freq",
        default="15min",
        help="Target sampling frequency (default: 15min)",
    )
    parser.add_argument(
        "--target-days",
        type=int,
        default=7,
        help="Desired temporal coverage in days (default: 7)",
    )
    parser.add_argument(
        "--max-interp-minutes",
        type=int,
        default=90,
        help="Maximum gap length (minutes) to fill via interpolation",
    )
    parser.add_argument(
        "--speed-noise",
        type=float,
        default=0.05,
        help="Relative noise applied to synthesized speed samples",
    )
    parser.add_argument(
        "--weather-noise",
        type=float,
        default=0.02,
        help="Relative noise applied to synthesized weather values",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for deterministic augmentation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {input_path}")

    print(f"Loading dataset: {input_path}")
    df = pd.read_parquet(input_path)

    augmentor = GapFillAugmentor(
        freq=args.freq,
        target_days=args.target_days,
        max_interp_minutes=args.max_interp_minutes,
        speed_noise=args.speed_noise,
        weather_noise=args.weather_noise,
        random_seed=args.random_seed,
    )

    filled_df, summary = augmentor.build(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filled_df.to_parquet(output_path, index=False)

    print("\nGap-filled dataset created")
    print(f"  Output:           {output_path}")
    print(f"  Rows:             {summary.total_rows:,}")
    print(f"  Real rows:        {summary.real_rows:,}")
    print(f"  Augmented rows:   {summary.augmented_rows:,} ({summary.augmented_ratio:.1%})")
    print(f"  Coverage (days):  {summary.coverage_days:.2f}")
    print(f"  Timeline:         {summary.timeline_start} → {summary.timeline_end}")


if __name__ == "__main__":
    main()
