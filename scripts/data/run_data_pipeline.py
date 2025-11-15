#!/usr/bin/env python3
"""
Data Pipeline Orchestrator

Convenience CLI to:
- Generate baseline 1-month runs and combine to parquet
- Generate augmented 1-year dataset

Examples:
  # Baseline 1-month (runs + parquet)
  python scripts/data/run_data_pipeline.py baseline \
    --runs-dir data/runs \
    --reference-run run_20251102_110036 \
    --output-parquet data/processed/baseline_1month.parquet

  # Augmented 1-year
  python scripts/data/run_data_pipeline.py augmented \
    --config configs/data/augmented_1year.yaml \
    --output-parquet data/processed/augmented_1year.parquet
"""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _abs(p: Path) -> Path:
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def run_baseline(runs_dir: Path, reference_run: str, output_parquet: Path, seed: int) -> None:
    import subprocess, sys
    gen_script = PROJECT_ROOT / 'scripts/data/03_generation/generate_baseline_1month.py'
    comb_script = PROJECT_ROOT / 'scripts/data/02_preprocessing/combine_baseline_runs.py'

    # Generate runs
    subprocess.check_call([
        sys.executable,
        str(gen_script),
        '--runs-dir', str(_abs(runs_dir)),
        '--output-dir', str(_abs(runs_dir)),
        '--reference-run', reference_run,
        '--random-seed', str(seed),
    ])

    # Combine to parquet
    subprocess.check_call([
        sys.executable,
        str(comb_script),
        '--runs-dir', str(_abs(runs_dir)),
        '--output-file', str(_abs(output_parquet)),
    ])


def run_augmented(config_path: Path, output_parquet: Path, visualize: bool) -> None:
    import subprocess, sys
    script = PROJECT_ROOT / 'scripts/data/03_generation/generate_augmented_1year.py'
    cmd = [
        sys.executable,
        str(script),
        '--config', str(_abs(config_path)),
        '--output', str(_abs(output_parquet)),
    ]
    if visualize:
        cmd.append('--visualize')
    subprocess.check_call(cmd)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Data Pipeline Orchestrator')
    sub = p.add_subparsers(dest='cmd', required=True)

    # Baseline
    b = sub.add_parser('baseline', help='Generate 1-month baseline and parquet')
    b.add_argument('--runs-dir', type=Path, default=Path('data/runs'))
    b.add_argument('--reference-run', type=str, default='run_20251102_110036')
    b.add_argument('--output-parquet', type=Path, default=Path('data/processed/baseline_1month.parquet'))
    b.add_argument('--seed', type=int, default=42)

    # Augmented
    a = sub.add_parser('augmented', help='Generate 1-year augmented dataset')
    a.add_argument('--config', type=Path, default=Path('configs/data/augmented_1year.yaml'))
    a.add_argument('--output-parquet', type=Path, default=Path('data/processed/augmented_1year.parquet'))
    a.add_argument('--visualize', action='store_true')

    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.cmd == 'baseline':
        run_baseline(args.runs_dir, args.reference_run, args.output_parquet, args.seed)
    elif args.cmd == 'augmented':
        run_augmented(args.config, args.output_parquet, args.visualize)


if __name__ == '__main__':
    main()
