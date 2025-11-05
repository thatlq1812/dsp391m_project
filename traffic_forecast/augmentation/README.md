# Data Augmentation System

Generate synthetic traffic data for training machine learning models.

## Overview

Creates realistic traffic data by:

1. **Run-based augmentation** - Interpolate between existing runs to achieve 20 samples/hour
2. **Historical generation** - Create synthetic data for date ranges using temporal patterns

## Quick Start

### Method 1: Augment Existing Runs

Interpolate between existing runs to fill gaps:

```bash
# Augment all runs to achieve 20 samples/hour
python scripts/augment_runs.py

# Dry run to preview
python scripts/augment_runs.py --dry-run

# Augment specific runs
python scripts/augment_runs.py --max-runs 10
```

**Result:** Creates intermediate runs with 3-minute intervals between originals.

### Method 2: Generate Historical Data

Create synthetic data for past dates using temporal patterns:

```bash
# Generate 2 months of data (Sept-Oct 2025)
python scripts/generate_historical_data.py \
  --start 2025-09-01 \
  --end 2025-10-31 \
  --variations 20

# Generate without variations (base runs only)
python scripts/generate_historical_data.py \
  --start 2025-09-01 \
  --end 2025-09-30 \
  --variations 0
```

**Result:**

- 1,464 base runs (1/hour × 61 days)
- 20 variations per base run
- Total: 30,744 synthetic runs

## How It Works

### Run-Based Augmentation

```
Original:  [00:00] -------- [01:00] -------- [02:00]
                     ↓
Augmented: [00:00][00:03][00:06]...[01:00][01:03]...
           └─────── 20 samples/hour ───────┘
```

**Features:**

- Cubic spline interpolation
- Noise injection (±2-5 km/h)
- Physics constraints (5-80 km/h)
- Variation control (max 5 km/h or 30% change)

### Historical Generation

```
Template: run_20251030_120000 (144 edges)
          ↓
Generate: Sept 1 - Oct 31, 2025
          • 1 run per hour
          • 20 variations each
          • Temporal patterns applied
```

**Temporal Patterns:**

- **Rush hours:** 7-9am, 5-8pm have lower speeds
- **Weekends:** Lighter traffic
- **Noise:** Realistic ±2-5 km/h variation

## Output Format

All augmented runs are standard run folders:

```
data/runs/
├── run_20250901_000000/      # Base or augmented
│   ├── traffic_edges.json
│   └── metadata.json
└── ...
```

Compatible with existing preprocessing - no changes needed!

## Configuration

### Python API

```python
from traffic_forecast.augmentation import RunBasedAugmentor
from pathlib import Path

# Augment existing runs
augmentor = RunBasedAugmentor(
    source_dir=Path('data/runs'),
    output_dir=Path('data/runs'),
    target_samples_per_hour=20
)
stats = augmentor.augment_all_runs()

# Generate historical data
from scripts.generate_historical_data import HistoricalDataGenerator

generator = HistoricalDataGenerator(
    template_run_dir=Path('data/runs/run_20251030_120000'),
    output_base_dir=Path('data/runs')
)
stats = generator.generate_date_range(
    start_date=datetime(2025, 9, 1),
    end_date=datetime(2025, 10, 31),
    variations_per_run=20
)
```

### CLI Options

**augment_runs.py:**

```
--source DIR           Source directory (default: data/runs)
--output DIR           Output directory (default: same)
--target-samples N     Samples per hour (default: 20)
--max-runs N           Process only first N runs
--dry-run              Simulate without creating files
```

**generate_historical_data.py:**

```
--start YYYY-MM-DD     Start date
--end YYYY-MM-DD       End date
--variations N         Variations per run (default: 20)
--template DIR         Template run directory
```

## Physics Constraints

```python
CONSTRAINTS = {
    'speed_range': (5, 80),           # km/h
    'max_variation': 5.0,             # km/h between points
    'max_variation_ratio': 0.3,       # 30% of range
    'noise_std_range': (2, 5),        # km/h noise
}
```

## Validation

```bash
# View generated runs
python scripts/view_collections.py

# Quick statistics
python scripts/analysis/quick_summary.py
```

## Tips

- Historical generation: ~30s for 30,000 runs
- Run-based augmentation: ~1-2 mins for 700 runs
- Mix synthetic + real data for best ML results

---

**Author:** thatlq1812 - DSP391m Traffic Forecasting
