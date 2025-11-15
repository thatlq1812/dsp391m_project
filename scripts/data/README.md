# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# Data Generation Scripts

Scripts for generating and processing traffic datasets.

---

## 📊 Overview

This directory contains scripts for:

1. **Real Data Collection & Processing** - Working with actual traffic data
2. **Monthly Baseline Generation** - Extending 4-day data to 1-month realistic dataset
3. **Super Dataset Generation** - Creating 1-year challenging simulation dataset

---

## 🚀 Quick Start

### Generate 1-Month Baseline Dataset

```bash
# Step 1: Generate 2,880 synthetic runs from 4-day collection
python scripts/data/03_generation/generate_baseline_1month.py \
  --runs-dir data/runs \
  --output-dir data/runs \
  --reference-run run_20251102_110036

# Step 2: Combine all runs into parquet
python scripts/data/02_preprocessing/combine_baseline_runs.py \
  --runs-dir data/runs \
  --output-file data/processed/baseline_1month.parquet

# Result: 424K records, 1.59 MB, 30 days coverage
```

### Generate 1-Year Augmented Dataset

```bash
# Full 1-year challenging augmented dataset
python scripts/data/03_generation/generate_augmented_1year.py \
  --config configs/super_dataset_config.yaml \
  --output data/processed/augmented_1year.parquet \
  --visualize
```

---

## 📁 Scripts Overview

### Real Data Collection

**`preprocess_runs.py`** - Convert JSON runs to Parquet

- Loads JSON runs from `data/runs/`
- Adds derived features
- Caches to Parquet for fast loading

**`combine_runs.py`** - Combine multiple runs

- Merges all runs into single dataset
- Validates data quality
- Legacy script (replaced by `combine_monthly_runs.py`)

### Monthly Baseline Generation

**`03_generation/generate_baseline_1month.py`**

- **Purpose:** Extend 4-day collection to 1-month realistic dataset
- **Input:** Existing runs in `data/runs/`
- **Output:** 2,880 synthetic runs (15-min intervals)
- **Features:**
  - Pattern analysis from existing data
  - Realistic speed generation (rush hours, weekends)
  - Weather simulation (temp, humidity, rain)
  - Random incidents (5% chance)
  - Randomized seconds for realism

**Example:**

```bash
python scripts/data/03_generation/generate_baseline_1month.py \
  --random-seed 42
```

**Output:**

- 2,880 run directories: `data/runs/run_YYYYMMDD_HHMMSS/`
- Manifest: `data/runs/generated_runs_manifest.json`
- Date range: 2025-10-03 to 2025-11-02

**`02_preprocessing/combine_baseline_runs.py`**

- **Purpose:** Combine all monthly runs into single parquet
- **Input:** Runs from `data/runs/`
- **Output:** `data/processed/all_runs_monthly.parquet` (1.59 MB)
- **Features:**
  - Encoding-safe JSON loading
  - Progress tracking with tqdm
  - Handles dict/list weather formats
  - Comprehensive statistics

**Example:**

```bash
python scripts/data/02_preprocessing/combine_baseline_runs.py
```

**Output Statistics:**

- Total: 424,224 records
- Edges: 144 unique
- Speed: 19.05 ± 7.83 km/h
- Range: [3.00, 52.84] km/h

## Augmented 1-Year Dataset Generator

### Overview

`03_generation/generate_augmented_1year.py` creates a challenging 1-year traffic simulation dataset designed to prevent autocorrelation exploitation and test true spatio-temporal learning.

### Quick Start

```bash
# Dry run (test configuration)
python scripts/data/03_generation/generate_augmented_1year.py --dry-run

# Generate full dataset
python scripts/data/03_generation/generate_augmented_1year.py \
  --config configs/super_dataset_config.yaml \
  --output data/processed/augmented_1year.parquet \
  --visualize
```

### Features

**Base Patterns:**

- Realistic weekday rush hour dynamics
- Weekend leisure patterns
- Night-time speed variations
- Edge-specific characteristics

**Disruption Events:**

- Random traffic incidents (accidents, breakdowns)
- Long-term construction zones
- Weather events (rain, fog)
- Special events (concerts, sports, festivals)

**Seasonal Patterns:**

- School calendar impact
- Economic cycles (month/quarter-end)
- Vietnamese public holidays
- Long weekend patterns

**Challenging Scenarios:**

- Cascading failures (incident → congestion wave)
- Multi-event interactions (rain + holiday + concert)
- Construction zone adaptation
- Temporal pattern shifts

### Configuration

Edit `configs/super_dataset_config.yaml` to customize:

```yaml
temporal:
  start_date: "2024-01-01"
  duration_days: 365
  interval_minutes: 10

incidents:
  rate_per_week: 3
  types: [minor_accident, major_accident, vehicle_breakdown]

construction:
  num_zones_per_year: 8
  duration_range: [14, 56] # days

weather:
  rain:
    probability_by_month: { ... }
  fog:
    probability: 0.05
```

### Output Structure

```
data/processed/
├── super_dataset_1year.parquet          # Main dataset (~500MB)
├── super_dataset_metadata.json          # Events, holidays, construction
├── super_dataset_statistics.json        # Validation metrics
└── super_dataset_splits.json            # Train/val/test boundaries
```

### Dataset Schema

```python
Columns:
- timestamp: datetime64[ns]              # 10-min intervals
- node_a_id: str                         # Source node
- node_b_id: str                         # Target node
- speed_kmh: float                       # Speed (3-52 km/h)
- is_incident: bool                      # Incident flag
- is_construction: bool                  # Construction flag
- weather_condition: str                 # clear/rain/fog
- is_holiday: bool                       # Holiday flag
- event_type: str                        # concert/sports/festival/null
- temperature_c: float                   # Temperature
- precipitation_mm: float                # Rainfall
- wind_speed_kmh: float                  # Wind speed
- humidity_percent: float                # Humidity

Size: ~7.5M rows (52,560 timestamps × 144 edges)
```

### Train/Val/Test Split

```python
Train:  Months 1-8   (35 weeks, 67%)
Gap:    Week 36-37   (2 weeks, prevents leakage)
Val:    Months 9-10  (9 weeks, 17%)
Test:   Months 11-12 (8 weeks, 16%)
```

### Implementation Status

**✅ Complete:**

- Configuration file template
- Generator class skeleton
- Documentation

**🚧 TODO:**

- `generate_base_pattern()` - Base traffic patterns
- `apply_seasonal_overlay()` - School calendar, economic cycles
- `inject_incidents()` - Random disruptions with propagation
- `add_construction_zones()` - Long-term disruptions
- `apply_weather_effects()` - Rain, fog simulation
- `inject_special_events()` - Concerts, sports, festivals
- `apply_holidays()` - Vietnamese public holidays
- `propagate_spatial_impact()` - Graph-based diffusion
- `validate_dataset()` - Quality checks
- `visualize()` - Sample plots

### Expected Results

**GraphWaveNet (after fix):**

- Current (1 week): MAE 0.25 km/h (autocorrelation)
- Expected (1 year): MAE 4-6 km/h (forced learning)

**LSTM Baseline:**

- Current: MAE 4.42 km/h
- Expected: MAE 5-7 km/h

**STMGT:**

- Current: MAE 1.88 km/h
- Expected: MAE 3-4 km/h (maintains advantage)

### Timeline

- **Week 1:** Base patterns + Seasonal overlay
- **Week 2:** Event injection + Spatial propagation
- **Week 3:** Integration + Quality control
- **Week 4:** Validation + Documentation

**Target Completion:** December 13, 2025

### References

- Design document: `docs/SUPER_DATASET_DESIGN.md`
- Configuration: `configs/super_dataset_config.yaml`
- Main script: `scripts/data/generate_super_dataset.py`

---

## Other Scripts

(To be added as more data processing scripts are created)
