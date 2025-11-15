# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# Data Generation Scripts

Scripts for generating and processing traffic datasets.

## Super Dataset Generator

### Overview

`generate_super_dataset.py` creates a challenging 1-year traffic simulation dataset designed to prevent autocorrelation exploitation and test true spatio-temporal learning.

### Quick Start

```bash
# Dry run (test configuration)
python scripts/data/generate_super_dataset.py --dry-run

# Generate full dataset
python scripts/data/generate_super_dataset.py \
    --config configs/super_dataset_config.yaml \
    --output data/processed/super_dataset_1year.parquet \
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
