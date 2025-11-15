# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# Data Documentation

Dataset structure, preprocessing, and augmentation documentation.

---

## 📊 Overview

This section contains comprehensive documentation about the datasets used in the STMGT Traffic Forecasting System.

**Key Datasets:**

- **Original Dataset:** 1-week HCMC traffic data (96,768 samples)
- **Super Dataset:** 1-year simulation with realistic disruptions (7.5M samples)

---

## 📖 Available Documentation

### [Data Overview](DATA.md)

Complete dataset specifications and structure.

**Contents:**

- Dataset format and schema
- Temporal and spatial coverage
- Data quality metrics
- Collection methodology

**Quick Stats:**

- Nodes: 62
- Edges: 144
- Interval: 10 minutes
- Speed range: 3-52 km/h

### [Augmentation Guide](AUGMENTATION.md)

Data augmentation strategies and techniques.

**Contents:**

- Temporal augmentation
- Spatial augmentation
- Noise injection methods
- Validation strategies

**Key Techniques:**

- Time shifting
- Speed perturbation
- Edge dropout
- Mixup strategies

### [Super Dataset](super_dataset/)

1-year simulation dataset with realistic disruptions.

**Purpose:**
Prevent autocorrelation exploitation and test genuine spatial-temporal learning.

**Key Features:**

- 365 days, 52,560 timestamps
- 171 incidents (Poisson λ=3/week)
- 8 construction zones
- 79 weather events
- 24 special events
- Vietnamese holiday calendar

**Documentation:**

- **[Design Document](super_dataset/SUPER_DATASET_DESIGN.md)** - Complete methodology
- **[Quick Start](super_dataset/SUPER_DATASET_QUICKSTART.md)** - Usage guide
- **[Generation Report](super_dataset/SUPER_DATASET_GENERATION_COMPLETE.md)** - Results

---

## 🗂️ Dataset Comparison

| Dataset        | Duration | Samples   | Size     | Autocorr (lag-12) | Use Case      |
| -------------- | -------- | --------- | -------- | ----------------- | ------------- |
| Original       | 1 week   | 96,768    | 3.2 MB   | 0.999             | Quick testing |
| Prototype      | 1 month  | 622,080   | 20.9 MB  | 0.558             | Development   |
| Super (1-year) | 365 days | 7,568,640 | 247.9 MB | 0.586             | Production    |

---

## 📁 Dataset Structure

### Parquet Format

All datasets use Apache Parquet format for efficiency:

```python
import pandas as pd

# Load dataset
df = pd.read_parquet('data/processed/super_dataset_1year.parquet')

# Columns
# - timestamp: datetime
# - node_a_id: str
# - node_b_id: str
# - speed_kmh: float [3.0, 52.0]
# - is_incident: bool
# - is_construction: bool
# - weather_condition: str
# - is_holiday: bool
# - temperature_c: float
# - precipitation_mm: float
# - wind_speed_kmh: float
# - humidity_percent: float
```

### Directory Layout

```
data/
├── processed/
│   ├── all_runs_gapfilled_week.parquet    # Original 1-week
│   ├── super_dataset_prototype.parquet     # 1-month test
│   ├── super_dataset_1year.parquet         # Full 1-year
│   ├── super_dataset_metadata.json         # Event metadata
│   ├── super_dataset_statistics.json       # Quality metrics
│   └── super_dataset_splits.json           # Train/val/test splits
└── runs/
    └── [raw collection data]
```

---

## 🔄 Data Pipeline

### 1. Collection

Raw speed data collected from HCMC road network.

### 2. Cleaning

- Gap filling (max 3 consecutive missing)
- Outlier removal (3σ rule)
- Speed range validation [3, 52] km/h

### 3. Augmentation

- Incident injection
- Weather simulation
- Holiday patterns
- Construction zones

### 4. Splitting

- Train: 67%
- Gap: 14 days (prevents leakage)
- Validation: 17%
- Test: 16%

### 5. Normalization

- StandardScaler per edge
- Fitted on training set only
- Applied consistently to val/test

---

## 🎯 Quality Metrics

### Super Dataset 1-Year

**Speed Statistics:**

- Mean: 31.87 km/h ✓
- Std: 11.34 km/h ✓
- Range: [3.00, 52.00] km/h ✓
- Invalid values: 0 ✓

**Temporal Quality:**

- Max jump: 18.00 km/h (threshold: 20) ✓
- Autocorr lag-12: 0.5864 (challenging) ✓

**Event Distribution:**

- Incident rate: 0.459% ✓
- Weather coverage: 12.4% ✓
- Construction: Quarterly rotation ✓

---

## 📚 Related Documentation

- **[Model Overview](../03_models/MODEL.md)** - How models use the data
- **[Training Workflow](../03_models/TRAINING_WORKFLOW.md)** - Data loading in training
- **[Evaluation](../04_evaluation/METRICS_VERIFICATION_ALL_MODELS.md)** - Model performance on datasets

---

**Last Updated:** November 15, 2025
