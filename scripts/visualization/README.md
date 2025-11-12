# Visualization Scripts for Final Report

**Purpose:** Generate all figures required for the DS Capstone final report

---

## Quick Start

```bash
# Generate all figures at once
cd /d/UNI/DSP391m/project
python scripts/visualization/generate_all_figures.py

# Or generate specific figure sets
python scripts/visualization/01_data_figures.py      # Figures 1-4
python scripts/visualization/02_eda_figures.py       # Figures 5-10
python scripts/visualization/04_results_figures.py   # Figures 13-17
```

---

## Output

All figures are saved to: `docs/final_report/figures/`

Format: **PDF** (300 DPI, publication quality)

---

## Figure List

### Data & Preprocessing (01_data_figures.py)

- **Fig 1:** Speed distribution histogram with KDE
- **Fig 2:** Road network topology graph
- **Fig 4:** Normalization before/after comparison

### Exploratory Data Analysis (02_eda_figures.py)

- **Fig 5:** Speed histogram with Gaussian mixture overlay
- **Fig 6:** Average speed by hour (with rush hour highlights)
- **Fig 7:** Speed by day of week (box plots)
- **Fig 8:** Spatial correlation heatmap (62x62)
- **Fig 9:** Temperature vs speed scatter plot
- **Fig 10:** Speed by weather condition (box plots)

### Results & Evaluation (04_results_figures.py)

- **Fig 13:** Training and validation curves
- **Fig 15:** Model comparison bar chart (MAE, R²)
- **Fig 16:** Good prediction example
- **Fig 17:** Bad prediction example (challenging case)

### Manual Creation Required

- **Fig 11:** STMGT architecture diagram (use draw.io or PowerPoint)
- **Fig 12:** Attention heatmap visualization
- **Fig 14:** Hyperparameter tuning results

---

## Dependencies

```bash
# Install required packages
pip install matplotlib seaborn pandas numpy scipy scikit-learn networkx
```

---

## Troubleshooting

**Issue:** `FileNotFoundError` for data files

**Solution:** Ensure you have:

- `data/processed/all_runs_extreme_augmented.parquet`
- `cache/overpass_topology.json`
- `cache/adjacency_matrix.npy`
- `outputs/stmgt_v2_20251112_091929/`

**Issue:** Import errors

**Solution:** Run from project root: `python scripts/visualization/generate_all_figures.py`

---

## Customization

Edit `utils.py` to change:

- Figure size: `FIGURE_CONFIG['figsize']`
- DPI: `FIGURE_CONFIG['dpi']`
- Format: `FIGURE_CONFIG['format']` (pdf, png, svg)
- Style: Modify `set_plot_style()` function

---

**Maintainer:** THAT Le Quang (thatlq1812)  
**Last Updated:** November 12, 2025
