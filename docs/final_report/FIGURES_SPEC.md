# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# Figures Specification

This document tracks all visualizations needed for the final report. Each figure includes placement, description, data source, and generation script.

---

## Section 5: Data Description

### Figure 1: Traffic Speed Distribution

- **ID:** `fig_01_speed_distribution`
- **Type:** Histogram + KDE
- **Placement:** Section 5.3.1
- **Description:** Multi-modal distribution showing free-flow, moderate, and congested traffic regimes
- **Data Source:** `data/processed/all_runs_extreme_augmented.parquet`
- **X-axis:** Speed (km/h), range 0-60
- **Y-axis:** Frequency/Density
- **Annotations:** Mean, median, modes marked
- **Script:** `scripts/analysis/generate_speed_distribution.py`
- **Status:** ⏳ TODO

### Figure 2: Road Network Topology

- **ID:** `fig_02_network_topology`
- **Type:** Network graph with geo-coordinates
- **Placement:** Section 5.3.4
- **Description:** 62 nodes (intersections) and 144 edges (road segments) on HCMC map
- **Data Source:** `cache/overpass_topology.json`, `cache/adjacency_matrix.npy`
- **Visualization:** NetworkX graph with lat/lon positions
- **Color coding:** Node degree or centrality
- **Script:** `tools/visualize_nodes.py` (adapt for report)
- **Status:** ⏳ TODO (existing script needs update)

### Figure 3: Spatial Distribution of Nodes

- **ID:** `fig_03_spatial_coverage`
- **Type:** Map overlay (Folium or matplotlib)
- **Placement:** Section 5.5.2
- **Description:** Geographic coverage across HCMC districts
- **Data Source:** `cache/overpass_topology.json`
- **Annotations:** District boundaries, major roads
- **Script:** `scripts/analysis/generate_spatial_map.py`
- **Status:** ⏳ TODO

---

## Section 6: Data Preprocessing

### Figure 4: Normalization Effects

- **ID:** `fig_04_normalization`
- **Type:** Before/After histograms (2 subplots)
- **Placement:** Section 6.1.2
- **Description:** Raw vs normalized speed distributions
- **Data Source:** Raw parquet + normalized data
- **Script:** `scripts/analysis/visualize_normalization.py`
- **Status:** ⏳ TODO

---

## Section 7: Exploratory Data Analysis

### Figure 5: Speed Distribution Histogram

- **ID:** `fig_05_eda_speed_hist`
- **Type:** Histogram with Gaussian mixture overlay
- **Placement:** Section 7.1
- **Description:** Multi-modal distribution with fitted K=3 Gaussians
- **Data Source:** `data/processed/all_runs_extreme_augmented.parquet`
- **Overlay:** 3 Gaussian components fitted
- **Script:** `scripts/analysis/fit_gaussian_mixture.py`
- **Status:** ⏳ TODO

### Figure 6: Average Speed by Hour

- **ID:** `fig_06_hourly_pattern`
- **Type:** Line plot with confidence interval
- **Placement:** Section 7.2.1
- **Description:** Average speed across 24 hours, showing morning/evening rush
- **Data Source:** Aggregated from parquet (group by hour)
- **Y-axis:** Speed (km/h)
- **X-axis:** Hour (0-23)
- **Confidence band:** ±1 std or 95% CI
- **Script:** `scripts/analysis/hourly_speed_pattern.py`
- **Status:** ⏳ TODO

### Figure 7: Speed by Day of Week

- **ID:** `fig_07_weekly_pattern`
- **Type:** Box plot
- **Placement:** Section 7.2.2
- **Description:** Speed distribution for each day (Mon-Sun)
- **Data Source:** Aggregated from parquet (group by dow)
- **Script:** `scripts/analysis/weekly_speed_pattern.py`
- **Status:** ⏳ TODO

### Figure 8: Spatial Correlation Heatmap

- **ID:** `fig_08_spatial_corr`
- **Type:** Heatmap (62×62)
- **Placement:** Section 7.3
- **Description:** Pairwise correlation between node speeds
- **Data Source:** Compute correlation matrix from parquet
- **Color map:** Viridis or RdBu_r
- **Annotations:** Highlight high-correlation clusters
- **Script:** `scripts/analysis/spatial_correlation.py`
- **Status:** ⏳ TODO

### Figure 9: Temperature vs Speed Scatter

- **ID:** `fig_09_temp_speed`
- **Type:** Scatter plot with regression line
- **Placement:** Section 7.4.1
- **Description:** Relationship between temperature and traffic speed
- **Data Source:** Parquet (temperature_c vs speed_kmh)
- **Script:** `scripts/analysis/weather_impact.py`
- **Status:** ⏳ TODO

### Figure 10: Speed by Weather Condition

- **ID:** `fig_10_weather_box`
- **Type:** Box plot (3 categories: clear, light rain, heavy rain)
- **Placement:** Section 7.4.2
- **Description:** Speed distribution under different weather conditions
- **Data Source:** Parquet (group by precipitation bins)
- **Categories:**
  - Clear: precip=0
  - Light rain: 0 < precip <= 5mm
  - Heavy rain: precip > 5mm
- **Script:** `scripts/analysis/weather_impact.py`
- **Status:** ⏳ TODO

---

## Section 9: Model Development

### Figure 11: STMGT Architecture Diagram

- **ID:** `fig_11_architecture`
- **Type:** Block diagram (draw.io or graphviz)
- **Placement:** Section 9.1
- **Description:** Complete architecture showing:
  - Input embedding
  - Parallel ST blocks (GAT || Transformer)
  - Gated fusion
  - Weather cross-attention
  - Mixture output head
- **Tool:** Draw.io or Python diagrams library
- **Script:** Manual creation or `scripts/visualize_architecture.py`
- **Status:** ⏳ TODO

### Figure 12: Attention Visualization

- **ID:** `fig_12_attention_weights`
- **Type:** Heatmap or graph with edge weights
- **Placement:** Section 9.2.2
- **Description:** Learned GATv2 attention weights for sample timestep
- **Data Source:** Extract attention weights during inference
- **Script:** `scripts/analysis/visualize_attention.py`
- **Status:** ⏳ TODO (optional, nice-to-have)

---

## Section 10: Evaluation & Tuning

### Figure 13: Training Curves

- **ID:** `fig_13_training_curves`
- **Type:** Multi-line plot (2 subplots: loss, MAE)
- **Placement:** Section 10.1
- **Description:** Training/validation loss and MAE over epochs
- **Data Source:** `outputs/stmgt_v2_20251109_195802/training_history.csv`
- **Lines:** Train loss, val loss, train MAE, val MAE
- **Annotations:** Best epoch marked (epoch 9)
- **Script:** `scripts/analysis/plot_training_curves.py`
- **Status:** ⏳ TODO

### Figure 14: Hyperparameter Tuning Results

- **ID:** `fig_14_hyperparam_tuning`
- **Type:** Bar chart or table visual
- **Placement:** Section 10.2
- **Description:** Comparison of different hidden_dim (64 vs 96) and K mixtures (3 vs 5)
- **Data Source:** Manual compilation from multiple training runs
- **Metrics:** MAE, RMSE, R²
- **Script:** `scripts/analysis/compare_hyperparams.py`
- **Status:** ⏳ TODO

---

## Section 11: Results & Visualization

### Figure 15: Baseline Comparison Table

- **ID:** `fig_15_baseline_comparison`
- **Type:** Table (styled)
- **Placement:** Section 11.1
- **Description:** Performance metrics for all 5 models
- **Columns:** Model, MAE, RMSE, R², MAPE, CRPS, Params
- **Rows:** LSTM, GCN, GraphWaveNet, ASTGCN, STMGT
- **Data Source:** Compile from `outputs/*/metrics.json` or training logs
- **Script:** `scripts/analysis/generate_comparison_table.py`
- **Status:** ⏳ TODO

### Figure 16: Prediction Example (Good Case)

- **ID:** `fig_16_prediction_good`
- **Type:** Time series plot with uncertainty band
- **Placement:** Section 11.2
- **Description:** Sample node showing accurate 3-hour forecast
- **Lines:** Ground truth, prediction (mean), 80% CI
- **X-axis:** Time (15-min intervals)
- **Y-axis:** Speed (km/h)
- **Data Source:** Test set predictions from `traffic_api/predictor.py`
- **Script:** `scripts/analysis/plot_prediction_examples.py`
- **Status:** ⏳ TODO

### Figure 17: Prediction Example (Challenging Case)

- **ID:** `fig_17_prediction_challenging`
- **Type:** Time series plot
- **Placement:** Section 11.2
- **Description:** Sample showing prediction during sudden congestion (e.g., heavy rain)
- **Purpose:** Demonstrate uncertainty quantification value
- **Script:** `scripts/analysis/plot_prediction_examples.py`
- **Status:** ⏳ TODO

### Figure 18: Calibration Plot

- **ID:** `fig_18_calibration`
- **Type:** Reliability diagram
- **Placement:** Section 11.3
- **Description:** Expected vs observed coverage for confidence intervals
- **X-axis:** Predicted probability (0-1)
- **Y-axis:** Observed frequency
- **Diagonal:** Perfect calibration line
- **Data Source:** Test set predictions with uncertainty
- **Script:** `scripts/analysis/calibration_plot.py`
- **Status:** ⏳ TODO

### Figure 19: Error Distribution by Hour

- **ID:** `fig_19_error_by_hour`
- **Type:** Box plot or violin plot
- **Placement:** Section 11.4
- **Description:** Prediction error distribution across different hours of day
- **Purpose:** Show when model performs best/worst
- **Data Source:** Test set errors grouped by hour
- **Script:** `scripts/analysis/error_analysis.py`
- **Status:** ⏳ TODO

### Figure 20: Spatial Error Heatmap

- **ID:** `fig_20_spatial_error`
- **Type:** Network graph with node colors = MAE
- **Placement:** Section 11.4
- **Description:** Which intersections have highest prediction errors
- **Color scale:** Low MAE (green) → High MAE (red)
- **Data Source:** Test set MAE per node
- **Script:** `scripts/analysis/spatial_error_map.py`
- **Status:** ⏳ TODO

---

## Section 12: Appendices

### Figure A1: Mixture Component Visualization

- **ID:** `fig_a1_mixture_components`
- **Type:** Stacked distribution plot
- **Placement:** Appendix A
- **Description:** Show 5 Gaussian components for a sample prediction
- **Purpose:** Illustrate how mixture models uncertainty
- **Script:** `scripts/analysis/visualize_mixture.py`
- **Status:** ⏳ TODO (optional)

### Figure A2: Full Network Topology (High Resolution)

- **ID:** `fig_a2_full_network`
- **Type:** Large network graph
- **Placement:** Appendix B
- **Description:** Detailed view of all 62 nodes and 144 edges with labels
- **Script:** `tools/visualize_nodes.py`
- **Status:** ⏳ TODO

---

## Priority Ranking

**High Priority (Must Have):**

1. Figure 13: Training Curves ⭐
2. Figure 15: Baseline Comparison Table ⭐
3. Figure 16: Prediction Example (Good) ⭐
4. Figure 6: Hourly Speed Pattern ⭐
5. Figure 10: Speed by Weather ⭐

**Medium Priority (Should Have):** 6. Figure 1: Speed Distribution 7. Figure 8: Spatial Correlation 8. Figure 11: Architecture Diagram 9. Figure 18: Calibration Plot 10. Figure 19: Error by Hour

**Low Priority (Nice to Have):** 11. Figure 2: Network Topology 12. Figure 12: Attention Visualization 13. Figure A1: Mixture Components 14. All other figures

---

## Generation Scripts TODO

**Create these scripts in `scripts/analysis/`:**

```bash
scripts/analysis/
├── generate_speed_distribution.py
├── hourly_speed_pattern.py
├── weekly_speed_pattern.py
├── spatial_correlation.py
├── weather_impact.py
├── plot_training_curves.py       # Priority 1
├── generate_comparison_table.py  # Priority 1
├── plot_prediction_examples.py   # Priority 1
├── calibration_plot.py
├── error_analysis.py
├── spatial_error_map.py
└── visualize_mixture.py
```

---

## Data Queries Needed

**Before generating figures, need to query:**

```python
# 1. Speed statistics
df = pd.read_parquet('data/processed/all_runs_extreme_augmented.parquet')
print(f"Min speed: {df['speed_kmh'].min():.2f}")
print(f"Max speed: {df['speed_kmh'].max():.2f}")
print(f"Mean speed: {df['speed_kmh'].mean():.2f}")
print(f"Std speed: {df['speed_kmh'].std():.2f}")

# 2. Weather statistics
print(f"Temp range: {df['temperature_c'].min():.1f} - {df['temperature_c'].max():.1f} °C")
print(f"Max precipitation: {df['precipitation_mm'].max():.1f} mm")

# 3. Sample counts
print(f"Total records: {len(df):,}")
print(f"Unique runs: {df['run_id'].nunique()}")

# 4. Missing data
print(df.isnull().sum())
```

---

**Status Summary:**

- Total Figures: 20+ (including appendices)
- Completed: 0
- In Progress: 0
- TODO: 20

**Next Steps:**

1. Generate training curves (Figure 13) - highest priority
2. Create baseline comparison table (Figure 15)
3. Generate prediction examples (Figure 16, 17)
4. Create EDA visualizations (Figures 5-10)
