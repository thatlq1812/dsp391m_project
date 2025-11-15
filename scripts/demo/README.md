# Demo Scripts

## Overview

This directory contains scripts for generating demo visualizations using the **back-prediction strategy**.

## Scripts

### `generate_demo_figures.py`

Main demo script that generates 4 comparison figures:

1. **Multi-Prediction Convergence Chart** - Shows predictions from different times vs actual
2. **Variance Analysis** - Shows prediction variance and convergence by horizon
3. **Accuracy Map** - Interactive map with nodes colored by prediction error
4. **Google Comparison** - Side-by-side comparison with Google Maps baseline

**Usage:**

```bash
# Basic usage
python scripts/demo/generate_demo_figures.py \
    --data data/demo/traffic_data_202511.parquet \
    --model outputs/stmgt_v3_production/best_model.pt \
    --demo-time "2025-11-20 17:00" \
    --output demo_output/

# With custom prediction points
python scripts/demo/generate_demo_figures.py \
    --data data/demo/traffic_data_202511.parquet \
    --model outputs/stmgt_v3_production/best_model.pt \
    --demo-time "2025-11-20 17:00" \
    --prediction-points "14:00,15:00,15:30,16:00" \
    --horizons "1,2,3" \
    --sample-edges 20 \
    --output demo_output/

# Include Google Maps baseline
python scripts/demo/generate_demo_figures.py \
    --data data/demo/traffic_data_202511.parquet \
    --model outputs/stmgt_v3_production/best_model.pt \
    --demo-time "2025-11-20 17:00" \
    --include-google \
    --output demo_output/
```

**Output:**

- `figure1_multi_prediction.png` (300 DPI)
- `figure2_variance_analysis.png` (300 DPI)
- `figure3_traffic_map.html` (interactive)
- `figure4_google_comparison.png` (300 DPI, if --include-google)
- `metrics.json` (raw metrics data)

## Back-Prediction Strategy

Instead of waiting for future data, we make predictions from multiple past time points:

```
Current time: 17:00

Prediction Point 1 (14:00): Use data [11:00-14:00] → Predict 15:00, 16:00, 17:00
Prediction Point 2 (15:00): Use data [12:00-15:00] → Predict 16:00, 17:00, 18:00
Prediction Point 3 (15:30): Use data [12:30-15:30] → Predict 16:30, 17:30, 18:30
Prediction Point 4 (16:00): Use data [13:00-16:00] → Predict 17:00, 18:00, 19:00

Compare all predictions with actual speeds (already collected!)
```

**Advantages:**

- ✅ No waiting for future data
- ✅ More data points (4 predictions vs 1)
- ✅ Shows convergence behavior
- ✅ Reproducible with historical data

## Requirements

```bash
pip install pandas pyarrow matplotlib seaborn folium
```

## Development Notes

**TODO items in code:**

- [ ] Implement actual STMGT model loading in `load_stmgt_model()`
- [ ] Implement actual prediction logic in `make_predictions()`
- [ ] Implement metric calculations in `calculate_metrics()`
- [ ] Add real node/edge data to map in `generate_figure3_map()`
- [ ] Extract Google baseline speeds from historical data

**Data Requirements:**

- Traffic data in Parquet format
- Columns: timestamp, edge_id, node_a_id, node_b_id, speed_kmh, google_predicted_speed
- At least 5 hours of data for demo (3h lookback + 2h prediction)

## Google Maps Baseline

The traffic data already contains Google's traffic predictions from the Directions API:

```python
{
    'duration_in_traffic_sec': 1800,  # Google's prediction
    'google_predicted_speed': 35.2    # Calculated from duration_in_traffic
}
```

No need to call API again - just extract from historical data!

## Example Workflow

1. **Collect data** (VM runs for 3-5 days):

   ```bash
   # On VM
   python /opt/traffic_data/traffic_collector.py
   ```

2. **Download data**:

   ```bash
   # From local machine
   scp vm:/opt/traffic_data/traffic_data_202511.parquet data/demo/
   ```

3. **Generate figures**:

   ```bash
   python scripts/demo/generate_demo_figures.py \
       --data data/demo/traffic_data_202511.parquet \
       --model outputs/stmgt_v3_production/best_model.pt \
       --demo-time "2025-11-20 17:00" \
       --include-google \
       --output demo_output/
   ```

4. **Create presentation**:
   - Import figures into PowerPoint
   - Add explanatory text
   - Practice 5-7 minute presentation

## Author

THAT Le Quang  
November 2025
