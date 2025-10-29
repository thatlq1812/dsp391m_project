# 📊 Traffic Forecasting - Jupyter Notebooks

Complete pipeline for traffic data analysis, modeling, and deployment.

---

## 📁 Available Notebooks

### 🚀 `00_Data_Pipeline_and_Modeling.ipynb` ⭐ **MAIN NOTEBOOK**

**All-in-One Complete Pipeline** - From data download to trained models.

#### ✨ Key Features:

**🎛️ Fully Configurable**
- Enable/disable any pipeline step
- Choose data source (download new or use existing)
- Quick mode for faster execution
- Control visualization output

**📋 Pipeline Steps:**

1. **Configuration** - Set pipeline options
   - Data source selection
   - Step enablers
   - Analysis options
   - Output paths

2. **Data Download** *(optional)*
   - Download latest from VM
   - Or use existing local data
   - Automatic data validation

3. **Data Exploration** *(optional)*
   - Preview all collection runs
   - Inspect latest run details
   - Validate data structure

4. **Preprocessing** *(optional)*
   - JSON → Parquet conversion (10x faster)
   - Add time-based features
   - Create congestion levels
   - Merge weather data

5. **Comprehensive EDA** *(optional)*
   - Traffic speed distribution
   - Hourly traffic patterns
   - Interactive Plotly charts
   - Geographic maps (Folium)
   - Correlation analysis

6. **Feature Engineering** *(required for ML)*
   - Lag features (previous speeds)
   - Rolling statistics
   - Cyclical time encoding
   - Rush hour indicators

7. **Model Training** *(required for ML)*
   - Linear Regression
   - Ridge Regression
   - Random Forest
   - Gradient Boosting
   - Auto performance comparison

8. **Save Models** *(optional)*
   - Export trained models
   - Save feature metadata
   - Export comparison results

#### 📊 What You Get:

- ✅ Trained ML models (`.pkl` files)
- ✅ Model comparison report (CSV)
- ✅ Performance visualizations
- ✅ Feature importance analysis
- ✅ Ready-to-deploy models

#### ⚙️ Configuration Example:

```python
# Step 1: Set your preferences
USE_EXISTING_DATA = False       # Download new data
ENABLE_COMPREHENSIVE_EDA = True  # Full EDA analysis
QUICK_MODE = False              # Train all 4 models
SHOW_INTERACTIVE_MAPS = True    # Include Folium maps
ENABLE_SAVE_MODELS = True       # Export models

# Then just run all cells!
```

#### 🎯 Use Cases:

| Scenario | Configuration |
|----------|---------------|
| **First Time User** | All defaults, run all steps |
| **Quick Test** | `QUICK_MODE = True`, skip EDA |
| **EDA Only** | Disable training, enable all EDA |
| **Model Training** | `USE_EXISTING_DATA = True`, skip exploration |
| **Re-train Models** | Use preprocessed data, train only |

---

## 🚀 Quick Start

### 1️⃣ **Setup Environment**

```bash
# Activate conda environment
conda activate dsp

# Install visualization libraries
pip install folium plotly seaborn matplotlib
```

### 2️⃣ **Open Notebook**

```bash
# Using Jupyter Lab (recommended)
jupyter lab notebooks/00_Data_Pipeline_and_Modeling.ipynb

# Or VS Code
code notebooks/00_Data_Pipeline_and_Modeling.ipynb
```

### 3️⃣ **Configure & Run**

```python
# Cell 1: Adjust configuration
USE_EXISTING_DATA = False  # Set to True if you have data

# Then: Run All Cells (Ctrl/Cmd + Shift + Enter)
```

---

## 📖 Usage Scenarios

### Scenario 1: **Complete First-Time Run**

```python
# Configuration (default)
USE_EXISTING_DATA = False
ENABLE_DATA_EXPLORATION = True
ENABLE_PREPROCESSING = True
ENABLE_COMPREHENSIVE_EDA = True
ENABLE_FEATURE_ENGINEERING = True
ENABLE_MODEL_TRAINING = True
ENABLE_SAVE_MODELS = True
QUICK_MODE = False

# Result: Complete pipeline execution (15-20 minutes)
```

### Scenario 2: **Quick Model Training**

```python
# Configuration
USE_EXISTING_DATA = True         # Use current data
ENABLE_DATA_EXPLORATION = False  # Skip preview
ENABLE_PREPROCESSING = False     # Skip if already done
ENABLE_COMPREHENSIVE_EDA = False # Skip visualizations
ENABLE_FEATURE_ENGINEERING = True
ENABLE_MODEL_TRAINING = True
QUICK_MODE = True               # Only 2 fast models

# Result: Fast training (2-3 minutes)
```

### Scenario 3: **EDA & Visualization Only**

```python
# Configuration
USE_EXISTING_DATA = True
ENABLE_DATA_EXPLORATION = True
ENABLE_PREPROCESSING = True
ENABLE_COMPREHENSIVE_EDA = True  # Full EDA
SHOW_INTERACTIVE_MAPS = True
SHOW_PLOTLY_CHARTS = True
ENABLE_FEATURE_ENGINEERING = False
ENABLE_MODEL_TRAINING = False    # No training

# Result: Comprehensive analysis only (5-8 minutes)
```

### Scenario 4: **Re-train with New Data**

```python
# Configuration
USE_EXISTING_DATA = False        # Download fresh
ENABLE_DATA_EXPLORATION = False
ENABLE_PREPROCESSING = True
ENABLE_COMPREHENSIVE_EDA = False
ENABLE_FEATURE_ENGINEERING = True
ENABLE_MODEL_TRAINING = True
ENABLE_SAVE_MODELS = True
QUICK_MODE = False

# Result: New data → trained models (10-12 minutes)
```

---

## 📊 Expected Outputs

### Console Output
```
✅ Configuration loaded!
🔽 Downloading latest data from VM...
📊 Found 18 collection runs
⚙️  Preprocessing data...
✅ Preprocessing completed!
🤖 Training models...
✅ All models trained!
🏆 Best Model: Random Forest
   RMSE: 3.245 km/h
   R²: 0.892
💾 Saving models...
✅ All artifacts saved
🎉 PIPELINE EXECUTION COMPLETE!
```

### Generated Files
```
data/
├── processed/
│   ├── run_*.parquet              # Individual runs
│   └── all_runs_combined.parquet  # Combined dataset

traffic_forecast/models/saved/
├── linear_regression.pkl
├── ridge_regression.pkl
├── random_forest.pkl
├── gradient_boosting.pkl
├── feature_info.pkl
└── model_comparison.csv
```

### Visualizations
- 📊 Traffic speed distribution (4 subplots)
- 📈 Hourly traffic patterns (2 subplots)
- 🏆 Model performance comparison (3 charts)
- 🗺️ Geographic maps (if enabled)

---

## 🎓 Tips & Best Practices

### ⚡ For Faster Execution:

1. **Use preprocessed data**
   ```python
   USE_PREPROCESSED = True
   ENABLE_PREPROCESSING = False
   ```

2. **Enable quick mode**
   ```python
   QUICK_MODE = True  # Trains only 2 models
   ```

3. **Skip visualizations**
   ```python
   ENABLE_COMPREHENSIVE_EDA = False
   SHOW_INTERACTIVE_MAPS = False
   ```

### 📊 For Comprehensive Analysis:

1. **Enable all visualizations**
   ```python
   ENABLE_COMPREHENSIVE_EDA = True
   SHOW_INTERACTIVE_MAPS = True
   SHOW_PLOTLY_CHARTS = True
   ```

2. **Train all models**
   ```python
   QUICK_MODE = False  # Trains 4 models
   ```

3. **Verbose output**
   ```python
   VERBOSE = True  # Detailed progress
   ```

### 🔄 For Iterative Development:

1. **Use existing data** to save time
2. **Skip exploration** after first run
3. **Focus on specific steps** you're working on
4. **Enable saving** only when satisfied with results

---

## ⚠️ Troubleshooting

### Issue: "No data found"
**Solution:**
```python
USE_EXISTING_DATA = False  # Download new data
```

### Issue: "Models not training"
**Solution:**
```python
ENABLE_FEATURE_ENGINEERING = True  # Required for training
ENABLE_MODEL_TRAINING = True
```

### Issue: "Slow execution"
**Solution:**
```python
QUICK_MODE = True
ENABLE_COMPREHENSIVE_EDA = False
```

### Issue: "Visualizations not showing"
**Solution:**
```bash
pip install folium plotly matplotlib seaborn
```

---

## 📚 Additional Resources

- **Developer Guide**: `docs/DEVELOPER_GUIDE.md`
- **Preprocessing Guide**: `docs/PREPROCESSING_GUIDE.md`
- **Team Guide**: `docs/TEAM_GUIDE.md`
- **Download Scripts**: `scripts/data/`

---

## 🤝 Contributing

When adding new analysis:

1. Add configuration flag in Step 1
2. Use conditional execution: `if ENABLE_YOUR_STEP:`
3. Add to summary at the end
4. Update this README

---

**Last Updated:** October 30, 2025  
**Notebook Version:** 2.0 (Unified Pipeline)  
**Compatible With:** Python 3.10+, Pandas 2.0+
- **Option B**: Run cells sequentially to understand each step

---

## 📊 Key Visualizations

The notebook includes:

- 📈 **Interactive Plotly charts** - Zoom, pan, hover for details
- 🗺️ **Folium maps** - Click markers for intersection details
- 📊 **Statistical distributions** - Histograms, box plots, scatter plots
- 🔥 **Heatmaps** - Correlation matrices, geographic heatmaps
- ⏰ **Time series plots** - Traffic patterns over time
- 📉 **Multi-panel dashboards** - Comprehensive overview

---

## 💡 Tips for Best Results

### Memory Management

```python
# If running low on memory, load fewer runs
runs = runs[-5:]  # Last 5 runs only
```

### Interactive Maps

- Click on intersection markers to see details
- Use `+/-` to zoom in/out
- Drag to pan the map

### Export Visualizations

```python
# Save Plotly figure as HTML
fig.write_html("output/traffic_analysis.html")

# Save map
m.save("output/traffic_map.html")
```

---

## 📈 Expected Results

After running the notebook, you should see:

✅ **Network Statistics**

- Total nodes, edges, coverage area
- Degree distribution
- Road type distribution

✅ **Traffic Insights**

- Average speed: ~22-28 km/h
- Congestion levels
- Peak/off-peak patterns

✅ **Weather Conditions**

- Temperature: ~25-27°C (typical for HCMC)
- Wind speed: ~5-10 km/h
- Rainfall patterns

✅ **Data Quality**

- Completeness: >95%
- Quality score: >90/100

---

## 🔧 Troubleshooting

### Issue: Import errors

```bash
# Install missing packages
pip install pandas numpy matplotlib seaborn plotly folium scipy
```

### Issue: Memory error

```python
# Reduce data size
df_all_traffic = df_all_traffic.sample(frac=0.5)  # Use 50% of data
```

### Issue: Slow rendering

```python
# Reduce marker count on maps
df_nodes_sample = df_nodes.sample(n=20)  # Show 20 nodes instead of 64
```

---

## 📚 Next Steps

After completing EDA:

1. **Feature Engineering**

   - Create lag features
   - Spatial features (nearby intersection speeds)
   - Temporal features (hour, day, weekend)

2. **Model Development**

   - Baseline models (Moving Average, ARIMA)
   - LSTM/GRU for time series
   - Graph Neural Networks for spatial dependencies

3. **Evaluation**
   - RMSE, MAE metrics
   - Visual inspection
   - Cross-validation

---

## 📞 Support

- Check **TEAM_GUIDE.md** for data download help
- Check **DEVELOPER_GUIDE.md** for deployment issues
- Review **project_config.yaml** for configuration

---

**Last Updated:** October 30, 2025  
**Author:** Traffic Forecasting Team  
**Project:** DSP391m - Real-time Traffic Forecasting
