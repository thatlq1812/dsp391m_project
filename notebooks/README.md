# Traffic Forecast Notebooks

## Notebook Overview

This directory contains **interactive Jupyter notebooks** for managing the entire Traffic Forecast pipeline.

---

## Available Notebooks

### 1. CONTROL_PANEL.ipynb

**Purpose:**Main control center for data collection pipeline

**Features:**

- Configuration management
- Node topology operations
- Data collection (test + full)
- Data quality validation
- Visualizations (maps, distributions)
- Deployment package creation
- Cost estimation
- Monitoring & logging

**When to use:**

- Daily: Check system status
- Before deployment: Validate configuration
- During collection: Monitor progress
- Post-collection: Validate data

**Quick actions:**

```python
run_collection_test() # Test with 5 edges
run_collection_full() # All 234 edges
validate_collection() # Check data quality
plot_node_coverage() # Visualize nodes
estimate_api_costs() # Calculate costs
```

---

### 2. GCP_DEPLOYMENT.ipynb

**Purpose:**Google Cloud Platform VM deployment and management

**Features:**

- VM creation and configuration
- Project deployment to VM
- Cron job scheduling
- Log monitoring
- Data download from VM
- VM start/stop/delete

**When to use:**

- Initial: Deploy to GCP VM
- Daily: Monitor collection on VM
- After 7 days: Download data and stop VM

**Quick actions:**

```python
create_vm_instance() # Create GCP VM
upload_project_files() # Upload project
deploy_on_vm() # Run deployment
setup_cron_collection(60) # Hourly collection
check_collection_logs() # Monitor logs
download_collected_data() # Get data from VM
stop_vm() # Stop to prevent charges
```

---

### 3. DATA_DASHBOARD.ipynb

**Purpose:**Exploratory Data Analysis and visualization

**Features:**

- Load and explore collected data
- Time series analysis
- Geographic visualization
- Traffic pattern analysis
- Weather correlation
- Statistical summaries

**When to use:**

- After data collection
- For EDA and insights
- Generating reports

---

### 4. ML_TRAINING.ipynb

**Purpose:**Machine learning model training and evaluation

**Features:**

- Feature engineering
- Model training (XGBoost, RF, LightGBM)
- Performance evaluation
- Hyperparameter tuning
- Model persistence

**When to use:**

- After collecting 7+ days of data
- For academic project analysis
- Model comparison and selection

---

## Typical Workflow

### Phase 1: Setup (Local)

1. Open **CONTROL_PANEL.ipynb**
2. Run all cells to verify setup
3. Test collection locally:
```python
run_collection_test()
```

### Phase 2: Deployment (GCP)

1. Open **GCP_DEPLOYMENT.ipynb**
2. Create VM and deploy:
```python
create_vm_instance()
upload_project_files()
deploy_on_vm()
setup_cron_collection(60) # Hourly
```

### Phase 3: Monitoring (Daily)

1. Check logs in **GCP_DEPLOYMENT.ipynb**:

```python
check_collection_logs()
validate_vm_data()
estimate_current_costs()
```

2. Or use **CONTROL_PANEL.ipynb** locally after downloading sample

### Phase 4: Data Analysis (After 7 days)

1. Download data:

```python
# In GCP_DEPLOYMENT.ipynb
download_collected_data()
stop_vm()
```

2. Explore in **DATA_DASHBOARD.ipynb**

3. Train models in **ML_TRAINING.ipynb**

---

## Notebook Comparison

| Notebook | Primary Use | Location | Frequency |
| -------------- | -------------------------- | -------- | ------------------- |
| CONTROL_PANEL | Local testing & validation | Local | Daily |
| GCP_DEPLOYMENT | VM management | Local | Setup + Daily check |
| DATA_DASHBOARD | EDA & visualization | Local | After collection |
| ML_TRAINING | Model training | Local | After collection |

---

## Tips

### For CONTROL_PANEL

- Run cells sequentially on first use
- Use quick actions for repeated tasks
- Check file status before collection

### For GCP_DEPLOYMENT

- Set `GCP_PROJECT_ID` before use
- Always verify VM status before operations
- Download data before stopping VM
- Monitor costs daily

### For DATA_DASHBOARD

- Load data from `data/vm_collected/`
- Generate visualizations for reports
- Export plots to `data/plots/`

### For ML_TRAINING

- Ensure sufficient data (7+ days recommended)
- Use cross-validation for evaluation
- Save trained models to `models/`

---

## Quick Start

### 1. Start Jupyter

```bash
cd /d/UNI/DSP391m/project
conda activate dsp
jupyter notebook
```

### 2. Open CONTROL_PANEL.ipynb

- Run cells 1-4 to check setup
- Test collection: Run cell with `run_collection_test()`

### 3. Deploy to GCP (Optional)

- Open GCP_DEPLOYMENT.ipynb
- Follow cells 1-6 for deployment

---

## Related Documentation

- **[../doc/v5/README_V5.md](../doc/v5/README_V5.md)** - Technical documentation
- **[../doc/v5/DEPLOYMENT_GUIDE.md](../doc/v5/DEPLOYMENT_GUIDE.md)** - GCP deployment guide
- **[../doc/v5/HOAN_TAT_V5.md](../doc/v5/HOAN_TAT_V5.md)** - Completion summary

---

## Important Notes

1. **Never commit** notebooks with sensitive data or API keys
2. **Clear output** before committing to git
3. **Download data** from VM before stopping/deleting
4. **Monitor costs** daily during GCP collection
5. **Validate data** regularly to catch issues early

---

**Last Updated:**October 29, 2025
**Version:** 5.0
**Status:**Production Ready
