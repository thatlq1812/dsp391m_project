# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# STMGT Dashboard V4

Complete Project Management and ML Operations Dashboard

## Documentation

For complete documentation, see:

- [Quick Start Guide](../docs/DASHBOARD_V4_QUICKSTART.md) - Get running in 5 minutes
- [Quick Reference](../docs/DASHBOARD_V4_REFERENCE.md) - One-page cheat sheet
- [Project Changelog](../docs/CHANGELOG.md) - Full version history
- [VM Configuration](../docs/VM_CONFIG_INTEGRATION.md) - Infrastructure setup
- [Architecture](../docs/STMGT_ARCHITECTURE.md) - Model architecture

## Quick Start

```bash
# Activate environment
conda activate dsp

# Launch dashboard (Phase 1 schema + registry integration)
streamlit run dashboard/Dashboard.py
```

Open: http://localhost:8501

## Dashboard Structure

12 pages organized in 4 groups:

**Infrastructure and DevOps (Pages 1-4)**

1. System Overview
2. VM Management
3. Deployment
4. Monitoring and Logs

**Data Pipeline (Pages 5-7)** 5. Data Collection 6. Data Overview 7. Data Augmentation

**ML Workflow (Pages 8-10)** 8. Data Visualization 9. Training Control 10. Model Registry

**Production (Pages 11-12)** 11. Predictions 12. API Integration

## Configuration

VM Configuration: `configs/vm_config.json`

```json
{
  "project_id": "sonorous-nomad-476606-g3",
  "zone": "asia-southeast1-a",
  "instance_name": "traffic-forecast-collector"
}
```

## Common Tasks

**Collect Data**

```bash
# Page 5 -> Copy "Single Collection" command -> Run in terminal
```

**Train Model**

```bash
# Page 9 -> Copy training command (conda run) -> Execute manually
```

**Deploy to VM**

```bash
# Page 3 -> Copy deployment command -> Execute in terminal
```

**Generate Predictions**

```bash
# Page 11 -> Select model -> Run "Generate" to preview forecast
```

## Prerequisites

```bash
# Google Cloud SDK
gcloud --version

# Python 3.10+
python --version

# Conda
conda --version
```

## Troubleshooting

**Dashboard will not start**

```bash
pip install --upgrade streamlit
streamlit run dashboard/Dashboard.py
```

**VM won't connect**

```bash
gcloud auth login
gcloud compute instances list
```

**Model not found**

```bash
ls -la outputs/stmgt_*
```

## Support

- Documentation: [docs/](../docs/)
- Issues: GitHub Issues
- Team: DSP391m

---

Version: 4.0  
Last Updated: November 2, 2025
