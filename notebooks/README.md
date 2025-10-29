# Notebooks - Deprecated

**Status:** Jupyter notebooks have been replaced with interactive shell scripts.

## Why Shell Scripts?

1. **Simpler**: No Jupyter dependencies
2. **Faster**: Direct execution
3. **Portable**: Works everywhere with bash
4. **Version control friendly**: Plain text
5. **Production ready**: Same scripts work locally and on servers

## Replacement Scripts

### Control Panel (replaces all interactive notebooks)

```bash
bash scripts/control_panel.sh
```

**Features:**
- Data collection (single, test, adaptive)
- Data management (view, merge, cleanup, export)
- Visualization
- API testing
- System monitoring

### Deployment Wizard (replaces GCP_DEPLOYMENT.ipynb)

```bash
bash scripts/deploy_wizard.sh
```

**Features:**
- Step-by-step GCP deployment
- Interactive menu
- Automatic environment setup
- Service monitoring
- Data download

## Migration Guide

If you need notebook functionality:

1. **Local development** → Use `bash scripts/control_panel.sh`
2. **GCP deployment** → Use `bash scripts/deploy_wizard.sh`
3. **Custom analysis** → Use Python scripts directly:
   ```python
   from traffic_forecast.collectors import GoogleCollector
   from traffic_forecast.scheduler import AdaptiveScheduler
   # Your custom code here
   ```

## See Also

- [docs/QUICK_START.md](../docs/QUICK_START.md) - Getting started
- [scripts/README.md](../scripts/README.md) - Scripts reference
- [docs/DEPLOYMENT.md](../docs/DEPLOYMENT.md) - Deployment guide

---

**Traffic Forecast v5.1** - Shell Scripts > Notebooks
