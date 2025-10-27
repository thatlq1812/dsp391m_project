# Traffic Forecast System - Quick Reference
## Maintainer Contact
- **Name:** THAT Le Quang (Xiel)
- **GitHub:** [thatlq1812](https://github.com/thatlq1812)
- **Email:** fxlqthat@gmail.com / thatlqse183256@fpt.edu.com / thatlq1812@gmail.com
- **Phone:** +84 33 863 6369 / +84 39 730 6450
## Essential Commands
### Environment
```bash
# Activate environment
conda activate dsp
# Update environment
conda env update -f environment.yml
```
### Data Collection
```bash
# Single collection
python scripts/collect_and_render.py --once
# View schedule
python scripts/collect_and_render.py --print-schedule
# Start adaptive collection
python scripts/collect_and_render.py --adaptive
# Or use helper script
bash scripts/start_collection.sh
```
### Service Management (Production)
```bash
# Start service
sudo systemctl start traffic-forecast.service
# Stop service
sudo systemctl stop traffic-forecast.service
# Status
sudo systemctl status traffic-forecast.service
# View logs
sudo journalctl -u traffic-forecast.service -f
```
### Monitoring
```bash
# Health check
bash scripts/health_check.sh
# Monitor collection status
bash scripts/monitor_collection.sh
# View logs
tail -f logs/collector.log
# Check disk usage
du -sh data/ cache/ logs/
# Database size (if using SQLite)
du -sh traffic_history.db
```
### Data Download from VM
```bash
# RECOMMENDED: Fast compressed download (5-10x faster)
bash scripts/download_data_compressed.sh
# Download as tar.gz (default)
bash scripts/download_data_compressed.sh traffic-collector-v4 asia-southeast1-b ./data/downloads/my_data tar.gz
# Download as zip
bash scripts/download_data_compressed.sh traffic-collector-v4 asia-southeast1-b ./data/downloads/my_data zip
# Old method (deprecated, slow)
bash scripts/download_data.sh
```
### Data Management
```bash
# Backfill missing Overpass data
python scripts/backfill_overpass_data.py --dry-run
python scripts/backfill_overpass_data.py # Apply changes
# Fix Overpass cache bug on VM
bash scripts/fix_overpass_cache.sh
```
### Maintenance
```bash
# Cleanup old runs
python scripts/cleanup_runs.py --days 14
# Backup
bash scripts/backup.sh
# Update code on VM
git pull
conda env update -f environment.yml
```
## Configuration Files
- **Main config**: `configs/project_config.yaml`
- **Environment**: `.env` (copy from `.env.template`)
- **Schema**: `configs/nodes_schema_v2.json`
## Important Directories
- **Data**: `data/node/` (collected data)
- **Processed**: `data/processed/` (analysis results)
- **Logs**: `logs/` (system logs)
- **Models**: `models/` (trained models)
- **Cache**: `cache/` (API cache)
## Key Settings
### Toggle Mock API (FREE vs PAID)
Edit `configs/project_config.yaml`:
```yaml
google_directions:
use_mock_api: true # true = FREE, false = PAID ($720/month)
```
### Adjust Collection Schedule
Edit `configs/project_config.yaml`:
```yaml
scheduler:
mode: adaptive # or 'fixed'
adaptive:
peak_interval_minutes: 30
offpeak_interval_minutes: 60
weekend_interval_minutes: 90
```
### Change Number of Nodes
Edit `configs/project_config.yaml`:
```yaml
node_selection:
max_nodes: 64 # Increase/decrease as needed
```
## Cost Information
### Academic v4.0 (Current)
- Nodes: 64
- Collections/day: 25
- Monthly cost: $720 (real API) or $0 (mock API)
### Calculate Your Cost
```python
from traffic_forecast.scheduler import AdaptiveScheduler
import yaml
with open('configs/project_config.yaml') as f:
config = yaml.safe_load(f)
scheduler = AdaptiveScheduler(config['scheduler'])
cost = scheduler.get_cost_estimate(nodes=64, k_neighbors=3, days=30)
print(f"Collections/day: {cost['collections_per_day']}")
print(f"Monthly cost: ${cost['total_cost_usd']:.2f}")
```
## Troubleshooting
### Import Errors
```bash
conda activate dsp
pip install -r requirements.txt --upgrade
```
### Permission Errors
```bash
chmod +x scripts/*.sh
chmod 755 scripts/*.py
```
### Database Locked
```bash
pkill -f collect_and_render.py
rm traffic_history.db-journal
```
### API Errors
```bash
# Switch to mock API
# Edit configs/project_config.yaml:
# use_mock_api: true
```
## Team Collaboration
### Connect to GVM
```bash
ssh USERNAME@SERVER_IP
cd /opt/traffic-forecast
conda activate dsp
```
### View System Status
```bash
bash scripts/health_check.sh
systemctl status traffic-forecast.service
```
## Documentation
- **Deployment**: [DEPLOY.md](DEPLOY.md)
- **Runbook**: [notebooks/RUNBOOK.ipynb](notebooks/RUNBOOK.ipynb)
- **API Costs**: [doc/reference/GOOGLE_API_COST_ANALYSIS.md](doc/reference/GOOGLE_API_COST_ANALYSIS.md)
- **Academic v4.0**: [doc/reference/ACADEMIC_V4_SUMMARY.md](doc/reference/ACADEMIC_V4_SUMMARY.md)
## Support
- Check logs: `tail -f logs/service.log`
- Health check: `bash scripts/health_check.sh`
- Documentation: `DEPLOY.md` and `notebooks/RUNBOOK.ipynb`
---
**Version**: Academic v4.0 
**Last Updated**: October 25, 2025
