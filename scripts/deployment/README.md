# Git-Based Deployment Scripts

New scripts for Git-based workflow (recommended).

## Quick Reference

```bash
# Deploy changes
./scripts/deployment/deploy_git.sh

# Check status  
./scripts/deployment/status.sh

# Monitor logs
./scripts/deployment/monitor_logs.sh

# Download data
./scripts/data/download_latest.sh
```

## Deployment Scripts (`deployment/`)

### `deploy_git.sh` ⭐ **Main deployment script**
- Git-based deployment workflow
- Auto push → pull → restart
- Regenerates topology if config changed

### `status.sh` - System status check
- Service status
- Recent collections  
- Disk usage
- Topology info

### `monitor_logs.sh` - Real-time logs
- Tail adaptive_scheduler.log
- Watch collections live

### `restart.sh` - Restart service
- Stop → Start → Status
- Safe restart with verification

## Data Scripts (`data/`)

### `download_latest.sh` ⭐ **Download data**
- Interactive download
- Choose: latest / last 10 / 24h / all
- Requires gcloud

### `download_simple.sh` - No-auth download
- HTTP-based download
- No gcloud needed
- Requires VM server running

### `serve_data_public.sh` - Public server
- Run on VM
- Serves data via HTTP
- Port 8080

## Monitoring Scripts (`monitoring/`)

### `view_stats.sh` - Collection statistics
- Total runs
- Success rate
- Collections by hour
- Latest runs detail

### `health_check_remote.sh` - Health check
- 7-point system check
- Service, disk, memory, network
- Returns healthy/issues status

## See Full Documentation

- [QUICK_START_GIT.md](../docs/QUICK_START_GIT.md) - Quick start guide
- [README.md](./README.md) - Full scripts reference
- [DEPLOYMENT_GUIDE.md](../docs/v5/DEPLOYMENT_GUIDE.md) - Detailed deployment

## Migration from Old Scripts

| Old Script | New Script | Notes |
|------------|------------|-------|
| `deploy_wizard.sh` | `deployment/deploy_git.sh` | Git-based, automated |
| `control_panel.sh` | `deployment/status.sh` | Simpler, remote-friendly |
| `data/download_data.sh` | `data/download_latest.sh` | Interactive, better UX |
| `monitoring/health_check.sh` | `monitoring/health_check_remote.sh` | Remote execution |

Old scripts still work but new scripts are recommended for Git workflow.
