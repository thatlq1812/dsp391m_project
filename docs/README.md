# Documentation - Traffic Forecast v5.1

Complete documentation for the Traffic Forecast system.

## 📖 Main Guides

### Getting Started

- **[QUICK_START.md](QUICK_START.md)** - 5-minute quick start guide
  - Setup environment
  - Run first collection
  - Use control panel
  - Deploy to GCP

### Deployment

- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Complete deployment guide
  - Prerequisites
  - Interactive wizard
  - Manual deployment
  - Monitoring
  - Cost estimation
  - Troubleshooting

### Operations

- **[OPERATIONS.md](OPERATIONS.md)** - Daily operations & maintenance
  - Service management
  - Data management
  - Monitoring
  - Cost monitoring
  - Troubleshooting
  - Update procedures

## 📚 Reference Documentation

### Scripts

- **[../scripts/README.md](../scripts/README.md)** - Scripts reference
  - Interactive scripts
  - Collection scripts
  - Data management scripts
  - Utility scripts

### Configuration

- **[../configs/project_config.yaml](../configs/project_config.yaml)** - Configuration file
  - Scheduler settings
  - API configuration
  - Collection parameters

### Version History

- **[../CHANGELOG.md](../CHANGELOG.md)** - Version changelog
  - v5.1 features
  - v5.0 features
  - Migration guides

## 📁 Additional Documentation

### v5 Archive

- **[v5/](v5/)** - v5.0 documentation archive
  - Original v5.0 docs
  - Detailed technical reports
  - Development history

### Full Reference

- **[README_v5_full.md](README_v5_full.md)** - Complete v5.0 README (archived)
  - Detailed feature comparison
  - Extended examples
  - Historical context

## 🚀 Quick Navigation

**I want to...**

- **...get started quickly** → [QUICK_START.md](QUICK_START.md)
- **...deploy to GCP** → [DEPLOYMENT.md](DEPLOYMENT.md)
- **...manage running system** → [OPERATIONS.md](OPERATIONS.md)
- **...understand scripts** → [../scripts/README.md](../scripts/README.md)
- **...configure system** → [../configs/project_config.yaml](../configs/project_config.yaml)
- **...see what changed** → [../CHANGELOG.md](../CHANGELOG.md)

## 📝 Documentation Structure

```
docs/
├── README.md                # This file - documentation index
├── QUICK_START.md           # Quick start guide
├── DEPLOYMENT.md            # Deployment guide
├── OPERATIONS.md            # Operations guide
├── README_v5_full.md        # Archived full v5.0 README
└── v5/                      # v5.0 documentation archive
    ├── README.md
    ├── HOAN_TAT_V5.md
    ├── DEPLOYMENT_GUIDE.md
    └── ...
```

## 🔗 External Resources

- **Google Directions API**: https://developers.google.com/maps/documentation/directions
- **Open-Meteo API**: https://open-meteo.com/en/docs
- **Overpass API**: https://wiki.openstreetmap.org/wiki/Overpass_API
- **Google Cloud Platform**: https://console.cloud.google.com

## 🆘 Getting Help

1. **Check documentation**: Start with QUICK_START.md
2. **Check operations guide**: Common issues in OPERATIONS.md
3. **Use interactive tools**:
   - Local: `bash scripts/control_panel.sh`
   - Deploy: `bash scripts/deploy_wizard.sh`
4. **Check logs**:
   - Local: `logs/` directory
   - GCP: `~/traffic-forecast/logs/` on VM

---

**Traffic Forecast v5.1** - Documentation Index
