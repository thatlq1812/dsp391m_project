# Dashboard Migration Notes

**Date:** November 9, 2025  
**Action:** Replace Streamlit Dashboard with CLI Tool

---

## Status

- [x] Created CLI tool (`traffic_forecast/cli.py`)
- [x] Created installation script (`setup_cli.py`)
- [x] Created documentation (`docs/guides/CLI_USER_GUIDE.md`)
- [x] Created test script (`scripts/test_cli.sh`)
- [x] Updated CHANGELOG
- [ ] Test CLI thoroughly
- [ ] Archive dashboard directory

---

## Archive Dashboard After Testing

Once CLI is tested and working, archive the old dashboard:

```bash
# Move dashboard to archive
mv dashboard/ archive/dashboard_streamlit/

# Update .gitignore if needed
echo "archive/dashboard_streamlit/" >> .gitignore
```

---

## Dashboard Files to Archive

```
dashboard/
├── Dashboard.py (512 lines)
├── realtime_stats.py
├── README.md
├── pages/
│   ├── 2_Data_Overview.py
│   ├── 3_Data_Collection.py
│   ├── 4_Data_Augmentation.py
│   ├── 5_Data_Visualization.py
│   ├── 6_Training_Control.py
│   ├── 7_Model_Registry.py
│   ├── 8_Predictions.py
│   ├── 9_API_Integration.py
│   ├── 10_Monitoring_Logs.py
│   ├── 11_Deployment.py
│   ├── 12_VM_Management.py
│   └── 13_Legacy_ASTGCN.py
└── utils/

Total: 13 pages, ~2000+ lines
```

---

## Rationale

**Problems with Dashboard:**

1. Too many pages (13) - overwhelming
2. Many features don't work properly
3. Heavy dependencies (streamlit, plotly, altair)
4. Slow startup (5-10 seconds)
5. Can't use over SSH or in Docker easily
6. Not scriptable or automatable
7. Over-engineered for simple tasks

**Benefits of CLI:**

1. Simple, fast (instant startup)
2. Works anywhere (local, SSH, Docker)
3. Scriptable and automatable
4. Lightweight dependencies
5. Professional production tool
6. Easier to maintain

---

## Web Interface Strategy

For visualization and user interaction, we'll build a **separate** lightweight web interface:

- Pure HTML/CSS/JavaScript (no framework bloat)
- Leaflet.js for maps (already implemented in `route_planner.html`)
- Focus on traffic visualization and route planning
- Independent from management tools

**Separation of Concerns:**

- **CLI** = Management, monitoring, operations (for developers/ops)
- **Web** = Visualization, user interaction (for end users)

This is cleaner architecture than trying to do everything in Streamlit.

---

## Migration Checklist

- [ ] Install CLI: `pip install -e . -f setup_cli.py`
- [ ] Test all commands:
  - [ ] `stmgt --help`
  - [ ] `stmgt info`
  - [ ] `stmgt model list`
  - [ ] `stmgt model info <name>`
  - [ ] `stmgt model compare <name1> <name2>`
  - [ ] `stmgt api start`
  - [ ] `stmgt api status`
  - [ ] `stmgt api test`
  - [ ] `stmgt train status`
  - [ ] `stmgt train logs`
  - [ ] `stmgt data info`
- [ ] Update README with CLI instructions
- [ ] Remove dashboard references from docs
- [ ] Archive dashboard directory
- [ ] Update deployment scripts to use CLI
- [ ] Update CI/CD if applicable

---

## Commands Comparison

### Old (Streamlit Dashboard)

```bash
# Start dashboard
streamlit run dashboard/Dashboard.py

# Then manually:
# 1. Open browser to http://localhost:8501
# 2. Wait 5-10 seconds for load
# 3. Navigate through 13 pages
# 4. Click buttons (some don't work)
# 5. Wait for each page load
```

### New (CLI Tool)

```bash
# List models
stmgt model list                    # Instant

# Compare models
stmgt model compare model1 model2   # Instant

# Start API
stmgt api start                     # One command

# Check status
stmgt train status                  # Real-time

# View logs
stmgt train logs -f                 # Follow logs
```

**Result:** 10x faster, more reliable, professional!

---

## Future Enhancements

**CLI additions:**

- `stmgt deploy` - Deploy to VM
- `stmgt backup` - Backup models/data
- `stmgt predict` - Make predictions from CLI
- `stmgt experiment` - Run experiments
- Interactive mode with prompts
- Configuration wizard

**Web interface:**

- Real-time traffic map (already done)
- Route planning (already done)
- Performance dashboards
- Interactive charts
- Historical data viewer

---

**Author:** THAT Le Quang  
**GitHub:** thatlq1812
