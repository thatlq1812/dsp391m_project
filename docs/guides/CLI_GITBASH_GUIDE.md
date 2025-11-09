# STMGT CLI - Quick Start for Git Bash

**For Windows users with Git Bash**

---

## Problem

Git Bash on Windows doesn't recognize `python`, `pip`, or `conda` commands directly.

---

## Solution: Use Wrapper Script

We created `stmgt.sh` wrapper script that handles the full paths automatically.

---

## Usage

```bash
# From project root
cd /d/UNI/DSP391m/project

# Run any CLI command
./stmgt.sh --help
./stmgt.sh model list
./stmgt.sh api start
./stmgt.sh data info
./stmgt.sh train status
```

---

## Add to PATH (Optional)

For easier access from anywhere:

```bash
# Add to ~/.bashrc or ~/.bash_profile
echo 'export PATH="$PATH:/d/UNI/DSP391m/project"' >> ~/.bashrc
echo 'alias stmgt="/d/UNI/DSP391m/project/stmgt.sh"' >> ~/.bashrc

# Reload
source ~/.bashrc

# Now use from anywhere
stmgt --help
stmgt model list
```

---

## Examples

### Check Dataset

```bash
./stmgt.sh data info
```

**Output:**

```
┌─────────────── Dataset Information ───────────────┐
│ Dataset: all_runs_extreme_augmented.parquet       │
│ Rows:    205,920                                  │
│ Columns: 18                                       │
│ Avg Speed: 18.72 km/h                             │
└───────────────────────────────────────────────────┘
```

### List Models

```bash
./stmgt.sh model list
```

### Start API Server

```bash
./stmgt.sh api start
```

### Check API Status

```bash
./stmgt.sh api status
```

### View Training Status

```bash
./stmgt.sh train status
```

---

## Alternative: Full Command

If you prefer not to use the wrapper:

```bash
C:/ProgramData/miniconda3/Scripts/conda.exe run -n dsp python traffic_forecast/cli.py --help
```

---

## Troubleshooting

### Script not found

```bash
# Make sure you're in project root
cd /d/UNI/DSP391m/project

# Check if script exists
ls -la stmgt.sh

# Make executable if needed
chmod +x stmgt.sh
```

### Permission denied

```bash
chmod +x stmgt.sh
```

### Command not working

```bash
# Check conda path
ls -la C:/ProgramData/miniconda3/Scripts/conda.exe

# Check Python in environment
C:/ProgramData/miniconda3/Scripts/conda.exe run -n dsp python --version
```

---

## Benefits

- ✅ Works in Git Bash
- ✅ No PATH configuration needed
- ✅ Handles conda environment automatically
- ✅ Simple one-command usage
- ✅ All CLI features available

---

**Author:** THAT Le Quang  
**Date:** November 9, 2025
