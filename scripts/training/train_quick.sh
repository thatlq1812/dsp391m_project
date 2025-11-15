#!/bin/bash
# Quick start training script for STMGT
# Run with improved 10/10 architecture

set -e

PROJECT_ROOT="/d/UNI/DSP391m/project"
cd "$PROJECT_ROOT" || exit 1

echo "========================================"
echo "STMGT Training"
echo "========================================"
echo ""

# Check data exists
DATA_FILE="data/processed/all_runs_gapfilled_week.parquet"
if [ ! -f "$DATA_FILE" ]; then
    echo "Error: Data file not found: $DATA_FILE"
    exit 1
fi

echo "Data file found: $DATA_FILE"
echo ""

# Configuration
CONFIG_FILE="configs/train_final_config.json"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Warning: Final config not found, will use defaults"
    CONFIG_ARG=""
else
    echo "Using config: $CONFIG_FILE"
    CONFIG_ARG="--config $CONFIG_FILE"
fi

echo ""
echo "Starting training..."
echo "  - Model: STMGT v2 with normalization"
echo "  - Data: 62 nodes, 144 edges"
echo "  - Features: Speed + Weather + Temporal"
echo "  - Output: Gaussian Mixture (K=3)"
echo ""

# Train
python scripts/training/train_stmgt.py $CONFIG_ARG

echo ""
echo "========================================"
echo "Training Complete!"
echo "========================================"
echo ""
echo "Check output in: outputs/stmgt_v2_<timestamp>/"
echo "  - best_model.pt: Best checkpoint"
echo "  - final_model.pt: Final model"
echo "  - training_history.csv: Training metrics"
echo "  - test_results.json: Test performance"
