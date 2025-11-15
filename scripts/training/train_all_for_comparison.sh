#!/bin/bash
# Master script to train all models with identical conditions for fair comparison

# Maintainer: THAT Le Quang (thatlq1812)

set -e  # Exit on error

# Activate conda environment
echo "Activating conda environment 'dsp'..."
eval "$(conda shell.bash hook 2>/dev/null)" || true
conda activate dsp 2>/dev/null || source activate dsp 2>/dev/null || true

# Verify Python is available
if ! command -v python &> /dev/null; then
    echo "ERROR: Python not found even after conda activation"
    echo "Please activate your Python environment manually and run this script"
    echo ""
    echo "Try: conda activate dsp"
    echo "Or:  source /path/to/your/venv/bin/activate"
    exit 1
fi

echo "Using Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

echo "========================================================================"
echo "TRAINING ALL MODELS FOR FINAL REPORT COMPARISON"
echo "========================================================================"
echo ""
echo "This script trains 3 models with identical conditions:"
echo "  1. STMGT V3 (current best)"
echo "  2. LSTM Baseline (temporal only)"
echo "  3. GraphWaveNet (adaptive graph + temporal)"
echo ""
echo "All models use:"
echo "  - Same dataset: data/processed/all_runs_gapfilled_week.parquet"
echo "  - Same split: 70/15/15 train/val/test"
echo "  - Same epochs: 100 with early stopping"
echo "  - Same evaluation metrics"
echo "========================================================================"
echo ""

# Configuration
DATASET="data/processed/all_runs_gapfilled_week.parquet"
EPOCHS=100
BATCH_SIZE=32
OUTPUT_BASE="outputs/final_comparison"

# Create output directory
mkdir -p "$OUTPUT_BASE"

# Timestamp for this comparison run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
COMPARISON_DIR="$OUTPUT_BASE/run_$TIMESTAMP"
mkdir -p "$COMPARISON_DIR"

echo "Comparison directory: $COMPARISON_DIR"
echo ""

# Save configuration
cat > "$COMPARISON_DIR/config.txt" <<EOF
Training Configuration for Fair Model Comparison
================================================

Dataset: $DATASET
Epochs: $EPOCHS
Batch Size: $BATCH_SIZE
Split: 70% train / 15% val / 15% test

Models:
1. STMGT V3 - Multi-modal spatial-temporal with Gaussian mixture
2. LSTM Baseline - Temporal only (no spatial information)
3. GraphWaveNet - Adaptive graph learning + dilated convolutions

Timestamp: $TIMESTAMP
================================================
EOF

echo "Configuration saved to $COMPARISON_DIR/config.txt"
echo ""

# Function to check if training succeeded
check_success() {
    local model_name=$1
    local output_dir=$2
    
    if [ -f "$output_dir/results.json" ] || [ -f "$output_dir/test_results.json" ]; then
        echo "[✓] $model_name training completed successfully"
        return 0
    fi

    local latest_run
    latest_run=$(ls -td "$output_dir"/run_* 2>/dev/null | head -n1 || true)
    if [ -n "$latest_run" ]; then
        if [ -f "$latest_run/results.json" ] || [ -f "$latest_run/test_results.json" ]; then
            echo "[✓] $model_name training completed successfully ($(basename "$latest_run"))"
            return 0
        fi
    fi

    echo "[✗] $model_name training failed"
    return 1
}

# ============================================================================
# 1. Train LSTM Baseline
# ============================================================================
echo "========================================================================"
echo "[1/3] Training LSTM Baseline"
echo "========================================================================"
echo ""

LSTM_OUTPUT="$COMPARISON_DIR/lstm_baseline"

python scripts/training/train_lstm_baseline.py \
    --dataset "$DATASET" \
    --output-dir "$LSTM_OUTPUT" \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --sequence-length 12 \
    --lstm-units 128 64 \
    --dropout 0.2 \
    --learning-rate 0.001

check_success "LSTM" "$LSTM_OUTPUT"
echo ""

# ============================================================================
# 2. Train GraphWaveNet
# ============================================================================
echo "========================================================================"
echo "[2/3] Training GraphWaveNet"
echo "========================================================================"
echo ""

GWNET_OUTPUT="$COMPARISON_DIR/graphwavenet"

python scripts/training/train_graphwavenet_baseline.py \
    --dataset "$DATASET" \
    --output-dir "$GWNET_OUTPUT" \
    --epochs $EPOCHS \
    --batch-size 16 \
    --sequence-length 12 \
    --num-layers 3 \
    --hidden-channels 32 \
    --kernel-size 2 \
    --dropout 0.2 \
    --learning-rate 0.001

check_success "GraphWaveNet" "$GWNET_OUTPUT"
echo ""

# ============================================================================
# 3. Train STMGT V3
# ============================================================================
echo "========================================================================"
echo "[3/3] Training STMGT V3"
echo "========================================================================"
echo ""

STMGT_OUTPUT="$COMPARISON_DIR/stmgt_v3"

python scripts/training/train_stmgt.py \
    --config configs/train_normalized_v3.json \
    --output-dir "$STMGT_OUTPUT"

check_success "STMGT V3" "$STMGT_OUTPUT"
echo ""

# ============================================================================
# Generate Comparison Report
# ============================================================================
echo "========================================================================"
echo "TRAINING COMPLETE - Generating Comparison Report"
echo "========================================================================"
echo ""

python scripts/training/compare_models.py \
    --comparison-dir "$COMPARISON_DIR" \
    --output "$COMPARISON_DIR/comparison_report.json"

echo ""
echo "========================================================================"
echo "ALL MODELS TRAINED SUCCESSFULLY"
echo "========================================================================"
echo ""
echo "Results saved to: $COMPARISON_DIR"
echo ""
echo "Next steps:"
echo "  1. Review comparison_report.json"
echo "  2. Generate visualizations with scripts/visualization/"
echo "  3. Update final report with results"
echo ""
echo "========================================================================"
