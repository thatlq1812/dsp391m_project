#!/bin/bash
# Train LSTM and STMGT on 1-year super dataset

set -e

# Setup conda path
export PATH="/c/ProgramData/miniconda3/bin:/c/ProgramData/miniconda3:/c/ProgramData/miniconda3/Scripts:$PATH"

DATASET="data/processed/super_dataset_1year.parquet"
OUTPUT_BASE="outputs/super_dataset_1year"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "================================================================================"
echo "TRAINING LSTM AND STMGT ON 1-YEAR SUPER DATASET"
echo "================================================================================"
echo "Dataset: ${DATASET}"
echo "Output: ${OUTPUT_BASE}"
echo "Started: ${TIMESTAMP}"
echo ""

# Check if dataset exists
if [ ! -f "${DATASET}" ]; then
    echo "ERROR: Dataset not found: ${DATASET}"
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_BASE}"

echo "[1/2] Training LSTM Baseline..."
echo "Expected time: 2-3 hours"
echo "--------------------------------------------------------------------------------"
conda run -n dsp python scripts/training/train_lstm_baseline.py \
    --dataset "${DATASET}" \
    --output-dir "${OUTPUT_BASE}/lstm" \
    --epochs 50 \
    --batch-size 64

echo ""
echo "[1/2] LSTM training complete!"
echo ""

echo "[2/2] Training STMGT..."
echo "Expected time: 4-5 hours"
echo "--------------------------------------------------------------------------------"
conda run -n dsp python scripts/training/train_stmgt.py \
    --config configs/train_super_dataset_1year.json \
    --output-dir "${OUTPUT_BASE}/stmgt"

echo ""
echo "================================================================================"
echo "ALL MODELS TRAINED"
echo "================================================================================"
echo "Results saved to: ${OUTPUT_BASE}"
echo ""
echo "Summary:"
echo "  - LSTM:  ${OUTPUT_BASE}/lstm"
echo "  - STMGT: ${OUTPUT_BASE}/stmgt"
echo ""
echo "Next steps:"
echo "  1. Compare results with GraphWaveNet"
echo "  2. Generate comparison report"
echo "================================================================================"
