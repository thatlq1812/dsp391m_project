#!/bin/bash
# Train all models on super dataset for comparison

set -e

# Setup conda path
export PATH="/c/ProgramData/miniconda3/bin:/c/ProgramData/miniconda3:/c/ProgramData/miniconda3/Scripts:$PATH"

DATASET="data/processed/super_dataset_1year.parquet"
OUTPUT_BASE="outputs/super_dataset_comparison"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${OUTPUT_BASE}/run_${TIMESTAMP}"

echo "================================================================================"
echo "TRAINING ALL MODELS ON SUPER DATASET"
echo "================================================================================"
echo "Dataset: ${DATASET}"
echo "Output: ${RUN_DIR}"
echo ""

# Check if dataset exists
if [ ! -f "${DATASET}" ]; then
    echo "ERROR: Dataset not found: ${DATASET}"
    echo "Please run generate_super_dataset.py first!"
    exit 1
fi

# Create output directory
mkdir -p "${RUN_DIR}"

echo "[1/3] Training GraphWaveNet..."
conda run -n dsp python scripts/training/train_graphwavenet_baseline.py \
    --dataset "${DATASET}" \
    --output-dir "${RUN_DIR}/graphwavenet" \
    --epochs 50 \
    --batch-size 32

echo ""
echo "[2/3] Training LSTM..."
conda run -n dsp python scripts/training/train_lstm_baseline.py \
    --dataset "${DATASET}" \
    --output-dir "${RUN_DIR}/lstm" \
    --epochs 50 \
    --batch-size 64

echo ""
echo "[3/3] Training STMGT..."
conda run -n dsp python scripts/training/train_stmgt.py \
    --dataset "${DATASET}" \
    --output-dir "${RUN_DIR}/stmgt" \
    --epochs 50 \
    --batch-size 32

echo ""
echo "================================================================================"
echo "ALL MODELS TRAINED"
echo "================================================================================"
echo "Results saved to: ${RUN_DIR}"
echo ""
echo "Next steps:"
echo "  1. Compare results: python scripts/evaluation/compare_all_models.py"
echo "  2. Generate report: python scripts/analysis/create_final_report.py"
