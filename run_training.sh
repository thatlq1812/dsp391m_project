#!/bin/bash
# Simple training launcher for STMGT baseline 1-month

cd "$(dirname "$0")"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="outputs/stmgt_baseline_1month_${TIMESTAMP}"
LOG_FILE="training_${TIMESTAMP}.log"

echo "========================================="
echo "STMGT Training - Baseline 1 Month"
echo "========================================="
echo "Output: ${OUTPUT_DIR}"
echo "Log: ${LOG_FILE}"
echo ""
echo "Training will take ~10-22 hours"
echo "Press Ctrl+C to stop (not recommended)"
echo "========================================="
echo ""

python scripts/training/train_stmgt.py \
    --config configs/training/stmgt_baseline_1month.json \
    --output "${OUTPUT_DIR}" \
    2>&1 | tee "${LOG_FILE}"

echo ""
echo "========================================="
echo "Training completed!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "========================================="
