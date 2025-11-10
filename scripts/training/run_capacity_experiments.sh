#!/bin/bash
# STMGT Model Capacity Experiments
# Testing SMALLER models (< 680K params) to find optimal capacity
# Date: 2025-11-10

echo "======================================================================"
echo "STMGT CAPACITY REDUCTION EXPERIMENTS"
echo "======================================================================"
echo ""
echo "Based on experimental results:"
echo "  - V1 (680K):  MAE 3.08 ✓ BEST"
echo "  - V1.5 (850K): MAE 3.18 (worse)"
echo "  - V2 (1.15M):  MAE 3.22 (worse)"
echo ""
echo "Conclusion: Need to test SMALLER capacities"
echo ""
echo "======================================================================"
echo ""

PROJECT_DIR="/d/UNI/DSP391m/project"
CONDA_ENV="dsp"

# Function to run training
run_training() {
    local config=$1
    local label=$2
    local params=$3
    
    echo "----------------------------------------------------------------------"
    echo "Training: $label ($params)"
    echo "Config: $config"
    echo "----------------------------------------------------------------------"
    echo ""
    
    cd "$PROJECT_DIR"
    /c/ProgramData/miniconda3/Scripts/conda.exe run -n "$CONDA_ENV" --no-capture-output \
        python scripts/training/train_stmgt.py --config "configs/$config"
    
    echo ""
    echo "✓ Completed: $label"
    echo ""
}

# Training queue (order by priority)
echo "Training Queue:"
echo "  1. V0.9 (600K, -12%) - Test K=3 vs K=5 impact"
echo "  2. V0.8 (520K, -23%) - Smaller capacity"
echo "  3. V0.6 (350K, -48%) - Minimal model"
echo ""
read -p "Press Enter to start training queue..."
echo ""

# 1. V0.9 - Ablation K=3 (600K params, -12%)
run_training "train_v0.9_ablation_k3.json" "V0.9 Ablation K=3" "600K"

# 2. V0.8 - Smaller (520K params, -23%)
run_training "train_v0.8_smaller.json" "V0.8 Smaller" "520K"

# 3. V0.6 - Minimal (350K params, -48%)
run_training "train_v0.6_minimal.json" "V0.6 Minimal" "350K"

echo "======================================================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "======================================================================"
echo ""
echo "Results will be in outputs/ directory"
echo "Run analysis script to compare all models"
echo ""
