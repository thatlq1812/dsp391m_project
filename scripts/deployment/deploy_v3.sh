#!/bin/bash
# STMGT V3 Deployment Automation Script
# Maintainer: THAT Le Quang
# Date: November 10, 2025

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}STMGT V3 Deployment Automation${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Step 1: Environment Check
echo -e "${YELLOW}[1/7] Checking environment...${NC}"
if ! command -v conda &> /dev/null; then
    echo -e "${RED}ERROR: Conda not found. Please install Anaconda/Miniconda.${NC}"
    exit 1
fi

if ! conda env list | grep -q "^dsp "; then
    echo -e "${RED}ERROR: Conda environment 'dsp' not found.${NC}"
    echo -e "${YELLOW}Run: conda env create -f environment.yml${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Environment OK${NC}"
echo ""

# Step 2: Model Validation
echo -e "${YELLOW}[2/7] Validating V3 model checkpoint...${NC}"
V3_CHECKPOINT="outputs/stmgt_v2_20251110_123931/best_model.pt"
if [ ! -f "$V3_CHECKPOINT" ]; then
    echo -e "${RED}ERROR: V3 checkpoint not found at $V3_CHECKPOINT${NC}"
    exit 1
fi

V3_CONFIG="outputs/stmgt_v2_20251110_123931/config.json"
if [ ! -f "$V3_CONFIG" ]; then
    echo -e "${RED}ERROR: V3 config not found at $V3_CONFIG${NC}"
    exit 1
fi

# Check model size (should be ~2.7MB for 680K params)
MODEL_SIZE=$(stat -f%z "$V3_CHECKPOINT" 2>/dev/null || stat -c%s "$V3_CHECKPOINT" 2>/dev/null)
if [ -z "$MODEL_SIZE" ]; then
    echo -e "${YELLOW}WARNING: Cannot determine model size${NC}"
else
    MODEL_SIZE_MB=$((MODEL_SIZE / 1024 / 1024))
    echo -e "${GREEN}✓ Model checkpoint found (${MODEL_SIZE_MB}MB)${NC}"
fi
echo ""

# Step 3: Data Validation
echo -e "${YELLOW}[3/7] Validating training data...${NC}"
DATA_PATH="data/processed/all_runs_extreme_augmented.parquet"
if [ ! -f "$DATA_PATH" ]; then
    echo -e "${RED}ERROR: Training data not found at $DATA_PATH${NC}"
    exit 1
fi

DATA_SIZE=$(stat -f%z "$DATA_PATH" 2>/dev/null || stat -c%s "$DATA_PATH" 2>/dev/null)
DATA_SIZE_MB=$((DATA_SIZE / 1024 / 1024))
echo -e "${GREEN}✓ Training data found (${DATA_SIZE_MB}MB)${NC}"
echo ""

# Step 4: Python Package Check
echo -e "${YELLOW}[4/7] Verifying Python packages...${NC}"
conda run -n dsp python -c "
import sys
required_packages = ['torch', 'torch_geometric', 'pandas', 'numpy', 'fastapi', 'uvicorn', 'streamlit']
missing = []
for pkg in required_packages:
    try:
        __import__(pkg.replace('-', '_').replace('_', ''))
    except ImportError:
        missing.append(pkg)
if missing:
    print(f'ERROR: Missing packages: {missing}')
    sys.exit(1)
print('✓ All required packages installed')
" || exit 1
echo ""

# Step 5: API Configuration Check
echo -e "${YELLOW}[5/7] Checking API configuration...${NC}"
conda run -n dsp python -c "
from traffic_api.config import config
print(f'Model: {config.model_checkpoint}')
print(f'Data: {config.data_path}')
print(f'Device: {config.device}')
print(f'Port: {config.port}')

# Verify V3 model is detected
if 'stmgt_v2_20251110_123931' not in str(config.model_checkpoint):
    print('WARNING: V3 model not auto-detected!')
    import sys
    sys.exit(1)
print('✓ V3 model correctly configured')
" || exit 1
echo ""

# Step 6: Start Services
echo -e "${YELLOW}[6/7] Starting services...${NC}"

# Check if API is already running
API_PID=$(pgrep -f "uvicorn traffic_api.main:app" || echo "")
if [ -n "$API_PID" ]; then
    echo -e "${YELLOW}API already running (PID: $API_PID). Restarting...${NC}"
    kill -9 $API_PID 2>/dev/null || true
    sleep 2
fi

# Start API in background
echo -e "${BLUE}Starting API server on port 8080...${NC}"
nohup conda run -n dsp --no-capture-output uvicorn traffic_api.main:app \
    --host 0.0.0.0 \
    --port 8080 \
    --log-level info \
    > api.log 2>&1 &
API_PID=$!
echo $API_PID > api.pid
echo -e "${GREEN}✓ API started (PID: $API_PID)${NC}"

# Wait for API to be ready
echo -e "${BLUE}Waiting for API to be ready...${NC}"
for i in {1..30}; do
    if curl -s http://localhost:8080/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ API is ready${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}ERROR: API failed to start within 30 seconds${NC}"
        echo -e "${YELLOW}Check api.log for details${NC}"
        exit 1
    fi
    sleep 1
done
echo ""

# Step 7: Health Check
echo -e "${YELLOW}[7/7] Running health checks...${NC}"

# API health check
API_HEALTH=$(curl -s http://localhost:8080/health)
echo -e "${BLUE}API Health Response:${NC}"
echo "$API_HEALTH" | python -m json.tool || echo "$API_HEALTH"

# Extract model info
MODEL_NAME=$(echo "$API_HEALTH" | grep -o '"model":"[^"]*"' | cut -d'"' -f4 || echo "unknown")
TEST_MAE=$(echo "$API_HEALTH" | grep -o '"test_mae":[0-9.]*' | cut -d':' -f2 || echo "unknown")

if [ "$MODEL_NAME" == "STMGT_V3" ]; then
    echo -e "${GREEN}✓ V3 model loaded successfully${NC}"
else
    echo -e "${RED}ERROR: Expected STMGT_V3, got $MODEL_NAME${NC}"
    exit 1
fi

if [ "$TEST_MAE" == "3.0468" ]; then
    echo -e "${GREEN}✓ Test MAE matches expected (3.0468)${NC}"
else
    echo -e "${YELLOW}WARNING: Test MAE is $TEST_MAE (expected 3.0468)${NC}"
fi
echo ""

# Final Summary
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Deployment Successful!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}Services:${NC}"
echo -e "  API:       http://localhost:8080"
echo -e "  Health:    http://localhost:8080/health"
echo -e "  Docs:      http://localhost:8080/docs"
echo ""
echo -e "${BLUE}Logs:${NC}"
echo -e "  API:       tail -f api.log"
echo ""
echo -e "${BLUE}Stop Services:${NC}"
echo -e "  API:       kill \$(cat api.pid)"
echo ""
echo -e "${BLUE}Quick Test:${NC}"
echo -e '  curl http://localhost:8080/health'
echo ""
echo -e "${YELLOW}Dashboard (optional):${NC}"
echo -e "  conda run -n dsp streamlit run dashboard/Dashboard.py"
echo ""
