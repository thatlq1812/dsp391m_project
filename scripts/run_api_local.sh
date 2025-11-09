#!/bin/bash
# Script to run STMGT Traffic API locally
# Author: THAT Le Quang (thatlq1812)
# Date: 2025-11-09

set -e  # Exit on error

echo "======================================"
echo "STMGT Traffic API - Local Server"
echo "======================================"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if in project root
if [ ! -f "traffic_api/main.py" ]; then
    echo -e "${RED}Error: Must run from project root directory${NC}"
    echo "Usage: ./scripts/run_api_local.sh"
    exit 1
fi

# Check Python environment
echo -e "${YELLOW}[1/4] Checking Python environment...${NC}"
if command -v conda &> /dev/null; then
    echo "  Conda environment: dsp"
    PYTHON_CMD="conda run -n dsp python"
else
    echo "  Using system Python"
    PYTHON_CMD="python"
fi

# Check dependencies
echo -e "${YELLOW}[2/4] Checking dependencies...${NC}"
$PYTHON_CMD -c "import fastapi, uvicorn, torch, networkx" 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "  ${GREEN}✓ All dependencies installed${NC}"
else
    echo -e "  ${RED}✗ Missing dependencies. Installing...${NC}"
    $PYTHON_CMD -m pip install fastapi uvicorn torch networkx httpx -q
fi

# Check model checkpoint
echo -e "${YELLOW}[3/4] Checking model checkpoint...${NC}"
if [ -f "outputs/stmgt_v2_20251102_200308/best_model.pt" ]; then
    echo -e "  ${GREEN}✓ Model checkpoint found${NC}"
else
    echo -e "  ${RED}✗ Model checkpoint not found${NC}"
    echo "  Looking in: outputs/stmgt_v2_20251102_200308/best_model.pt"
    echo "  Please train a model first or update config.py"
    exit 1
fi

# Check data
echo -e "${YELLOW}[4/4] Checking data...${NC}"
if [ -f "data/processed/all_runs_extreme_augmented.parquet" ]; then
    echo -e "  ${GREEN}✓ Data file found${NC}"
else
    echo -e "  ${RED}✗ Data file not found${NC}"
    echo "  Looking in: data/processed/all_runs_extreme_augmented.parquet"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ All checks passed!${NC}"
echo ""
echo "======================================"
echo "Starting FastAPI server..."
echo "======================================"
echo ""
echo "Server URLs:"
echo "  - API:          http://localhost:8000"
echo "  - Docs:         http://localhost:8000/docs"
echo "  - Web UI:       http://localhost:8000/route_planner.html"
echo "  - Health:       http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Run server
cd "$(dirname "$0")/.." || exit 1
$PYTHON_CMD -m uvicorn traffic_api.main:app --reload --host 0.0.0.0 --port 8000
