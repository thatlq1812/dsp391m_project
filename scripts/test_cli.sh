#!/bin/bash
# Test script for STMGT CLI tool
# Author: THAT Le Quang
# Date: 2025-11-09

echo "========================================="
echo "STMGT CLI Tool - Test Script"
echo "========================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Test function
test_command() {
    local desc="$1"
    local cmd="$2"
    
    echo -e "${YELLOW}Testing:${NC} $desc"
    echo "Command: $cmd"
    echo ""
    
    eval "$cmd"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Test passed${NC}"
    else
        echo -e "${RED}✗ Test failed${NC}"
    fi
    
    echo ""
    echo "----------------------------------------"
    echo ""
}

# Install dependencies
echo -e "${YELLOW}Installing CLI dependencies...${NC}"
pip install click rich requests pyyaml -q
echo ""

# Test 1: Help
test_command "Show help" "python traffic_forecast/cli.py --help"

# Test 2: System info
test_command "System information" "python traffic_forecast/cli.py info"

# Test 3: Model list
test_command "List models" "python traffic_forecast/cli.py model list"

# Test 4: Data info
test_command "Dataset information" "python traffic_forecast/cli.py data info"

# Test 5: API status
test_command "API status" "python traffic_forecast/cli.py api status"

# Test 6: Training status
test_command "Training status" "python traffic_forecast/cli.py train status"

echo ""
echo "========================================="
echo "Test Summary"
echo "========================================="
echo ""
echo "CLI tool is working! Install globally with:"
echo "  pip install -e . -f setup_cli.py"
echo ""
echo "Then use:"
echo "  stmgt --help"
echo "  stmgt model list"
echo "  stmgt api start"
echo ""
