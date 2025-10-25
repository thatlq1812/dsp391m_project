#!/bin/bash
# Health Check Script for Traffic Forecast System
# Checks system health and reports status

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

PASS="${GREEN}[PASS]${NC}"
WARN="${YELLOW}[WARN]${NC}"
FAIL="${RED}[FAIL]${NC}"

echo "========================================="
echo "Traffic Forecast - Health Check"
echo "========================================="
echo ""

# Check 1: Python environment
echo -n "Checking Python environment... "
if conda env list | grep -q "^dsp "; then
    echo -e "$PASS"
else
    echo -e "$FAIL"
    echo "  Environment 'dsp' not found"
fi

# Check 2: Required files
echo -n "Checking configuration files... "
if [ -f "configs/project_config.yaml" ]; then
    echo -e "$PASS"
else
    echo -e "$FAIL"
    echo "  configs/project_config.yaml missing"
fi

# Check 3: Data directories
echo -n "Checking data directories... "
if [ -d "data/node" ] && [ -d "data/processed" ]; then
    echo -e "$PASS"
else
    echo -e "$FAIL"
    echo "  Required directories missing"
fi

# Check 4: Service status (if systemd available)
if command -v systemctl &> /dev/null; then
    echo -n "Checking service status... "
    if systemctl is-active --quiet traffic-forecast.service; then
        echo -e "$PASS"
    else
        echo -e "$WARN"
        echo "  Service not running"
    fi
fi

# Check 5: Disk space
echo -n "Checking disk space... "
DISK_USAGE=$(df -h . | tail -1 | awk '{print $5}' | sed 's/%//')
if [ "$DISK_USAGE" -lt 80 ]; then
    echo -e "$PASS (${DISK_USAGE}% used)"
else
    echo -e "$WARN (${DISK_USAGE}% used)"
    echo "  Consider cleanup"
fi

# Check 6: Database
echo -n "Checking database... "
if [ -f "traffic_history.db" ]; then
    DB_SIZE=$(du -h traffic_history.db | cut -f1)
    echo -e "$PASS (${DB_SIZE})"
else
    echo -e "$WARN"
    echo "  Database not initialized"
fi

# Check 7: Recent data collection
echo -n "Checking recent collections... "
if [ -d "data/node" ]; then
    LATEST=$(ls -t data/node | head -1)
    if [ -n "$LATEST" ]; then
        echo -e "$PASS (latest: $LATEST)"
    else
        echo -e "$WARN"
        echo "  No collections found"
    fi
fi

# Check 8: Logs
echo -n "Checking logs... "
if [ -d "logs" ]; then
    LOG_SIZE=$(du -sh logs | cut -f1)
    echo -e "$PASS (${LOG_SIZE})"
else
    echo -e "$WARN"
    echo "  Logs directory missing"
fi

echo ""
echo "========================================="
echo "Health check complete"
echo "========================================="
