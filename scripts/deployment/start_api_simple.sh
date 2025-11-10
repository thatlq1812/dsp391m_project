#!/bin/bash
# Quick Start API for Testing
# Usage: ./start_api_simple.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "Starting STMGT V3 API..."
echo "Project root: $PROJECT_ROOT"
echo ""

# Check if API is already running
if curl -s http://localhost:8080/health > /dev/null 2>&1; then
    echo "API is already running!"
    curl -s http://localhost:8080/health | python -m json.tool
    exit 0
fi

# Start API
echo "Starting API server on port 8080..."
echo "(Press Ctrl+C to stop)"
echo ""

# Use Python directly if conda is not available in PATH
python -c "
import sys
import os
sys.path.insert(0, '.')
os.chdir('$PROJECT_ROOT')

# Start uvicorn
import uvicorn
from traffic_api.main import app

uvicorn.run(app, host='0.0.0.0', port=8080, log_level='info')
"
