#!/bin/bash
# Quick restart script for STMGT API
# Usage: ./restart_api.sh

echo "Restarting STMGT Traffic API..."

# Kill existing API process
pkill -f "uvicorn traffic_api.main:app" 2>/dev/null || echo "No existing process found"
sleep 2

# Start API
echo "Starting API server..."
cd "$(dirname "$0")/../.."
python -m uvicorn traffic_api.main:app --host 0.0.0.0 --port 8080 --reload &

sleep 3

# Check if started
if curl -s http://localhost:8080/health > /dev/null 2>&1; then
    echo "✓ API started successfully on http://localhost:8080"
    echo "✓ Open http://localhost:8080 in your browser to view the traffic map"
else
    echo "✗ Failed to start API"
    exit 1
fi
