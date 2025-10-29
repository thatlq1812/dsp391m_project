#!/bin/bash
# Serve Data Publicly via Simple HTTP Server
# Run this on the VM to allow team members to download data without gcloud

PORT=8080
DATA_DIR="$HOME/traffic-forecast/data/runs"

echo "======================================================================"
echo "  PUBLIC DATA SERVER"
echo "======================================================================"
echo ""
echo "Starting HTTP server on port $PORT..."
echo "Data directory: $DATA_DIR"
echo ""
echo "Team members can download data using:"
echo "  curl http://$(curl -s ifconfig.me):$PORT/runs.json"
echo ""
echo "Press Ctrl+C to stop"
echo "======================================================================"
echo ""

cd "$DATA_DIR/.."

# Generate runs list JSON
echo "Generating runs list..."
python3 << 'EOF'
import json
import os
from pathlib import Path

runs_dir = Path("runs")
runs = sorted([d.name for d in runs_dir.iterdir() if d.is_dir()], reverse=True)

with open("runs/runs.json", "w") as f:
    json.dump({"runs": runs[:50]}, f, indent=2)  # Latest 50 runs

print(f"Generated runs.json with {len(runs[:50])} runs")
EOF

# Start HTTP server
echo ""
echo "Starting server..."
python3 -m http.server $PORT
