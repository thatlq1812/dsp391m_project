#!/bin/bash
# Run Traffic Forecasting API

echo "Starting STMGT Traffic Forecasting API..."

# Activate conda environment
source ~/.bashrc
conda activate dsp

# Run API
python -m uvicorn traffic_api.main:app --host 0.0.0.0 --port 8080 --reload
