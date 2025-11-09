#!/bin/bash
# STMGT CLI wrapper for Git Bash on Windows
# Usage: ./stmgt.sh [command] [args]
# Or add to PATH: export PATH="$PATH:/d/UNI/DSP391m/project"

set -e  # Exit on error

# Configuration
CONDA_PATH="C:/ProgramData/miniconda3/Scripts/conda.exe"
CONDA_ENV="dsp"
PROJECT_ROOT="/d/UNI/DSP391m/project"
CLI_SCRIPT="traffic_forecast/cli.py"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Validate conda exists
if [ ! -f "$CONDA_PATH" ]; then
    echo -e "${RED}Error: Conda not found at $CONDA_PATH${NC}" >&2
    echo "Please update CONDA_PATH in this script" >&2
    exit 1
fi

# Validate project root
if [ ! -d "$PROJECT_ROOT" ]; then
    echo -e "${RED}Error: Project root not found at $PROJECT_ROOT${NC}" >&2
    exit 1
fi

# Validate CLI script
if [ ! -f "$PROJECT_ROOT/$CLI_SCRIPT" ]; then
    echo -e "${RED}Error: CLI script not found at $PROJECT_ROOT/$CLI_SCRIPT${NC}" >&2
    exit 1
fi

# Change to project directory
cd "$PROJECT_ROOT" || {
    echo -e "${RED}Error: Failed to change directory to $PROJECT_ROOT${NC}" >&2
    exit 1
}

# Run CLI with conda
"$CONDA_PATH" run -n "$CONDA_ENV" python "$CLI_SCRIPT" "$@"
exit_code=$?

# Handle exit codes
if [ $exit_code -ne 0 ]; then
    echo -e "${YELLOW}Command exited with code $exit_code${NC}" >&2
fi

exit $exit_code
