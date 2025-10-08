#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# Run collection loop with interval (seconds). Default 300s if not provided.
INTERVAL=${1:-300}
echo "Starting collection loop with interval=${INTERVAL}s"
python scripts/collect_and_render.py --interval ${INTERVAL}
