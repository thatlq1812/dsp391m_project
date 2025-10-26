#!/bin/bash
#
# Fix script: Ensure nodes.json exists before collection
# This fixes the "FileNotFoundError: Could not find nodes.json" issue
#

set -e

PROJECT_ROOT="$HOME/dsp391m_project"
NODES_SOURCE="$PROJECT_ROOT/data/node/20251025115239/collectors/overpass/nodes.json"
NODES_DEST="$PROJECT_ROOT/data/nodes.json"

echo "=== Fixing nodes.json issue ==="

# 1. Ensure nodes.json exists in data/
if [ ! -f "$NODES_DEST" ]; then
    if [ -f "$NODES_SOURCE" ]; then
        echo "✓ Copying nodes.json from successful run..."
        cp "$NODES_SOURCE" "$NODES_DEST"
        echo "✓ nodes.json copied to $NODES_DEST"
    else
        echo "✗ ERROR: Source nodes.json not found at $NODES_SOURCE"
        exit 1
    fi
else
    echo "✓ nodes.json already exists at $NODES_DEST"
fi

# 2. Clear broken cache
if [ -d "$PROJECT_ROOT/cache" ]; then
    echo "✓ Clearing cache to force fresh data..."
    rm -rf "$PROJECT_ROOT/cache/"*
    echo "✓ Cache cleared"
fi

# 3. Disable caching in config (set enabled: false)
CONFIG_FILE="$PROJECT_ROOT/configs/project_config.yaml"
if [ -f "$CONFIG_FILE" ]; then
    if grep -q "^  enabled: true" "$CONFIG_FILE"; then
        echo "✓ Disabling cache in config..."
        sed -i 's/^  enabled: true/  enabled: false/' "$CONFIG_FILE"
        echo "✓ Cache disabled"
    else
        echo "✓ Cache already disabled in config"
    fi
fi

echo "=== Fix completed ==="
echo ""
echo "Next steps:"
echo "1. Restart service: sudo systemctl restart traffic-collector"
echo "2. Monitor logs: tail -f $PROJECT_ROOT/logs/collector.log"
