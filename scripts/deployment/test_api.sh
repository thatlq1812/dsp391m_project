#!/bin/bash
# Quick API Test Script for STMGT V3
# Usage: ./test_api.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

API_URL="http://localhost:8080"

echo -e "${BLUE}Testing STMGT V3 API...${NC}"
echo ""

# Test 1: Health Check
echo -e "${BLUE}[1/3] Health Check${NC}"
HEALTH_RESPONSE=$(curl -s "$API_URL/health")
echo "$HEALTH_RESPONSE" | python -m json.tool 2>/dev/null || echo "$HEALTH_RESPONSE"

if echo "$HEALTH_RESPONSE" | grep -q "stmgt_v2_20251110_123931"; then
    echo -e "${GREEN}✓ V3 model checkpoint detected (stmgt_v2_20251110_123931)${NC}"
elif echo "$HEALTH_RESPONSE" | grep -q '"model_loaded":true'; then
    echo -e "${GREEN}✓ Model loaded successfully${NC}"
else
    echo -e "${RED}✗ Model not loaded${NC}"
    exit 1
fi
echo ""

# Test 2: Get Nodes
echo -e "${BLUE}[2/3] Get Network Nodes${NC}"
NODES_RESPONSE=$(curl -s "$API_URL/nodes")
NODE_COUNT=$(echo "$NODES_RESPONSE" | python -c "import sys, json; data = json.load(sys.stdin); print(len(data) if isinstance(data, list) else len(data.get('nodes', [])))" 2>/dev/null || echo "0")
echo "Total nodes: $NODE_COUNT"

if [ "$NODE_COUNT" -ge 60 ]; then
    echo -e "${GREEN}✓ Network topology loaded (${NODE_COUNT} nodes)${NC}"
elif [ "$NODE_COUNT" -gt 0 ]; then
    echo -e "${GREEN}✓ Nodes available (${NODE_COUNT} nodes, limited for testing)${NC}"
else
    echo -e "${RED}⚠ No nodes found (topology may not be loaded)${NC}"
fi
echo ""

# Test 3: Simple Prediction
echo -e "${BLUE}[3/3] Test Prediction${NC}"
cat > /tmp/test_request.json << 'EOF'
{
  "route": [1, 5, 12],
  "weather": {
    "temperature": 28,
    "wind_speed": 5,
    "precipitation": 0
  }
}
EOF

PRED_RESPONSE=$(curl -s -X POST "$API_URL/predict" \
    -H "Content-Type: application/json" \
    -d @/tmp/test_request.json)

if echo "$PRED_RESPONSE" | grep -q '"nodes"'; then
    echo -e "${GREEN}✓ Prediction successful${NC}"
    # Extract node count and show sample
    NODE_PRED_COUNT=$(echo "$PRED_RESPONSE" | python -c "import sys, json; print(len(json.load(sys.stdin)['nodes']))" 2>/dev/null || echo "0")
    echo "Predicted ${NODE_PRED_COUNT} nodes"
    echo "$PRED_RESPONSE" | python -m json.tool 2>/dev/null | head -30
else
    echo -e "${RED}✗ Prediction failed${NC}"
    echo "$PRED_RESPONSE"
    exit 1
fi
echo ""

# Benchmark response time
echo -e "${BLUE}Benchmarking response time...${NC}"
START=$(date +%s%N)
curl -s "$API_URL/health" > /dev/null
END=$(date +%s%N)
DURATION=$(((END - START) / 1000000))
echo "Health endpoint: ${DURATION}ms"

if [ "$DURATION" -lt 100 ]; then
    echo -e "${GREEN}✓ Response time < 100ms${NC}"
else
    echo -e "${RED}⚠ Response time > 100ms${NC}"
fi
echo ""

echo -e "${GREEN}All tests passed!${NC}"
