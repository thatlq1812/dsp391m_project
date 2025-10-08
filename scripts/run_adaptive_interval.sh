#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Adaptive collection interval based on time of day
# Peak hours (6-9 AM, 4-7 PM): 15 minutes (900s)
# Off-peak hours: 60 minutes (3600s)

get_adaptive_interval() {
    # Get current hour in Vietnam timezone (UTC+7)
    HOUR=$(TZ=Asia/Ho_Chi_Minh date +%H | sed 's/^0//')  # Remove leading zero

    # Peak hours: 6-9 AM (6-9) and 4-7 PM (16-19)
    if [[ ($HOUR -ge 6 && $HOUR -le 9) || ($HOUR -ge 16 && $HOUR -le 19) ]]; then
        echo 900   # 15 minutes during peak hours
    else
        echo 3600  # 60 minutes during off-peak hours
    fi
}

INTERVAL=$(get_adaptive_interval)
HOUR=$(TZ=Asia/Ho_Chi_Minh date +%H:%M)
echo "Starting adaptive collection loop at ${HOUR} (interval=${INTERVAL}s)"

# Run indefinitely with adaptive intervals
while true; do
    START_TIME=$(date +%s)

    echo "$(date '+%Y-%m-%d %H:%M:%S') - Running collection cycle..."
    if python scripts/collect_and_render.py --once --no-visualize; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Collection completed successfully"
    else
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Collection failed, will retry in next cycle"
    fi

    # Calculate sleep time (accounting for execution time)
    END_TIME=$(date +%s)
    EXECUTION_TIME=$((END_TIME - START_TIME))
    SLEEP_TIME=$((INTERVAL - EXECUTION_TIME))

    if [ $SLEEP_TIME -gt 0 ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Sleeping for ${SLEEP_TIME}s until next cycle..."
        sleep $SLEEP_TIME
    else
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Execution took longer than interval, starting next cycle immediately"
    fi

    # Recalculate interval for next cycle (in case we crossed time boundaries)
    INTERVAL=$(get_adaptive_interval)
done