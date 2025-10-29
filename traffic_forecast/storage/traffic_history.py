"""
Traffic History Store - Persistent storage for traffic data.

This store manages historical traffic data for lag feature computation:
- Stores traffic snapshots with timestamp indexing
- Retrieves historical data for specific time ranges
- Automatically manages data retention (configurable)
- Efficient lookups for recent data (last 60-120 minutes)

Storage format: SQLite database for efficient time-based queries

Usage:
 store = TrafficHistoryStore('data/traffic_history.db')

 # Store current snapshot
 store.save_snapshot(timestamp, traffic_data)

 # Retrieve historical data (for lag features)
 lag_5min = store.get_snapshot(timestamp - timedelta(minutes=5))
 lag_15min = store.get_snapshot(timestamp - timedelta(minutes=15))

 # Get time range
 history = store.get_range(start_time, end_time)
"""

import sqlite3
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd

logger = logging.getLogger(__name__)


class TrafficHistoryStore:
    """
    Persistent storage for traffic history data.

    Optimized for:
    - Fast recent data retrieval (for lag features)
    - Time-based indexing
    - Automatic cleanup of old data
    """

    def __init__(
        self,
        db_path: str = 'data/traffic_history.db',
        retention_days: int = 7
    ):
    """
 Initialize traffic history store.

 Args:
 db_path: Path to SQLite database
 retention_days: Number of days to keep historical data
 """
    self.db_path = Path(db_path)
    self.db_path.parent.mkdir(parents=True, exist_ok=True)
    self.retention_days = retention_days

    self._init_database()
    logger.info(f"Initialized TrafficHistoryStore at {db_path}")

    def _init_database(self):
    """Create database schema if not exists."""
    with sqlite3.connect(self.db_path) as conn:
    conn.execute("""
 CREATE TABLE IF NOT EXISTS traffic_snapshots (
 timestamp TEXT NOT NULL,
 node_id TEXT NOT NULL,
 avg_speed_kmh REAL,
 congestion_level INTEGER,
 temperature_c REAL,
 rain_mm REAL,
 wind_speed_kmh REAL,
 data_json TEXT NOT NULL,
 PRIMARY KEY (timestamp, node_id)
 )
 """)

    # Create index for fast time-based queries
    conn.execute("""
 CREATE INDEX IF NOT EXISTS idx_timestamp
 ON traffic_snapshots(timestamp DESC)
 """)

    conn.execute("""
 CREATE INDEX IF NOT EXISTS idx_node_time
 ON traffic_snapshots(node_id, timestamp DESC)
 """)

    conn.commit()

    def save_snapshot(
        self,
        timestamp: datetime,
        traffic_data: List[Dict]
    ) -> int:
    """
 Save traffic snapshot to database.

 Args:
 timestamp: Snapshot timestamp
 traffic_data: List of node traffic data dicts

 Returns:
 Number of records saved
 """
    with sqlite3.connect(self.db_path) as conn:
    records_saved = 0

    for node_data in traffic_data:
    try:
    conn.execute("""
 INSERT OR REPLACE INTO traffic_snapshots
 (timestamp, node_id, avg_speed_kmh, congestion_level,
 temperature_c, rain_mm, wind_speed_kmh, data_json)
 VALUES (?, ?, ?, ?, ?, ?, ?, ?)
 """, (
        timestamp.isoformat(),
        node_data.get('node_id'),
        node_data.get('avg_speed_kmh'),
        node_data.get('congestion_level'),
        node_data.get('temperature_c'),
        node_data.get('rain_mm'),
        node_data.get('wind_speed_kmh'),
        json.dumps(node_data)
    ))
    records_saved += 1
    except Exception as e:
    logger.warning(f"Failed to save node {node_data.get('node_id')}: {e}")

    conn.commit()

    logger.info(f"Saved {records_saved} traffic records at {timestamp}")
    return records_saved

    def get_snapshot(
        self,
        timestamp: datetime,
        tolerance_minutes: int = 3
    ) -> Optional[pd.DataFrame]:
    """
 Get traffic snapshot at specific time.

 Args:
 timestamp: Target timestamp
 tolerance_minutes: Accept data within ±N minutes

 Returns:
 DataFrame with traffic data or None if not found
 """
    start_time = timestamp - timedelta(minutes=tolerance_minutes)
    end_time = timestamp + timedelta(minutes=tolerance_minutes)

    with sqlite3.connect(self.db_path) as conn:
    query = """
 SELECT data_json
 FROM traffic_snapshots
 WHERE timestamp BETWEEN ? AND ?
 ORDER BY ABS(julianday(timestamp) - julianday(?))
 LIMIT 1000
 """

    cursor = conn.execute(
        query,
        (start_time.isoformat(), end_time.isoformat(), timestamp.isoformat())
    )

    rows = cursor.fetchall()

    if not rows:
    logger.warning(f"No data found near {timestamp}")
    return None

    # Parse JSON data
    data = [json.loads(row[0]) for row in rows]
    df = pd.DataFrame(data)

    logger.debug(f"Retrieved {len(df)} records near {timestamp}")
    return df

    def get_lag_data(
        self,
        current_time: datetime,
        lag_minutes: List[int] = [5, 15, 30, 60]
    ) -> Dict[int, pd.DataFrame]:
    """
 Get historical data for lag feature computation.

 Args:
 current_time: Current timestamp
 lag_minutes: List of lag intervals in minutes

 Returns:
 Dict mapping lag_minutes -> DataFrame

 Example:
 lags = store.get_lag_data(now, [5, 15, 30, 60])
 # lags[5] = data from 5 minutes ago
 # lags[15] = data from 15 minutes ago
 """
    result = {}

    for lag in lag_minutes:
    target_time = current_time - timedelta(minutes=lag)
    df = self.get_snapshot(target_time, tolerance_minutes=2)
    if df is not None:
    result[lag] = df
    else:
    logger.warning(f"No data found for lag {lag} minutes")

    return result

    def get_range(
        self,
        start_time: datetime,
        end_time: datetime,
        node_ids: Optional[List[int]] = None
    ) -> pd.DataFrame:
    """
 Get traffic data for time range.

 Args:
 start_time: Start of time range
 end_time: End of time range
 node_ids: Optional list of node IDs to filter

 Returns:
 DataFrame with all traffic data in range
 """
    with sqlite3.connect(self.db_path) as conn:
    if node_ids:
    placeholders = ','.join('?' * len(node_ids))
    query = f"""
 SELECT data_json
 FROM traffic_snapshots
 WHERE timestamp BETWEEN ? AND ?
 AND node_id IN ({placeholders})
 ORDER BY timestamp ASC
 """
    params = [start_time.isoformat(), end_time.isoformat()] + node_ids
    else:
    query = """
 SELECT data_json
 FROM traffic_snapshots
 WHERE timestamp BETWEEN ? AND ?
 ORDER BY timestamp ASC
 """
    params = [start_time.isoformat(), end_time.isoformat()]

    cursor = conn.execute(query, params)
    rows = cursor.fetchall()

    if not rows:
    logger.warning(f"No data found between {start_time} and {end_time}")
    return pd.DataFrame()

    data = [json.loads(row[0]) for row in rows]
    df = pd.DataFrame(data)

    logger.info(f"Retrieved {len(df)} records from {start_time} to {end_time}")
    return df

    def get_recent_history(
        self,
        current_time: datetime,
        hours: int = 2,
        node_ids: Optional[List[int]] = None
    ) -> pd.DataFrame:
    """
 Get recent history (for rolling features).

 Args:
 current_time: Current timestamp
 hours: Number of hours to look back
 node_ids: Optional node filter

 Returns:
 DataFrame with recent traffic history
 """
    start_time = current_time - timedelta(hours=hours)
    return self.get_range(start_time, current_time, node_ids)

    def cleanup_old_data(self, older_than_days: Optional[int] = None):
    """
 Remove old data beyond retention period.

 Args:
 older_than_days: Days to keep (default: use retention_days)
 """
    days = older_than_days or self.retention_days
    cutoff = datetime.now() - timedelta(days=days)

    with sqlite3.connect(self.db_path) as conn:
    cursor = conn.execute("""
 DELETE FROM traffic_snapshots
 WHERE timestamp < ?
 """, (cutoff.isoformat(),))

    deleted = cursor.rowcount
    conn.commit()

    logger.info(f"Cleaned up {deleted} records older than {days} days")
    return deleted

    def get_stats(self) -> Dict:
    """Get storage statistics."""
    with sqlite3.connect(self.db_path) as conn:
        # Total records
    cursor = conn.execute("SELECT COUNT(*) FROM traffic_snapshots")
    total_records = cursor.fetchone()[0]

    # Unique timestamps
    cursor = conn.execute("SELECT COUNT(DISTINCT timestamp) FROM traffic_snapshots")
    unique_timestamps = cursor.fetchone()[0]

    # Unique nodes
    cursor = conn.execute("SELECT COUNT(DISTINCT node_id) FROM traffic_snapshots")
    unique_nodes = cursor.fetchone()[0]

    # Time range
    cursor = conn.execute("""
 SELECT MIN(timestamp), MAX(timestamp)
 FROM traffic_snapshots
 """)
    min_time, max_time = cursor.fetchone()

    # Database size
    db_size_mb = self.db_path.stat().st_size / (1024 * 1024)

    return {
        'total_records': total_records,
        'unique_timestamps': unique_timestamps,
        'unique_nodes': unique_nodes,
        'earliest_timestamp': min_time,
        'latest_timestamp': max_time,
        'database_size_mb': round(db_size_mb, 2),
        'retention_days': self.retention_days
    }

    def close(self):
    """Close database connection (if needed)."""
    # SQLite connections are created per-operation
    pass
