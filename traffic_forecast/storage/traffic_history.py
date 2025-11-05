"""SQLite-backed traffic history storage used in integration tests."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class TrafficHistory:
    """Persist and retrieve traffic measurements."""

    db_path: Path

    def __post_init__(self) -> None:
        self.db_path = Path(self.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialise()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _initialise(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS traffic_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    node_a_id TEXT NOT NULL,
                    node_b_id TEXT NOT NULL,
                    speed_kmh REAL NOT NULL,
                    duration_sec REAL NOT NULL,
                    distance_km REAL NOT NULL
                )
                """
            )
            conn.commit()

    def save(self, timestamp: datetime, frame: pd.DataFrame) -> None:
        if frame.empty:
            return

        data = frame.copy()
        data["timestamp"] = pd.to_datetime(timestamp).isoformat()

        columns = [
            "timestamp",
            "node_a_id",
            "node_b_id",
            "speed_kmh",
            "duration_sec",
            "distance_km",
        ]
        missing = [col for col in columns if col not in data.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        with self._connect() as conn:
            data[columns].to_sql("traffic_history", conn, if_exists="append", index=False)
            conn.commit()

    def get_latest(self, limit: int = 10) -> Optional[pd.DataFrame]:
        with self._connect() as conn:
            result = pd.read_sql_query(
                "SELECT timestamp, node_a_id, node_b_id, speed_kmh, duration_sec, distance_km "
                "FROM traffic_history ORDER BY timestamp DESC LIMIT ?",
                conn,
                params=(limit,),
            )
        if result.empty:
            return None
        result["timestamp"] = pd.to_datetime(result["timestamp"])
        return result
