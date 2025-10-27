#!/usr/bin/env python3
"""
Run all collectors sequentially to produce a complete data snapshot.
"""

import argparse
import json
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from traffic_forecast import PROJECT_ROOT

PYTHON = sys.executable
COLLECTOR_MODULES = [
 "traffic_forecast.collectors.overpass.collector",
 "traffic_forecast.collectors.open_meteo.collector",
 "traffic_forecast.collectors.google.collector",
]


def run_cmd(args: Iterable[str], env: Optional[dict] = None) -> bool:
 args = list(args)
 print(f"$ {' '.join(args)}")
 proc_env = os.environ.copy()
 if env:
 proc_env.update(env)
 try:
 result = subprocess.run(
 args,
 check=True,
 capture_output=True,
 text=True,
 env=proc_env,
 )
 except subprocess.CalledProcessError as exc:
 print(f"Command failed: {exc}")
 if exc.stdout:
 print(exc.stdout)
 if exc.stderr:
 print(exc.stderr, file=sys.stderr)
 return False

 if result.stdout:
 print(result.stdout)
 if result.stderr:
 print(result.stderr, file=sys.stderr)
 return True


def run_collectors(run_dir: Path, bbox: Optional[str] = None) -> None:
 run_env = {"RUN_DIR": str(run_dir)}
 for module in COLLECTOR_MODULES:
 cmd = [PYTHON, "-m", module]
 if bbox:
 cmd.extend(["--bbox", bbox])
 if not run_cmd(cmd, env=run_env):
 print(f"Failed to run collector module {module}")

 generate_mock_traffic(run_dir)


def generate_mock_traffic(run_dir: Path) -> None:
 """Create a mock traffic snapshot when the traffic collector has no output."""
 traffic_path = run_dir / "collectors" / "mock" / "traffic_snapshot_normalized.json"
 if traffic_path.exists():
 return

 nodes_path = run_dir / "collectors" / "overpass" / "nodes.json"
 if not nodes_path.exists():
 return

 with nodes_path.open(encoding="utf-8") as fh:
 nodes = json.load(fh)

 traffic_data = []
 for node in nodes[:1000]:
 traffic_data.append(
 {
 "node_id": node["node_id"],
 "timestamp": datetime.now().isoformat(),
 "avg_speed_kmh": round(random.uniform(10, 55), 1),
 "vehicle_count": random.randint(10, 300),
 }
 )

 traffic_path.parent.mkdir(parents=True, exist_ok=True)
 with traffic_path.open("w", encoding="utf-8") as fh:
 json.dump(traffic_data, fh, indent=2)

 print(f"Generated mock traffic for {len(traffic_data)} nodes")


def main() -> None:
 parser = argparse.ArgumentParser()
 parser.add_argument("--run-dir", help="Run directory to store collector outputs")
 parser.add_argument("--bbox", help="Optional bbox as min_lat,min_lon,max_lat,max_lon to filter collectors")
 args = parser.parse_args()

 run_dir = Path(args.run_dir or os.getenv("RUN_DIR") or PROJECT_ROOT / "data_runs" / "latest")
 run_dir.mkdir(parents=True, exist_ok=True)
 run_collectors(run_dir, bbox=args.bbox)


if __name__ == "__main__":
 main()
