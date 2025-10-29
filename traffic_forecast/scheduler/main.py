"""
Scheduler for periodic collectors.
"""

import subprocess
import sys
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger
import yaml
from traffic_forecast import PROJECT_ROOT

CONFIG_PATH = PROJECT_ROOT / "configs" / "project_config.yaml"
PYTHON = sys.executable


def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as fh:
    return yaml.safe_load(fh) or {}


def run_module(module_path: str) -> None:
    """Execute a Python module using `python -m` to respect package imports."""
    result = subprocess.run(
        [PYTHON, "-m", module_path],
        capture_output=True,
        text=True,
        check=False,
    )
    print(f"Ran {module_path}: {result.returncode}")
    if result.stdout:
    print(result.stdout)
    if result.stderr:
    print(result.stderr, file=sys.stderr)


def register_jobs(scheduler: BlockingScheduler, config: dict) -> None:
    scheduler_config = config.get("scheduler", {})
    collectors_config = [
        ("traffic_forecast.collectors.open_meteo.collector", scheduler_config.get("weather_nowcast_interval_min")),
        ("traffic_forecast.collectors.open_meteo.forecast", scheduler_config.get("weather_forecast_interval_min")),
        ("traffic_forecast.collectors.google.collector", scheduler_config.get("traffic_interval_min")),
        ("traffic_forecast.pipelines.normalize.normalize", scheduler_config.get("normalize_interval_min")),
        ("traffic_forecast.pipelines.features.build_features", scheduler_config.get("features_interval_min")),
        ("traffic_forecast.pipelines.model.infer", scheduler_config.get("infer_interval_min")),
    ]
    for module_path, interval in collectors_config:
    if not interval:
    continue
    scheduler.add_job(
        run_module,
        trigger=IntervalTrigger(minutes=interval),
        args=[module_path],
    )


def schedule_collectors() -> None:
    config = load_config()
    scheduler = BlockingScheduler()
    register_jobs(scheduler, config)
    print("Scheduler started. Press Ctrl+C to exit.")
    scheduler.start()


if __name__ == "__main__":
    schedule_collectors()
