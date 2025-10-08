"""
Scheduler for periodic collectors.
"""

import os
import subprocess
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger
import yaml

def load_config():
    with open('configs/project_config.yaml', 'r') as f:
        return yaml.safe_load(f)

def run_collector(script_path):
    """Run a collector script."""
    result = subprocess.run(['python', script_path], capture_output=True, text=True)
    print(f"Ran {script_path}: {result.returncode}")
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)

def schedule_collectors():
    config = load_config()
    scheduler_config = config['scheduler']
    
    scheduler = BlockingScheduler()
    
    # Run weather nowcast
    scheduler.add_job(run_collector, args=['collectors/open_meteo/collector.py'], 
                      trigger=IntervalTrigger(minutes=scheduler_config['weather_nowcast_interval_min']))
    
    # Run weather forecast (new)
    scheduler.add_job(run_collector, args=['collectors/open_meteo/forecast.py'], 
                      trigger=IntervalTrigger(minutes=scheduler_config['weather_forecast_interval_min']))
    
    # Run traffic
    scheduler.add_job(run_collector, args=['collectors/google/collector.py'], 
                      trigger=IntervalTrigger(minutes=scheduler_config['traffic_interval_min']))
    
    # Run normalize
    scheduler.add_job(run_collector, args=['pipelines/normalize/normalize.py'], 
                      trigger=IntervalTrigger(minutes=scheduler_config['normalize_interval_min']))
    
    # Run build features v2
    scheduler.add_job(run_collector, args=['pipelines/features/build_features.py'], 
                      trigger=IntervalTrigger(minutes=scheduler_config['features_interval_min']))
    
    # Run infer batch
    scheduler.add_job(run_collector, args=['pipelines/model/infer.py'], 
                      trigger=IntervalTrigger(minutes=scheduler_config['infer_interval_min']))
    
    print("Scheduler started. Press Ctrl+C to exit.")
    scheduler.start()

if __name__ == "__main__":
    schedule_collectors()