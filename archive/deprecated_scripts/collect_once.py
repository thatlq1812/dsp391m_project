"""
Simple collection script for testing
Run all three collectors in sequence
Creates timestamped run directory
"""
import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

def run_collector(script_path, name):
    """Run a collector script"""
    print(f"\n{'='*60}")
    print(f"Running {name}...")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        print(result.stdout)
        if result.stderr:
            print(f"Warnings/Errors:\n{result.stderr}")
        
        if result.returncode != 0:
            print(f"FAILED with return code {result.returncode}")
            return False
        
        print(f"SUCCESS")
        return True
    
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT after 5 minutes")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def main():
    start_time = datetime.now()
    timestamp = start_time.strftime('%Y%m%d_%H%M%S')
    
    print(f"\n{'='*60}")
    print(f"TRAFFIC FORECAST DATA COLLECTION")
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    # Create run directory
    run_dir = PROJECT_ROOT / 'data' / 'runs' / f'run_{timestamp}'
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Set RUN_DIR environment variable for collectors
    os.environ['RUN_DIR'] = str(run_dir)
    
    print(f"\nRun directory: {run_dir}")
    print()
    
    # Define collectors
    collectors = [
        (PROJECT_ROOT / 'traffic_forecast' / 'collectors' / 'overpass' / 'collector.py', 'Topology (Overpass)'),
        (PROJECT_ROOT / 'traffic_forecast' / 'collectors' / 'open_meteo' / 'collector.py', 'Weather (Open-Meteo)'),
        (PROJECT_ROOT / 'traffic_forecast' / 'collectors' / 'google' / 'collector.py', 'Traffic (Google Directions)'),
    ]
    
    # Run collectors
    results = {}
    for script, name in collectors:
        success = run_collector(script, name)
        results[name] = success
        
        if not success:
            print(f"\n\nCollection FAILED at {name}")
            break
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n{'='*60}")
    print(f"COLLECTION SUMMARY")
    print(f"{'='*60}")
    print(f"Started:  {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration.total_seconds():.1f} seconds")
    print()
    
    for name, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"  {name}: {status}")
    
    all_success = all(results.values())
    print()
    if all_success:
        print("All collectors completed successfully!")
        print(f"\nRun directory: {run_dir}/")
        print(f"  - nodes.json (topology)")
        print(f"  - traffic_edges.json (traffic data)")
        print(f"  - weather_snapshot.json (weather data)")
        print(f"\nCache (shared):")
        print(f"  - cache/overpass_topology.json (permanent)")
    else:
        print("Some collectors FAILED. Check output above.")
        sys.exit(1)

if __name__ == '__main__':
    main()
