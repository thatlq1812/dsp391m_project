#!/usr/bin/env python3
"""
Helper: run collectors then visualization with adaptive scheduling.

Usage examples:
  # One-time run
  python scripts/collect_and_render.py --once

  # Run with adaptive scheduling (uses config from project_config.yaml)
  python scripts/collect_and_render.py --adaptive

  # Run with fixed interval (legacy mode)
  python scripts/collect_and_render.py --interval 600

Adaptive Mode (NEW in v4.0):
  Automatically adjusts collection frequency based on time of day:
  - Peak hours (7-9 AM, 12-1 PM, 5-7 PM): 30 min intervals
  - Off-peak hours: 60 min intervals
  - Weekends: 90 min intervals
  
  This reduces API costs by ~85% vs fixed 15-minute intervals.

This script runs the packaged collectors and visualization modules.
"""
import argparse
import subprocess
import time
import sys
import os
import datetime
import pathlib
import yaml

# bootstrap project root so scripts can be run directly (avoids ModuleNotFoundError)
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run_cmd(cmd, cwd=None):
    print(f"$ {' '.join(cmd)}")
    try:
        r = subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)
        if r.stdout:
            print(r.stdout)
        if r.stderr:
            print(r.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}; stderr:\n{e.stderr}")
        return False


def run_collectors(args):
    cmd = [sys.executable, '-m', 'traffic_forecast.cli.run_collectors']
    if args.bbox:
        cmd.extend(['--bbox', args.bbox])
    return run_cmd(cmd)


def run_visualization():
    cmd = [sys.executable, '-m', 'traffic_forecast.cli.visualize']
    return run_cmd(cmd)


def main():
    parser = argparse.ArgumentParser(description='Run collectors then visualization (once or periodically)')
    parser.add_argument('--once', action='store_true', help='Run once and exit')
    parser.add_argument('--interval', type=int, default=0, help='Loop interval in seconds (0 disables looping)')
    parser.add_argument('--adaptive', action='store_true', help='Use adaptive scheduling from config (v4.0)')
    parser.add_argument('--bbox', help='Optional bbox to pass to collectors (min_lat,min_lon,max_lat,max_lon)')
    parser.add_argument('--no-visualize', action='store_true', help='Skip visualization step')
    parser.add_argument('--print-schedule', action='store_true', help='Print schedule info and exit')
    args = parser.parse_args()

    # Load config for adaptive scheduling
    config_path = ROOT / 'configs' / 'project_config.yaml'
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        scheduler_config = config.get('scheduler', {})
    except Exception as e:
        print(f"Warning: Could not load config: {e}")
        scheduler_config = {}

    # Initialize adaptive scheduler
    from traffic_forecast.scheduler import AdaptiveScheduler
    scheduler = AdaptiveScheduler(scheduler_config)

    # Print schedule info if requested
    if args.print_schedule:
        scheduler.print_schedule_summary()
        print()
        # Print cost estimate
        nodes = config.get('node_selection', {}).get('max_nodes', 64)
        k = config.get('collectors', {}).get('google_directions', {}).get('k_neighbors', 3)
        cost_info = scheduler.get_cost_estimate(nodes=nodes, k_neighbors=k)
        print("COST ESTIMATE (30 days):")
        print(f"  Nodes: {cost_info['nodes']}")
        print(f"  k_neighbors: {cost_info['k_neighbors']}")
        print(f"  Edges/collection: {cost_info['edges_per_collection']}")
        print(f"  Collections/day: {cost_info['collections_per_day']}")
        print(f"  Total requests: {cost_info['total_requests']:,}")
        print(f"  Total cost: ${cost_info['total_cost_usd']:,.2f}")
        print(f"  Cost/week: ${cost_info['cost_per_week']:,.2f}")
        print(f"  Cost/day: ${cost_info['cost_per_day']:,.2f}")
        return

    if args.once or args.interval <= 0:
        # single run: create a timestamped run dir, export env vars and run
        ts = datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')
        node_dir = os.path.join('data', 'node', ts)
        image_file = os.path.join('data', 'images', f'{ts}.png')
        os.makedirs(node_dir, exist_ok=True)
        os.makedirs(os.path.dirname(image_file), exist_ok=True)
        os.environ['RUN_DIR'] = node_dir
        os.environ['RUN_IMAGE_FILE'] = image_file
        # write a small manifest for this run
        try:
            import json
            manifest = {'timestamp': ts, 'node_dir': node_dir, 'image_file': image_file}
            with open(os.path.join(node_dir, 'manifest.json'), 'w', encoding='utf-8') as mf:
                json.dump(manifest, mf)
        except Exception:
            pass

        ok = run_collectors(args)
        if ok and not args.no_visualize:
            run_visualization()

        # After successful run, consolidate outputs into data/node/<ts>/run_data.json
        try:
            # collect all files under node_dir/collectors/* and merge into run_data.json
            combined = {}
            collectors_root = os.path.join(node_dir, 'collectors')
            if os.path.exists(collectors_root):
                for root, dirs, files in os.walk(collectors_root):
                    for fn in files:
                        if fn.endswith('.json'):
                            p = os.path.join(root, fn)
                            try:
                                with open(p, 'r', encoding='utf-8') as f:
                                    combined[fn] = json.load(f)
                            except Exception:
                                try:
                                    with open(p, 'r', encoding='utf-8') as f:
                                        combined[fn] = f.read()
                                except Exception:
                                    combined[fn] = None
            with open(os.path.join(node_dir, 'run_data.json'), 'w', encoding='utf-8') as outj:
                json.dump(combined, outj, indent=2)
        except Exception:
            pass
        return

    print(f"Starting collection loop every {args.interval} seconds. Press Ctrl+C to stop.")
    try:
        while True:
            # For each iteration create a fresh timestamped run folder so runs do not overwrite previous data
            ts = datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')
            node_dir = os.path.join('data', 'node', ts)
            image_file = os.path.join('data', 'images', f'{ts}.png')
            os.makedirs(node_dir, exist_ok=True)
            os.makedirs(os.path.dirname(image_file), exist_ok=True)
            os.environ['RUN_DIR'] = node_dir
            os.environ['RUN_IMAGE_FILE'] = image_file
            # write per-run manifest
            try:
                import json
                manifest = {'timestamp': ts, 'node_dir': node_dir, 'image_file': image_file}
                with open(os.path.join(node_dir, 'manifest.json'), 'w', encoding='utf-8') as mf:
                    json.dump(manifest, mf)
            except Exception:
                pass

            ts_now = time.strftime('%Y-%m-%d %H:%M:%S')
            print(f"\n=== Run at {ts_now} (node_dir={node_dir}) ===")
            ok = run_collectors(args)
            if ok and not args.no_visualize:
                run_visualization()
                # After each iteration consolidate into data/node/<ts>/run_data.json
                try:
                    combined = {}
                    collectors_root = os.path.join(node_dir, 'collectors')
                    if os.path.exists(collectors_root):
                        for root, dirs, files in os.walk(collectors_root):
                            for fn in files:
                                if fn.endswith('.json'):
                                    p = os.path.join(root, fn)
                                    try:
                                        with open(p, 'r', encoding='utf-8') as f:
                                            combined[fn] = json.load(f)
                                    except Exception:
                                        try:
                                            with open(p, 'r', encoding='utf-8') as f:
                                                combined[fn] = f.read()
                                        except Exception:
                                            combined[fn] = None
                    with open(os.path.join(node_dir, 'run_data.json'), 'w', encoding='utf-8') as outj:
                        json.dump(combined, outj, indent=2)
                except Exception:
                    pass
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print('Stopped by user')

    # Adaptive scheduling mode (NEW in v4.0)
    if args.adaptive:
        print("=" * 70)
        print("ADAPTIVE SCHEDULING MODE")
        print("=" * 70)
        scheduler.print_schedule_summary()
        print()
        
        last_collection_time = None
        
        try:
            while True:
                current_time = datetime.datetime.now()
                
                # Check if should collect now
                if scheduler.should_collect_now(last_collection_time, current_time):
                    # Create timestamped run folder
                    ts = datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')
                    node_dir = os.path.join('data', 'node', ts)
                    image_file = os.path.join('data', 'images', f'{ts}.png')
                    os.makedirs(node_dir, exist_ok=True)
                    os.makedirs(os.path.dirname(image_file), exist_ok=True)
                    os.environ['RUN_DIR'] = node_dir
                    os.environ['RUN_IMAGE_FILE'] = image_file
                    
                    # Write manifest with schedule info
                    try:
                        import json
                        schedule_info = scheduler.get_schedule_info(current_time)
                        manifest = {
                            'timestamp': ts,
                            'node_dir': node_dir,
                            'image_file': image_file,
                            'schedule': schedule_info
                        }
                        with open(os.path.join(node_dir, 'manifest.json'), 'w', encoding='utf-8') as mf:
                            json.dump(manifest, mf, indent=2)
                    except Exception:
                        pass
                    
                    # Run collection
                    ts_now = time.strftime('%Y-%m-%d %H:%M:%S')
                    schedule_info = scheduler.get_schedule_info(current_time)
                    print(f"\n=== Collection at {ts_now} ({schedule_info['schedule_type']}) ===")
                    ok = run_collectors(args)
                    
                    if ok and not args.no_visualize:
                        run_visualization()
                        
                        # Consolidate outputs
                        try:
                            combined = {}
                            collectors_root = os.path.join(node_dir, 'collectors')
                            if os.path.exists(collectors_root):
                                for root, dirs, files in os.walk(collectors_root):
                                    for fn in files:
                                        if fn.endswith('.json'):
                                            p = os.path.join(root, fn)
                                            try:
                                                with open(p, 'r', encoding='utf-8') as f:
                                                    combined[fn] = json.load(f)
                                            except Exception:
                                                try:
                                                    with open(p, 'r', encoding='utf-8') as f:
                                                        combined[fn] = f.read()
                                                except Exception:
                                                    combined[fn] = None
                            with open(os.path.join(node_dir, 'run_data.json'), 'w', encoding='utf-8') as outj:
                                json.dump(combined, outj, indent=2)
                        except Exception:
                            pass
                    
                    last_collection_time = current_time
                
                # Calculate next check time
                next_collection = scheduler.get_next_collection_time(last_collection_time, current_time)
                if next_collection > current_time:
                    wait_seconds = (next_collection - current_time).total_seconds()
                    wait_seconds = min(wait_seconds, 60)  # Check at least every minute
                    print(f"Next collection at {next_collection.strftime('%Y-%m-%d %H:%M:%S')} "
                          f"(waiting {wait_seconds:.0f}s)")
                    time.sleep(wait_seconds)
                else:
                    time.sleep(10)  # Short sleep before rechecking
                    
        except KeyboardInterrupt:
            print('\nStopped by user')


if __name__ == '__main__':
    main()
