#!/usr/bin/env python3
"""
Helper: run collectors then visualization.

Usage examples:
  # one-time run
  python scripts/collect_and_render.py --once

  # run collectors every 10 minutes
  python scripts/collect_and_render.py --interval 600

This script runs the packaged collectors and visualization modules.
"""
import argparse
import subprocess
import time
import sys
import os
import datetime
import pathlib

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
    parser.add_argument('--bbox', help='Optional bbox to pass to collectors (min_lat,min_lon,max_lat,max_lon)')
    parser.add_argument('--no-visualize', action='store_true', help='Skip visualization step')
    args = parser.parse_args()

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


if __name__ == '__main__':
    main()
