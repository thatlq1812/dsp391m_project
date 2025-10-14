#!/usr/bin/env python3
"""
Cleanup old run folders under data/node and data/images.
Usage: python scripts/cleanup_runs.py --days 7
"""
import argparse
import os
import shutil
import time
from pathlib import Path

from traffic_forecast import PROJECT_ROOT

def rm_old(base_dir, days):
    cutoff = time.time() - days * 86400
    if not os.path.exists(base_dir):
        return 0
    removed = 0
    for name in os.listdir(base_dir):
        path = os.path.join(base_dir, name)
        try:
            mtime = os.path.getmtime(path)
            if mtime < cutoff:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
                removed += 1
        except Exception:
            continue
    return removed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, default=30, help='Remove runs older than DAYS')
    args = parser.parse_args()
    node_root = PROJECT_ROOT / 'data' / 'node'
    image_root = PROJECT_ROOT / 'data' / 'images'
    r1 = rm_old(node_root, args.days)
    r2 = rm_old(image_root, args.days)
    print(f'Removed {r1} node folders and {r2} image folders older than {args.days} days')
