#!/usr/bin/env python3
"""
Backfill Overpass data for collections that are missing it.

Since Overpass data (nodes and edges) is static topology data that doesn't change,
we can safely copy it from a successful collection to all other collections.

Usage:
 python scripts/backfill_overpass_data.py [--data-dir DATA_DIR] [--dry-run]
"""

import os
import json
import shutil
import argparse
from pathlib import Path
from typing import Optional, Tuple


def find_source_overpass_data(data_dir: Path) -> Optional[Path]:
 """
 Find a collection run that has complete Overpass data.
 
 Returns:
 Path to the collectors/overpass directory with nodes.json and edges.json
 """
 # Check if data_dir has 'node' subdirectory
 if (data_dir / 'node').exists():
 search_dir = data_dir / 'node'
 else:
 search_dir = data_dir
 
 node_dirs = sorted(search_dir.glob('*/collectors/overpass'))
 
 for overpass_dir in node_dirs:
 nodes_file = overpass_dir / 'nodes.json'
 edges_file = overpass_dir / 'edges.json'
 stats_file = overpass_dir / 'statistics.json'
 
 # Check if required files exist (edges can be empty array)
 if (nodes_file.exists() and nodes_file.stat().st_size > 100 and
 edges_file.exists()):
 print(f"[OK] Found source Overpass data: {overpass_dir.parent.parent.name}")
 return overpass_dir
 
 return None


def validate_overpass_data(overpass_dir: Path) -> Tuple[bool, str]:
 """
 Validate that Overpass data is complete and valid.
 
 Returns:
 (is_valid, message)
 """
 nodes_file = overpass_dir / 'nodes.json'
 edges_file = overpass_dir / 'edges.json'
 
 try:
 with open(nodes_file) as f:
 nodes = json.load(f)
 with open(edges_file) as f:
 edges = json.load(f)
 
 if not isinstance(nodes, list) or len(nodes) == 0:
 return False, "nodes.json is empty or invalid"
 
 if not isinstance(edges, list):
 return False, "edges.json is invalid"
 
 # Check structure of first node
 if nodes:
 required_fields = ['node_id', 'lat', 'lon']
 if not all(field in nodes[0] for field in required_fields):
 return False, f"nodes missing required fields: {required_fields}"
 
 return True, f"Valid: {len(nodes)} nodes, {len(edges)} edges"
 
 except Exception as e:
 return False, f"Error validating data: {e}"


def backfill_collection(collection_dir: Path, source_overpass_dir: Path, dry_run: bool = False) -> bool:
 """
 Copy Overpass data to a collection directory if it's missing.
 
 Returns:
 True if backfilled, False if skipped
 """
 target_overpass_dir = collection_dir / 'collectors' / 'overpass'
 nodes_file = target_overpass_dir / 'nodes.json'
 
 # Skip if already has nodes.json
 if nodes_file.exists() and nodes_file.stat().st_size > 100:
 return False
 
 if dry_run:
 print(f" [DRY RUN] Would backfill: {collection_dir.name}")
 return True
 
 # Create target directory
 target_overpass_dir.mkdir(parents=True, exist_ok=True)
 
 # Copy all Overpass files
 for file in source_overpass_dir.glob('*.json'):
 target_file = target_overpass_dir / file.name
 shutil.copy2(file, target_file)
 
 print(f" [OK] Backfilled: {collection_dir.name}")
 return True


def main():
 parser = argparse.ArgumentParser(description='Backfill missing Overpass data')
 parser.add_argument('--data-dir', type=str, default='data',
 help='Path to data directory (default: data)')
 parser.add_argument('--dry-run', action='store_true',
 help='Show what would be done without making changes')
 args = parser.parse_args()
 
 data_dir = Path(args.data_dir)
 
 if not data_dir.exists():
 print(f"[ERROR] Error: Data directory not found: {data_dir}")
 return 1
 
 print("="*70)
 print("OVERPASS DATA BACKFILL SCRIPT")
 print("="*70)
 print()
 
 # Step 1: Find source data
 print("Step 1: Finding source Overpass data...")
 source_dir = find_source_overpass_data(data_dir)
 
 if not source_dir:
 print("[ERROR] Error: Could not find any collection with complete Overpass data")
 print("\nTroubleshooting:")
 print(" 1. Run a manual collection to generate Overpass data:")
 print(" conda run -n dsp python -m traffic_forecast.collectors.overpass.collector")
 print(" 2. Or copy nodes.json from a successful run manually")
 return 1
 
 print()
 
 # Step 2: Validate source data
 print("Step 2: Validating source data...")
 is_valid, message = validate_overpass_data(source_dir)
 
 if not is_valid:
 print(f"[ERROR] Error: Source data validation failed: {message}")
 return 1
 
 print(f"[OK] {message}")
 print()
 
 # Step 3: Find collections to backfill
 print("Step 3: Finding collections to backfill...")
 
 # Check if data_dir has 'node' subdirectory
 if (data_dir / 'node').exists():
 search_dir = data_dir / 'node'
 else:
 search_dir = data_dir
 
 collection_dirs = sorted(search_dir.glob('*'))
 # Filter to only directories that look like collection runs (timestamp format)
 collection_dirs = [d for d in collection_dirs if d.is_dir() and d.name.isdigit()]
 
 if not collection_dirs:
 print("[ERROR] Error: No collection directories found")
 return 1
 
 print(f"Found {len(collection_dirs)} total collections")
 print()
 
 # Step 4: Backfill missing data
 print("Step 4: Backfilling missing Overpass data...")
 if args.dry_run:
 print("[DRY RUN MODE - No changes will be made]")
 print()
 
 backfilled_count = 0
 for collection_dir in collection_dirs:
 if backfill_collection(collection_dir, source_dir, args.dry_run):
 backfilled_count += 1
 
 print()
 print("="*70)
 print("BACKFILL COMPLETE")
 print("="*70)
 print(f"Total collections: {len(collection_dirs)}")
 print(f"Backfilled: {backfilled_count}")
 print(f"Already had data: {len(collection_dirs) - backfilled_count}")
 
 if args.dry_run:
 print()
 print("This was a DRY RUN. Run without --dry-run to apply changes.")
 
 return 0


if __name__ == '__main__':
 exit(main())
