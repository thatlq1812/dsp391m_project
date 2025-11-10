"""
Extract edge coordinates from training data and save to cache.

This script:
1. Reads parquet file with traffic data
2. Extracts unique edges with lat/lon coordinates
3. Saves to cache/edge_coordinates.json for fast API loading

Author: THAT Le Quang
"""

import json
import logging
from pathlib import Path
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_FILE = PROJECT_ROOT / 'data' / 'processed' / 'all_runs_extreme_augmented.parquet'
OUTPUT_FILE = PROJECT_ROOT / 'cache' / 'edge_coordinates.json'


def extract_edges():
    """Extract unique edges with coordinates."""
    logger.info(f"Loading data from {DATA_FILE}...")
    df = pd.read_parquet(DATA_FILE)
    
    logger.info(f"Data shape: {df.shape}")
    
    # Get unique edges with coordinates
    edges_df = df[['node_a_id', 'node_b_id', 'lat_a', 'lon_a', 'lat_b', 'lon_b']].drop_duplicates()
    
    logger.info(f"Found {len(edges_df)} unique edges")
    
    # Convert to list of dicts
    edges = []
    for _, row in edges_df.iterrows():
        edge = {
            'edge_id': f"{row['node_a_id']}-{row['node_b_id']}",
            'start_node': row['node_a_id'],
            'end_node': row['node_b_id'],
            'coordinates': [
                [row['lat_a'], row['lon_a']],  # Start point [lat, lon]
                [row['lat_b'], row['lon_b']]   # End point [lat, lon]
            ]
        }
        edges.append(edge)
    
    # Save to JSON
    logger.info(f"Saving to {OUTPUT_FILE}...")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump({
            'total_edges': len(edges),
            'edges': edges
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✓ Saved {len(edges)} edges to {OUTPUT_FILE}")
    
    # Print sample
    logger.info("\nSample edge:")
    logger.info(json.dumps(edges[0], indent=2))


if __name__ == '__main__':
    extract_edges()
