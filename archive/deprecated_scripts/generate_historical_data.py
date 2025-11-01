#!/usr/bin/env python3
"""
Generate historical traffic data for training.

Creates synthetic traffic runs for a date range by:
1. Using an existing run as a template
2. Generating runs for each hour in the date range
3. Adding realistic temporal variations (time-of-day, day-of-week patterns)
4. Augmenting each synthetic run to create diverse samples
"""

import json
import logging
import random
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from traffic_forecast.augmentation.run_based_augmentor import RunBasedAugmentor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HistoricalDataGenerator:
    """Generate historical traffic data from a template run."""
    
    def __init__(
        self,
        template_run_dir: Path,
        output_base_dir: Path,
        augmentor: Optional[RunBasedAugmentor] = None
    ):
        """
        Initialize the generator.
        
        Args:
            template_run_dir: Path to template run directory
            output_base_dir: Base directory for output runs
            augmentor: Optional augmentor for creating variations
        """
        self.template_run_dir = Path(template_run_dir)
        self.output_base_dir = Path(output_base_dir)
        self.augmentor = augmentor
        
        # Load template data
        self.template_data = self._load_template()
        
        # Time-of-day patterns (multipliers for base speed)
        # Morning rush: 6-9am, Evening rush: 17-20pm
        self.hourly_patterns = {
            0: 1.2,   # Midnight - very light traffic
            1: 1.25,  # Early morning
            2: 1.3,
            3: 1.3,
            4: 1.25,
            5: 1.15,
            6: 0.85,  # Morning rush starts
            7: 0.75,  # Peak morning rush
            8: 0.80,
            9: 0.90,
            10: 1.0,  # Normal daytime
            11: 1.0,
            12: 0.95, # Lunch hour
            13: 0.95,
            14: 1.0,
            15: 0.95,
            16: 0.90,
            17: 0.80, # Evening rush starts
            18: 0.75, # Peak evening rush
            19: 0.85,
            20: 0.90,
            21: 1.0,
            22: 1.1,
            23: 1.15,
        }
        
        # Day-of-week patterns (multipliers)
        self.weekday_patterns = {
            0: 1.0,   # Monday
            1: 1.0,   # Tuesday
            2: 1.0,   # Wednesday
            3: 1.0,   # Thursday
            4: 0.95,  # Friday - slightly lighter
            5: 1.1,   # Saturday - lighter traffic
            6: 1.15,  # Sunday - lightest traffic
        }
    
    def _load_template(self) -> List[Dict]:
        """Load template traffic data."""
        template_file = self.template_run_dir / "traffic_edges.json"
        
        if not template_file.exists():
            raise FileNotFoundError(f"Template file not found: {template_file}")
        
        with open(template_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded template with {len(data)} edges from {self.template_run_dir.name}")
        return data
    
    def _apply_temporal_patterns(
        self,
        base_speed: float,
        timestamp: datetime
    ) -> float:
        """
        Apply time-of-day and day-of-week patterns to speed.
        
        Args:
            base_speed: Base speed value
            timestamp: Target timestamp
            
        Returns:
            Adjusted speed with temporal patterns
        """
        hour = timestamp.hour
        weekday = timestamp.weekday()
        
        # Get multipliers
        hour_mult = self.hourly_patterns.get(hour, 1.0)
        weekday_mult = self.weekday_patterns.get(weekday, 1.0)
        
        # Combine patterns
        combined_mult = hour_mult * weekday_mult
        
        # Apply with some randomness (±5%)
        noise = np.random.uniform(0.95, 1.05)
        adjusted_speed = base_speed * combined_mult * noise
        
        # Clamp to reasonable range
        return np.clip(adjusted_speed, 5.0, 80.0)
    
    def generate_run(
        self,
        target_timestamp: datetime,
        add_noise: bool = True
    ) -> List[Dict]:
        """
        Generate a synthetic run for a specific timestamp.
        
        Args:
            target_timestamp: Target datetime for the run
            add_noise: Whether to add random noise
            
        Returns:
            List of edge dictionaries with synthetic data
        """
        synthetic_data = []
        
        for edge in self.template_data:
            # Create new edge with same structure
            new_edge = edge.copy()
            
            # Update timestamp
            new_edge['timestamp'] = target_timestamp.isoformat()
            
            # Apply temporal patterns to speed
            base_speed = edge.get('speed_kmh', 30.0)
            new_speed = self._apply_temporal_patterns(base_speed, target_timestamp)
            
            # Add random noise if requested
            if add_noise:
                noise = np.random.normal(0, 2.0)  # ±2 km/h std dev
                new_speed += noise
            
            # Clamp to valid range
            new_speed = np.clip(new_speed, 5.0, 80.0)
            new_edge['speed_kmh'] = round(new_speed, 2)
            
            synthetic_data.append(new_edge)
        
        return synthetic_data
    
    def save_run(
        self,
        data: List[Dict],
        timestamp: datetime,
        run_dir: Optional[Path] = None
    ) -> Path:
        """
        Save synthetic run to disk.
        
        Args:
            data: Edge data to save
            timestamp: Run timestamp
            run_dir: Optional custom run directory
            
        Returns:
            Path to created run directory
        """
        if run_dir is None:
            run_name = f"run_{timestamp.strftime('%Y%m%d_%H%M%S')}"
            run_dir = self.output_base_dir / run_name
        
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save traffic data
        output_file = run_dir / "traffic_edges.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        # Copy metadata if exists
        metadata_file = self.template_run_dir / "metadata.json"
        if metadata_file.exists():
            # Update metadata with new timestamp
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            metadata['collection_timestamp'] = timestamp.isoformat()
            metadata['synthetic'] = True
            metadata['template_run'] = self.template_run_dir.name
            
            with open(run_dir / "metadata.json", 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
        
        return run_dir
    
    def generate_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        hours_per_day: int = 24,
        variations_per_run: int = 20,
        skip_existing: bool = True
    ) -> Dict[str, int]:
        """
        Generate synthetic runs for a date range.
        
        Args:
            start_date: Start datetime
            end_date: End datetime
            hours_per_day: Number of runs per day (evenly spaced)
            variations_per_run: Number of variations per base run
            skip_existing: Skip runs that already exist
            
        Returns:
            Statistics dictionary
        """
        stats = {
            'total_generated': 0,
            'total_variations': 0,
            'skipped': 0,
            'errors': 0
        }
        
        # Calculate timestamps
        current = start_date
        hour_interval = 24 // hours_per_day
        
        timestamps = []
        while current <= end_date:
            for hour_offset in range(0, 24, hour_interval):
                ts = current.replace(hour=hour_offset, minute=0, second=0, microsecond=0)
                if ts <= end_date:
                    timestamps.append(ts)
            current += timedelta(days=1)
        
        total_timestamps = len(timestamps)
        logger.info(f"Generating {total_timestamps} base runs from {start_date.date()} to {end_date.date()}")
        logger.info(f"Each run will have {variations_per_run} variations")
        logger.info(f"Total expected: {total_timestamps * (1 + variations_per_run)} runs")
        
        # Generate runs
        for idx, ts in enumerate(timestamps, 1):
            try:
                # Generate base synthetic run
                synthetic_data = self.generate_run(ts, add_noise=True)
                
                # Create base run + variations
                base_run_name = f"run_{ts.strftime('%Y%m%d_%H%M%S')}"
                base_run_dir = self.output_base_dir / base_run_name
                
                # Skip if base exists
                if skip_existing and base_run_dir.exists():
                    stats['skipped'] += (1 + variations_per_run)
                    if idx % 100 == 0:
                        logger.info(f"Progress: {idx}/{total_timestamps} ({stats['skipped']} skipped)")
                    continue
                
                # Save base run
                saved_dir = self.save_run(synthetic_data, ts, base_run_dir)
                stats['total_generated'] += 1
                
                # Create variations
                for var_idx in range(variations_per_run):
                    # Create variation with different noise
                    var_data = self.generate_run(ts, add_noise=True)
                    
                    # Calculate offset timestamp (small seconds offset to avoid collision)
                    offset_seconds = (var_idx + 1) * 10  # 10, 20, 30, ... seconds
                    var_ts = ts + timedelta(seconds=offset_seconds)
                    
                    var_run_name = f"run_{var_ts.strftime('%Y%m%d_%H%M%S')}"
                    var_run_dir = self.output_base_dir / var_run_name
                    
                    if skip_existing and var_run_dir.exists():
                        stats['skipped'] += 1
                        continue
                    
                    # Save variation
                    self.save_run(var_data, var_ts, var_run_dir)
                    stats['total_variations'] += 1
                
                # Log progress
                if idx % 100 == 0 or idx == total_timestamps:
                    logger.info(
                        f"Progress: {idx}/{total_timestamps} "
                        f"(Base: {stats['total_generated']}, "
                        f"Variations: {stats['total_variations']}, "
                        f"Skipped: {stats['skipped']})"
                    )
                
            except Exception as e:
                logger.error(f"Error generating run at {ts}: {e}")
                stats['errors'] += 1
        
        return stats


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate historical traffic data from template',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate data for September 2025 (1 run per hour, 20 variations each)
  python scripts/generate_historical_data.py --start 2025-09-01 --end 2025-09-30
  
  # Generate data for 2 months with different variation count
  python scripts/generate_historical_data.py --start 2025-09-01 --end 2025-10-31 --variations 50
  
  # Generate without variations
  python scripts/generate_historical_data.py --start 2025-09-01 --end 2025-09-30 --variations 0
  
  # Use specific template run
  python scripts/generate_historical_data.py --template data/runs/run_20251030_032440 --start 2025-09-01 --end 2025-09-30
        """
    )
    
    parser.add_argument(
        '--template',
        type=str,
        default=None,
        help='Path to template run directory (default: auto-select first available run)'
    )
    
    parser.add_argument(
        '--start',
        type=str,
        required=True,
        help='Start date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end',
        type=str,
        required=True,
        help='End date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/runs',
        help='Output base directory (default: data/runs)'
    )
    
    parser.add_argument(
        '--hours-per-day',
        type=int,
        default=24,
        help='Number of runs per day (default: 24, one per hour)'
    )
    
    parser.add_argument(
        '--variations',
        type=int,
        default=20,
        help='Number of variations per base run (default: 20, set 0 to disable)'
    )
    
    parser.add_argument(
        '--no-skip',
        action='store_true',
        help='Regenerate existing runs instead of skipping them'
    )
    
    args = parser.parse_args()
    
    # Parse dates
    start_date = datetime.strptime(args.start, '%Y-%m-%d')
    end_date = datetime.strptime(args.end, '%Y-%m-%d').replace(hour=23, minute=59, second=59)
    
    # Find template
    output_dir = Path(args.output)
    if args.template:
        template_dir = Path(args.template)
    else:
        # Auto-select first run
        run_dirs = sorted(output_dir.glob('run_*'))
        if not run_dirs:
            logger.error(f"No runs found in {output_dir}")
            return 1
        template_dir = run_dirs[0]
    
    if not template_dir.exists():
        logger.error(f"Template directory not found: {template_dir}")
        return 1
    
    logger.info(f"Using template: {template_dir}")
    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    logger.info(f"Runs per day: {args.hours_per_day}")
    logger.info(f"Variations: {args.variations}x per base run")
    
    # Create generator
    generator = HistoricalDataGenerator(
        template_run_dir=template_dir,
        output_base_dir=output_dir,
        augmentor=None  # Not using RunBasedAugmentor for variations
    )
    
    # Generate data
    logger.info("Starting generation...")
    stats = generator.generate_date_range(
        start_date=start_date,
        end_date=end_date,
        hours_per_day=args.hours_per_day,
        variations_per_run=args.variations,
        skip_existing=not args.no_skip
    )
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("GENERATION COMPLETE")
    logger.info("="*60)
    logger.info(f"Base runs generated: {stats['total_generated']}")
    logger.info(f"Variations created: {stats['total_variations']}")
    logger.info(f"Runs skipped (already exist): {stats['skipped']}")
    logger.info(f"Errors: {stats['errors']}")
    logger.info(f"Total new runs: {stats['total_generated'] + stats['total_variations']}")
    logger.info("="*60)
    
    return 0


if __name__ == '__main__':
    exit(main())
