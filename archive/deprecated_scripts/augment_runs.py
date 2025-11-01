"""
Run-Based Data Augmentation Script

Generate synthetic run folders to achieve target sample density (20/hour).

Usage:
    python scripts/augment_runs.py --dry-run                    # Test mode
    python scripts/augment_runs.py --max-runs 5                 # Process first 5 runs
    python scripts/augment_runs.py --target-samples 30          # 30 samples/hour
    python scripts/augment_runs.py                              # Process all runs

Author: thatlq1812
"""

import sys
import logging
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from traffic_forecast.augmentation import RunBasedAugmentor


def main():
    parser = argparse.ArgumentParser(
        description='Generate augmented traffic data runs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test mode (no files created)
  python scripts/augment_runs.py --dry-run
  
  # Process first 5 runs
  python scripts/augment_runs.py --max-runs 5
  
  # Custom target density
  python scripts/augment_runs.py --target-samples 30
  
  # Full augmentation
  python scripts/augment_runs.py
        """
    )
    
    parser.add_argument(
        '--source',
        type=str,
        default=str(PROJECT_ROOT / 'data' / 'runs'),
        help='Source directory containing original runs (default: data/runs)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output directory for augmented runs (default: same as source)'
    )
    
    parser.add_argument(
        '--max-runs',
        type=int,
        help='Maximum number of original runs to process'
    )
    
    parser.add_argument(
        '--target-samples',
        type=int,
        default=20,
        help='Target samples per hour (default: 20)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Simulate augmentation without creating files'
    )
    
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Verbose output (DEBUG level)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    logger = logging.getLogger(__name__)
    
    # Print configuration
    print("=" * 70)
    print("Run-Based Data Augmentation")
    print("=" * 70)
    print(f"Source directory:     {args.source}")
    print(f"Output directory:     {args.output or args.source}")
    print(f"Target density:       {args.target_samples} samples/hour")
    print(f"Max runs to process:  {args.max_runs or 'All'}")
    print(f"Mode:                 {'DRY RUN (simulation)' if args.dry_run else 'PRODUCTION'}")
    print("=" * 70)
    print()
    
    if args.dry_run:
        print("WARNING: DRY RUN MODE - No files will be created\n")
    
    # Create augmentor
    try:
        augmentor = RunBasedAugmentor(
            source_dir=Path(args.source),
            output_dir=Path(args.output) if args.output else None,
            target_samples_per_hour=args.target_samples,
            dry_run=args.dry_run
        )
    except Exception as e:
        logger.error(f"Failed to initialize augmentor: {e}")
        return 1
    
    # Run augmentation
    try:
        stats = augmentor.augment_all_runs(max_runs=args.max_runs)
        
        # Print summary
        print("\n" + "=" * 70)
        print("Augmentation Summary")
        print("=" * 70)
        print(f"Original runs processed:  {stats['runs_processed']}")
        print(f"Augmented runs created:   {stats['runs_created']}")
        print(f"Total samples generated:  {stats['total_samples_generated']:,}")
        print(f"Errors encountered:       {len(stats['errors'])}")
        
        if stats['errors']:
            print("\nErrors:")
            for error in stats['errors'][:5]:  # Show first 5
                print(f"  - {error['run']}: {error['error']}")
            if len(stats['errors']) > 5:
                print(f"  ... and {len(stats['errors']) - 5} more")
        
        print("=" * 70)
        
        if stats['runs_created'] > 0 and not args.dry_run:
            print(f"\nSuccess! Created {stats['runs_created']} augmented runs")
            print(f"  Location: {args.output or args.source}")
        elif args.dry_run:
            print(f"\nDry run complete! Would create {stats['runs_created']} runs")
            print("  Run without --dry-run to create files")
        else:
            print("\nNo augmentation needed - data already at target density")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nWARNING: Augmentation interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Augmentation failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
