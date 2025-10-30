"""
CLI Runner for Traffic Forecasting Pipeline

Run the complete pipeline from command line.

Usage:
    python cli/run_pipeline.py --config configs/pipeline_config.yaml
    python cli/run_pipeline.py --quick-test
    python cli/run_pipeline.py --train-only --models lstm astgcn

Author: thatlq1812
"""

import argparse
import sys
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from traffic_forecast.core import PipelineConfig, TrafficForecastPipeline


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('pipeline.log')
        ]
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Traffic Forecasting Pipeline CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline with default config
  python cli/run_pipeline.py

  # Run with custom config
  python cli/run_pipeline.py --config configs/my_config.yaml

  # Quick test (1 epoch)
  python cli/run_pipeline.py --quick-test

  # Train specific models only
  python cli/run_pipeline.py --train-only --models lstm astgcn

  # Skip preprocessing (use existing)
  python cli/run_pipeline.py --use-preprocessed

  # Verbose output
  python cli/run_pipeline.py --verbose
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file (YAML or JSON)'
    )
    
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Quick test mode (1 epoch, small data)'
    )
    
    parser.add_argument(
        '--train-only',
        action='store_true',
        help='Skip data loading and preprocessing'
    )
    
    parser.add_argument(
        '--models',
        nargs='+',
        choices=['lstm', 'astgcn'],
        help='Models to train (default: all)'
    )
    
    parser.add_argument(
        '--use-preprocessed',
        action='store_true',
        help='Use preprocessed data instead of raw'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save trained models'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        help='Number of training epochs (overrides config)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Batch size (overrides config)'
    )
    
    return parser.parse_args()


def main():
    """Main CLI entry point."""
    args = parse_args()
    
    # Setup logging
    setup_logging(verbose=args.verbose)
    logger = logging.getLogger(__name__)
    
    logger.info("="*70)
    logger.info("Traffic Forecasting Pipeline CLI")
    logger.info("="*70)
    
    try:
        # Load configuration
        if args.config:
            config_path = Path(args.config)
            if config_path.suffix == '.yaml' or config_path.suffix == '.yml':
                config = PipelineConfig.from_yaml(config_path)
            elif config_path.suffix == '.json':
                config = PipelineConfig.from_json(config_path)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")
            
            logger.info(f"Loaded config from: {config_path}")
        else:
            config = PipelineConfig()
            logger.info("Using default configuration")
        
        # Apply CLI overrides
        if args.quick_test:
            logger.info("Quick test mode enabled")
            config.max_runs = 2
            for model_config in config.models.values():
                model_config.epochs = 1
                model_config.batch_size = 16
        
        if args.use_preprocessed:
            config.use_preprocessed = True
            logger.info("Using preprocessed data")
        
        if args.train_only:
            config.enable_preprocessing = False
            config.enable_feature_engineering = False
            logger.info("Training only mode")
        
        if args.no_save:
            config.enable_save_models = False
            logger.info("Model saving disabled")
        
        if args.models:
            # Disable models not in the list
            for model_name in config.models.keys():
                config.models[model_name].enabled = model_name in args.models
            logger.info(f"Enabled models: {args.models}")
        
        if args.epochs:
            for model_config in config.models.values():
                model_config.epochs = args.epochs
            logger.info(f"Epochs set to: {args.epochs}")
        
        if args.batch_size:
            for model_config in config.models.values():
                model_config.batch_size = args.batch_size
            logger.info(f"Batch size set to: {args.batch_size}")
        
        config.verbose = args.verbose
        
        # Create and run pipeline
        logger.info("\nInitializing pipeline...")
        pipeline = TrafficForecastPipeline(config)
        
        logger.info("\nRunning pipeline...")
        results = pipeline.run_full_pipeline()
        
        # Save summary
        summary_path = pipeline.save_summary()
        
        # Print summary
        logger.info("\n" + "="*70)
        logger.info("Pipeline Execution Summary")
        logger.info("="*70)
        
        if results['status'] == 'success':
            logger.info("✅ Pipeline completed successfully!")
            
            if 'evaluation' in results['steps']:
                logger.info("\nModel Performance:")
                for model_name, metrics in results['steps']['evaluation'].items():
                    if isinstance(metrics, dict) and 'rmse' in metrics:
                        logger.info(f"  {model_name.upper()}:")
                        logger.info(f"    - RMSE: {metrics['rmse']:.4f} km/h")
                        logger.info(f"    - MAE:  {metrics['mae']:.4f} km/h")
            
            logger.info(f"\nSummary saved to: {summary_path}")
            return 0
        
        else:
            logger.error("❌ Pipeline failed!")
            if 'error' in results:
                logger.error(f"Error: {results['error']}")
            return 1
    
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
