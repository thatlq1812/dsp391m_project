"""
Test training script - Quick verification of model training pipeline

Author: thatlq1812
"""

import logging
from pathlib import Path
import sys

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from traffic_forecast.core import PipelineConfig, TrafficForecastPipeline
from traffic_forecast.core.config import ModelConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_training():
    """Test model training with quick settings."""
    
    logger.info("="*70)
    logger.info("Testing Traffic Forecast Training Pipeline")
    logger.info("="*70)
    
    # Create config
    config = PipelineConfig()
    config.use_preprocessed = True
    config.test_size = 0.2
    
    # Quick test - just 2 epochs
    config.models['lstm'] = ModelConfig(
        name='lstm',
        enabled=True,
        epochs=2,
        batch_size=32,
        learning_rate=0.001,
        sequence_length=12,
        validation_split=0.2
    )
    
    # Disable ASTGCN for now - requires graph-structured input
    config.models['astgcn'] = ModelConfig(
        name='astgcn',
        enabled=False,
        epochs=2,
        batch_size=32,
        learning_rate=0.001,
        sequence_length=12,
        validation_split=0.2
    )
    
    # Create pipeline
    logger.info("\n[1] Creating pipeline...")
    pipeline = TrafficForecastPipeline(config)
    
    # Load data
    logger.info("\n[2] Loading preprocessed data...")
    pipeline.data_manager.load_processed_data()
    logger.info(f"✓ Loaded {len(pipeline.data_manager.data_features)} samples")
    
    # Prepare train/test split
    logger.info("\n[3] Preparing train/test split...")
    pipeline.data_manager.prepare_train_test()
    logger.info(f"✓ Train: {len(pipeline.data_manager.X_train)}, Test: {len(pipeline.data_manager.X_test)}")
    
    # Train models
    logger.info("\n[4] Training models...")
    training_results = pipeline.train_models()
    
    for model_name, result in training_results.items():
        if result['status'] == 'success':
            logger.info(f"✓ {model_name.upper()}: {result['epochs_trained']} epochs, loss={result['final_loss']:.4f}")
        else:
            logger.warning(f"✗ {model_name.upper()}: {result.get('error', 'Unknown error')}")
    
    # Evaluate
    logger.info("\n[5] Evaluating models...")
    evaluation_results = pipeline.evaluate_models()
    
    for model_name, metrics in evaluation_results.items():
        if 'error' not in metrics:
            logger.info(f"✓ {model_name.upper()}: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}")
        else:
            logger.warning(f"✗ {model_name.upper()}: {metrics['error']}")
    
    # Save models
    logger.info("\n[6] Saving models...")
    save_results = pipeline.save_models()
    
    for model_name, result in save_results.items():
        if result.get('status') == 'success':
            logger.info(f"✓ {model_name.upper()}: {result['model_path']}")
        else:
            logger.warning(f"✗ {model_name.upper()}: {result.get('error', 'Save failed')}")
    
    logger.info("\n" + "="*70)
    logger.info("✅ Training test completed!")
    logger.info("="*70)
    
    return pipeline


if __name__ == '__main__':
    try:
        pipeline = test_training()
    except Exception as e:
        logger.error(f"Training test failed: {e}", exc_info=True)
        sys.exit(1)
