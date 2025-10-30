"""
Main pipeline orchestrator for traffic forecasting.

Manages end-to-end workflow:
1. Data loading
2. Preprocessing
3. Feature engineering
4. Model training (LSTM & ASTGCN)
5. Evaluation
6. Saving results

Author: thatlq1812
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

import numpy as np
import pandas as pd

from traffic_forecast.core.config import PipelineConfig, ModelConfig
from traffic_forecast.core.data_manager import DataManager

# Import DL trainers
try:
    from traffic_forecast.ml.dl_trainer import DLModelTrainer
    HAS_DL_TRAINER = True
except ImportError:
    HAS_DL_TRAINER = False
    logging.warning("DL Trainer not available. Some features may be limited.")

logger = logging.getLogger(__name__)


class TrafficForecastPipeline:
    """Unified pipeline for traffic forecasting with deep learning."""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize pipeline.
        
        Args:
            config: Pipeline configuration (uses default if None)
        """
        self.config = config or PipelineConfig()
        self.data_manager = DataManager(self.config)
        
        # Trained models
        self.models = {}
        self.training_history = {}
        self.evaluation_results = {}
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = logging.INFO if self.config.verbose else logging.WARNING
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete pipeline from start to finish.
        
        Returns:
            Dictionary with results summary
        """
        logger.info("=" * 70)
        logger.info("Starting Traffic Forecasting Pipeline")
        logger.info("=" * 70)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config.to_dict(),
            'steps': {}
        }
        
        try:
            # Step 1: Load data
            if self.config.use_preprocessed:
                logger.info("\n[Step 1] Loading preprocessed data...")
                self.data_manager.load_processed_data()
                results['steps']['load_data'] = {'status': 'loaded_preprocessed'}
            else:
                logger.info("\n[Step 1] Loading raw data...")
                self.data_manager.load_data(max_runs=self.config.max_runs)
                results['steps']['load_data'] = {
                    'status': 'success',
                    'rows': len(self.data_manager.data_raw)
                }
            
            # Step 2: Preprocessing
            if self.config.enable_preprocessing and not self.config.use_preprocessed:
                logger.info("\n[Step 2] Preprocessing data...")
                self.data_manager.preprocess_data()
                results['steps']['preprocessing'] = {
                    'status': 'success',
                    'rows': len(self.data_manager.data_processed)
                }
            else:
                logger.info("\n[Step 2] Skipping preprocessing")
                results['steps']['preprocessing'] = {'status': 'skipped'}
            
            # Step 3: Feature engineering
            if self.config.enable_feature_engineering:
                logger.info("\n[Step 3] Engineering features...")
                if self.config.use_preprocessed:
                    # Already has features
                    self.data_manager.data_features = self.data_manager.data_features
                else:
                    self.data_manager.engineer_features()
                
                results['steps']['feature_engineering'] = {
                    'status': 'success',
                    'num_features': len(self.data_manager.data_features.columns),
                    'rows': len(self.data_manager.data_features)
                }
            else:
                logger.info("\n[Step 3] Skipping feature engineering")
                results['steps']['feature_engineering'] = {'status': 'skipped'}
            
            # Step 4: Prepare train/test
            logger.info("\n[Step 4] Preparing train/test split...")
            self.data_manager.prepare_train_test()
            results['steps']['train_test_split'] = {
                'status': 'success',
                'train_samples': len(self.data_manager.X_train),
                'test_samples': len(self.data_manager.X_test)
            }
            
            # Step 5: Train models
            if self.config.enable_model_training:
                logger.info("\n[Step 5] Training deep learning models...")
                training_results = self.train_models()
                results['steps']['model_training'] = training_results
            else:
                logger.info("\n[Step 5] Skipping model training")
                results['steps']['model_training'] = {'status': 'skipped'}
            
            # Step 6: Evaluate models
            if self.config.enable_evaluation and self.models:
                logger.info("\n[Step 6] Evaluating models...")
                evaluation_results = self.evaluate_models()
                results['steps']['evaluation'] = evaluation_results
            else:
                logger.info("\n[Step 6] Skipping evaluation")
                results['steps']['evaluation'] = {'status': 'skipped'}
            
            # Step 7: Save models
            if self.config.enable_save_models and self.models:
                logger.info("\n[Step 7] Saving models...")
                save_results = self.save_models()
                results['steps']['save_models'] = save_results
            else:
                logger.info("\n[Step 7] Skipping model saving")
                results['steps']['save_models'] = {'status': 'skipped'}
            
            results['status'] = 'success'
            
            logger.info("\n" + "=" * 70)
            logger.info("Pipeline completed successfully!")
            logger.info("=" * 70)
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            results['status'] = 'failed'
            results['error'] = str(e)
        
        return results
    
    def train_models(self) -> Dict[str, Any]:
        """
        Train all enabled deep learning models.
        
        Returns:
            Training results summary
        """
        if not HAS_DL_TRAINER:
            logger.error("DL Trainer not available. Cannot train models.")
            return {'status': 'failed', 'error': 'DL Trainer not available'}
        
        results = {}
        
        # Build adjacency matrix if needed for ASTGCN
        adjacency_matrix = None
        if 'astgcn' in self.config.models and self.config.models['astgcn'].enabled:
            adjacency_matrix = self._build_adjacency_matrix()
        
        for model_name, model_config in self.config.models.items():
            if not model_config.enabled:
                logger.info(f"Skipping {model_name} (disabled)")
                results[model_name] = {'status': 'disabled'}
                continue
            
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"Training {model_name.upper()}")
                logger.info(f"{'='*60}")
                
                # Prepare model params
                model_params = model_config.to_dict()
                
                # LSTM handles sequences internally, ASTGCN needs pre-made sequences
                if model_name == 'lstm':
                    # Pass tabular data - LSTM creates sequences internally
                    X_train_data = self.data_manager.X_train
                    y_train_data = self.data_manager.y_train
                    X_val_data = self.data_manager.X_test
                    y_val_data = self.data_manager.y_test
                    
                    logger.info(f"Data prepared: X_train={X_train_data.shape}, y_train={y_train_data.shape}")
                    
                elif model_name == 'astgcn':
                    # Create sequences for ASTGCN
                    sequence_length = model_config.sequence_length
                    X_train_seq, y_train_seq = self.data_manager.create_sequences(
                        self.data_manager.X_train,
                        self.data_manager.y_train,
                        sequence_length
                    )
                    X_test_seq, y_test_seq = self.data_manager.create_sequences(
                        self.data_manager.X_test,
                        self.data_manager.y_test,
                        sequence_length
                    )
                    
                    X_train_data = X_train_seq
                    y_train_data = y_train_seq
                    X_val_data = X_test_seq
                    y_val_data = y_test_seq
                    
                    logger.info(f"Sequences created: X_train={X_train_seq.shape}, y_train={y_train_seq.shape}")
                    
                    # Add adjacency matrix for ASTGCN
                    if adjacency_matrix is not None:
                        model_params['adjacency'] = adjacency_matrix
                        model_params['num_nodes'] = adjacency_matrix.shape[0]
                else:
                    # Generic case
                    sequence_length = model_config.sequence_length
                    X_train_seq, y_train_seq = self.data_manager.create_sequences(
                        self.data_manager.X_train,
                        self.data_manager.y_train,
                        sequence_length
                    )
                    X_test_seq, y_test_seq = self.data_manager.create_sequences(
                        self.data_manager.X_test,
                        self.data_manager.y_test,
                        sequence_length
                    )
                    
                    X_train_data = X_train_seq
                    y_train_data = y_train_seq
                    X_val_data = X_test_seq
                    y_val_data = y_test_seq
                    
                    logger.info(f"Sequences created: X_train={X_train_seq.shape}, y_train={y_train_seq.shape}")
                
                # Initialize trainer
                trainer = DLModelTrainer(
                    model_type=model_name,
                    params=model_params,
                    model_dir=self.config.models_dir
                )
                
                # Train
                trainer.train(
                    X_train_data, y_train_data,
                    X_val=X_val_data,
                    y_val=y_val_data,
                    epochs=model_config.epochs,
                    batch_size=model_config.batch_size,
                    verbose=1
                )
                
                # Store
                self.models[model_name] = trainer
                if trainer.training_history:
                    # Convert History object to dict
                    if hasattr(trainer.training_history, 'history'):
                        self.training_history[model_name] = trainer.training_history.history
                    else:
                        self.training_history[model_name] = trainer.training_history
                
                # Get history for results
                history = self.training_history.get(model_name, {})
                
                results[model_name] = {
                    'status': 'success',
                    'epochs_trained': len(history.get('loss', [])),
                    'final_loss': float(history['loss'][-1]) if 'loss' in history and history['loss'] else None,
                    'final_val_loss': float(history['val_loss'][-1]) if 'val_loss' in history and history['val_loss'] else None
                }
                
                logger.info(f"✓ {model_name.upper()} trained successfully!")
                
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}", exc_info=True)
                results[model_name] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        return results
    
    def _build_adjacency_matrix(self) -> np.ndarray:
        """
        Build adjacency matrix for graph models.
        
        Returns:
            Adjacency matrix
        """
        try:
            from traffic_forecast.utils.graph_builder import GraphBuilder, build_adjacency_from_runs
            from traffic_forecast import PROJECT_ROOT
            
            # Try to load from cache first
            cache_path = PROJECT_ROOT / 'cache' / 'adjacency_matrix.npy'
            
            if cache_path.exists():
                logger.info(f"Loading cached adjacency matrix from {cache_path}")
                builder = GraphBuilder()
                adj = builder.load_adjacency(cache_path)
                logger.info(f"Loaded adjacency matrix: {adj.shape}")
                return adj
            
            # Build from traffic data in runs
            logger.info("Building adjacency matrix from traffic data...")
            runs_dir = PROJECT_ROOT / 'data' / 'runs'
            
            adj = build_adjacency_from_runs(
                runs_dir=runs_dir,
                output_path=cache_path,
                method='distance'
            )
            
            logger.info(f"Built and cached adjacency matrix: {adj.shape}")
            
            return adj
            
        except Exception as e:
            logger.error(f"Failed to build adjacency matrix: {e}")
            logger.warning("Using identity matrix as fallback")
            # Fallback to identity matrix
            num_nodes = len(self.data_manager.X_train.columns) if hasattr(self.data_manager.X_train, 'columns') else 64
            return np.eye(num_nodes, dtype=np.float32)
    
    def evaluate_models(self) -> Dict[str, Any]:
        """
        Evaluate all trained models.
        
        Returns:
            Evaluation results
        """
        if not self.models:
            logger.warning("No models to evaluate")
            return {'status': 'no_models'}
        
        results = {}
        
        for model_name, trainer in self.models.items():
            try:
                logger.info(f"\nEvaluating {model_name}...")
                
                # Get config
                model_config = self.config.models[model_name]
                
                # Different data formats for different models
                if model_name == 'lstm':
                    # LSTM expects tabular data
                    X_test_data = self.data_manager.X_test
                    y_test_data = self.data_manager.y_test
                else:
                    # ASTGCN and others expect sequences
                    sequence_length = model_config.sequence_length
                    X_test_seq, y_test_seq = self.data_manager.create_sequences(
                        self.data_manager.X_test,
                        self.data_manager.y_test,
                        sequence_length
                    )
                    X_test_data = X_test_seq
                    y_test_data = y_test_seq
                
                # Evaluate
                metrics = trainer.evaluate(X_test_data, y_test_data, set_name='test')
                
                results[model_name] = metrics
                
                logger.info(f"✓ {model_name}:")
                for metric_name, value in metrics.items():
                    logger.info(f"  {metric_name.upper()}: {value:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to evaluate {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        self.evaluation_results = results
        return results
    
    def save_models(self) -> Dict[str, Any]:
        """
        Save all trained models.
        
        Returns:
            Save results
        """
        if not self.models:
            logger.warning("No models to save")
            return {'status': 'no_models'}
        
        results = {}
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for model_name, trainer in self.models.items():
            try:
                # Save model
                filename = f"{model_name}_{timestamp}"
                model_path = trainer.save_model(filename=filename)
                
                # Save additional metadata
                metadata = {
                    'model_name': model_name,
                    'timestamp': timestamp,
                    'config': self.config.models[model_name].to_dict(),
                    'feature_names': self.data_manager.feature_names,
                    'evaluation': self.evaluation_results.get(model_name, {}),
                    'training_history': {
                        k: [float(v) for v in vals] if isinstance(vals, (list, np.ndarray)) else vals
                        for k, vals in self.training_history.get(model_name, {}).items()
                    } if model_name in self.training_history else {}
                }
                
                metadata_path = self.config.models_dir / f"{model_name}_{timestamp}_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                results[model_name] = {
                    'status': 'success',
                    'model_path': str(model_path),
                    'metadata_path': str(metadata_path)
                }
                
                logger.info(f"✓ Saved {model_name} to {model_path}")
                
            except Exception as e:
                logger.error(f"Failed to save {model_name}: {e}")
                results[model_name] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        return results
    
    def predict(
        self,
        model_name: str,
        X: pd.DataFrame,
        sequence_length: Optional[int] = None
    ) -> np.ndarray:
        """
        Make predictions using a trained model.
        
        Args:
            model_name: Name of model to use
            X: Input features
            sequence_length: Sequence length (uses config if None)
            
        Returns:
            Predictions array
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")
        
        trainer = self.models[model_name]
        
        # Use config sequence length if not provided
        if sequence_length is None:
            sequence_length = self.config.models[model_name].sequence_length
        
        # Create sequences (without target)
        X_values = X.values
        X_seq = []
        
        for i in range(len(X) - sequence_length + 1):
            X_seq.append(X_values[i:i+sequence_length])
        
        X_seq = np.array(X_seq)
        
        # Predict
        predictions = trainer.predict(X_seq)
        
        return predictions.flatten()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get pipeline execution summary."""
        summary = {
            'config': self.config.to_dict(),
            'data': self.data_manager.get_summary(),
            'models': {
                name: {
                    'trained': True,
                    'evaluation': self.evaluation_results.get(name, {})
                }
                for name in self.models.keys()
            }
        }
        
        return summary
    
    def save_summary(self, output_path: Optional[Path] = None):
        """Save pipeline summary to JSON."""
        summary = self.get_summary()
        
        output_path = output_path or (
            self.config.models_dir / f"pipeline_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"✓ Saved summary to {output_path}")
        return output_path
