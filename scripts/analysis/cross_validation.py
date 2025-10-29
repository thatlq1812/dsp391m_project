"""
Cross-Validation Script for Traffic Forecast Model
Performs k-fold cross-validation with multiple metrics
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import cross_val_score, cross_validate, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / 'models'
RESULTS_DIR = PROJECT_ROOT / 'data' / 'cv_results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_training_data():
    """Load and prepare training data"""
    # Find latest data directory
    data_dirs = sorted((PROJECT_ROOT / 'data' / 'downloads').glob('download_*'))
    if not data_dirs:
        raise FileNotFoundError("No data directories found")
    
    latest_dir = data_dirs[-1]
    normalized_file = latest_dir / 'normalized_traffic.json'
    
    if not normalized_file.exists():
        raise FileNotFoundError(f"Normalized data not found: {normalized_file}")
    
    with open(normalized_file, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    print(f"Loaded data: {len(df)} records from {latest_dir.name}")
    
    return df, latest_dir


def prepare_features(df):
    """Prepare features and target"""
    # Target
    if 'speed_kmh' not in df.columns:
        raise ValueError("Target 'speed_kmh' not found in dataframe")
    
    y = df['speed_kmh'].values
    
    # Features - exclude target and metadata columns
    exclude_cols = ['speed_kmh', 'timestamp', 'node_id', 'edge_id']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    
    # Handle categorical variables if any
    X = pd.get_dummies(X, drop_first=True)
    
    # Fill NaN
    X = X.fillna(0)
    
    print(f"Features: {X.shape[1]} columns")
    print(f"Target: {len(y)} samples")
    print(f"Target range: {y.min():.2f} - {y.max():.2f} km/h")
    
    return X, y, feature_cols


def perform_cv_xgboost(X, y, n_folds=5):
    """Perform cross-validation with XGBoost"""
    print(f"\n{'='*60}")
    print(f"CROSS-VALIDATION: XGBoost ({n_folds}-Fold)")
    print(f"{'='*60}")
    
    # Initialize model with same params as trained model
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        random_state=42,
        n_jobs=-1
    )
    
    # Define scoring metrics
    scoring = {
        'r2': 'r2',
        'neg_mse': 'neg_mean_squared_error',
        'neg_mae': 'neg_mean_absolute_error'
    }
    
    # Perform cross-validation
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    print(f"\nRunning {n_folds}-fold cross-validation...")
    cv_results = cross_validate(model, X, y, cv=kfold, scoring=scoring, 
                                 return_train_score=True, n_jobs=-1)
    
    # Calculate metrics
    results = {
        'model': 'XGBoost',
        'n_folds': n_folds,
        'train_r2': cv_results['train_r2'].mean(),
        'train_r2_std': cv_results['train_r2'].std(),
        'test_r2': cv_results['test_r2'].mean(),
        'test_r2_std': cv_results['test_r2'].std(),
        'train_rmse': np.sqrt(-cv_results['train_neg_mse']).mean(),
        'train_rmse_std': np.sqrt(-cv_results['train_neg_mse']).std(),
        'test_rmse': np.sqrt(-cv_results['test_neg_mse']).mean(),
        'test_rmse_std': np.sqrt(-cv_results['test_neg_mse']).std(),
        'train_mae': -cv_results['train_neg_mae'].mean(),
        'train_mae_std': -cv_results['train_neg_mae'].std(),
        'test_mae': -cv_results['test_neg_mae'].mean(),
        'test_mae_std': -cv_results['test_neg_mae'].std(),
        'fold_scores': cv_results['test_r2'].tolist()
    }
    
    return results


def print_cv_results(results):
    """Print cross-validation results"""
    print(f"\n{'='*60}")
    print(f"CROSS-VALIDATION RESULTS: {results['model']}")
    print(f"{'='*60}")
    
    print(f"\n📊 Training Set Performance:")
    print(f"   R² Score:  {results['train_r2']:.4f} ± {results['train_r2_std']:.4f}")
    print(f"   RMSE:      {results['train_rmse']:.4f} ± {results['train_rmse_std']:.4f} km/h")
    print(f"   MAE:       {results['train_mae']:.4f} ± {results['train_mae_std']:.4f} km/h")
    
    print(f"\n📊 Test Set Performance ({results['n_folds']}-Fold CV):")
    print(f"   R² Score:  {results['test_r2']:.4f} ± {results['test_r2_std']:.4f}")
    print(f"   RMSE:      {results['test_rmse']:.4f} ± {results['test_rmse_std']:.4f} km/h")
    print(f"   MAE:       {results['test_mae']:.4f} ± {results['test_mae_std']:.4f} km/h")
    
    print(f"\n📈 Per-Fold R² Scores:")
    for i, score in enumerate(results['fold_scores'], 1):
        print(f"   Fold {i}: {score:.4f}")
    
    # Check for overfitting
    overfitting = results['train_r2'] - results['test_r2']
    print(f"\n🔍 Overfitting Analysis:")
    print(f"   Train-Test Gap (R²): {overfitting:.4f}")
    
    if overfitting < 0.05:
        print(f"   ✅ Good generalization (gap < 0.05)")
    elif overfitting < 0.10:
        print(f"   ⚠️  Slight overfitting (gap < 0.10)")
    else:
        print(f"   ❌ Significant overfitting (gap >= 0.10)")


def save_cv_results(results, output_dir):
    """Save cross-validation results"""
    # Save as JSON
    json_file = output_dir / 'cv_results.json'
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Saved JSON: {json_file}")
    
    # Save as markdown
    md_file = output_dir / 'cv_results.md'
    with open(md_file, 'w') as f:
        f.write(f"# Cross-Validation Results\n\n")
        f.write(f"**Model:** {results['model']}\n")
        f.write(f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Number of Folds:** {results['n_folds']}\n\n")
        
        f.write(f"## Performance Metrics\n\n")
        f.write(f"### Training Set\n\n")
        f.write(f"| Metric | Mean | Std Dev |\n")
        f.write(f"|--------|------|----------|\n")
        f.write(f"| R² Score | {results['train_r2']:.4f} | {results['train_r2_std']:.4f} |\n")
        f.write(f"| RMSE (km/h) | {results['train_rmse']:.4f} | {results['train_rmse_std']:.4f} |\n")
        f.write(f"| MAE (km/h) | {results['train_mae']:.4f} | {results['train_mae_std']:.4f} |\n\n")
        
        f.write(f"### Test Set ({results['n_folds']}-Fold CV)\n\n")
        f.write(f"| Metric | Mean | Std Dev |\n")
        f.write(f"|--------|------|----------|\n")
        f.write(f"| R² Score | {results['test_r2']:.4f} | {results['test_r2_std']:.4f} |\n")
        f.write(f"| RMSE (km/h) | {results['test_rmse']:.4f} | {results['test_rmse_std']:.4f} |\n")
        f.write(f"| MAE (km/h) | {results['test_mae']:.4f} | {results['test_mae_std']:.4f} |\n\n")
        
        f.write(f"## Per-Fold Performance\n\n")
        f.write(f"| Fold | R² Score |\n")
        f.write(f"|------|----------|\n")
        for i, score in enumerate(results['fold_scores'], 1):
            f.write(f"| {i} | {score:.4f} |\n")
        
        f.write(f"\n## Overfitting Analysis\n\n")
        overfitting = results['train_r2'] - results['test_r2']
        f.write(f"**Train-Test Gap (R²):** {overfitting:.4f}\n\n")
        
        if overfitting < 0.05:
            f.write(f"✅ **Good generalization** (gap < 0.05)\n")
        elif overfitting < 0.10:
            f.write(f"⚠️ **Slight overfitting** (gap < 0.10)\n")
        else:
            f.write(f"❌ **Significant overfitting** (gap >= 0.10)\n")
    
    print(f"💾 Saved Markdown: {md_file}")


def main():
    """Main execution"""
    print("=" * 60)
    print("CROSS-VALIDATION FOR TRAFFIC FORECAST MODEL")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    df, data_dir = load_training_data()
    
    # Prepare features
    print("\nPreparing features...")
    X, y, feature_cols = prepare_features(df)
    
    # Perform cross-validation
    cv_results = perform_cv_xgboost(X, y, n_folds=5)
    
    # Print results
    print_cv_results(cv_results)
    
    # Save results
    print(f"\n{'='*60}")
    print("SAVING RESULTS")
    print(f"{'='*60}")
    save_cv_results(cv_results, RESULTS_DIR)
    
    print(f"\n{'='*60}")
    print("✅ CROSS-VALIDATION COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
