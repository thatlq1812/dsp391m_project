# Traffic Forecast ML Pipeline

Production-ready machine learning pipeline for traffic speed prediction with **8 algorithms** including deep learning.

## Quick Start

```python
from traffic_forecast.ml import (
    DataLoader,
    build_features,
    split_data,
    prepare_features_target,
    DataPreprocessor,
    ModelTrainer,
    DLModelTrainer,  # Deep Learning
    HAS_DL
)

# 1. Load data
loader = DataLoader()
df = loader.load_merged_data(0)  # Latest run

# 2. Engineer features
df_features = build_features(df, include_temporal=True, include_weather=True)

# 3. Split data
train_df, val_df, test_df = split_data(df_features, test_size=0.2, val_size=0.1)

# 4. Prepare features
X_train, y_train = prepare_features_target(train_df, target_column='speed_kmh')
X_test, y_test = prepare_features_target(test_df, target_column='speed_kmh')

# 5. Preprocess
preprocessor = DataPreprocessor(scaler_type='standard')
preprocessor.fit(train_df, feature_columns=list(X_train.columns))
X_train_scaled = preprocessor.transform(train_df)[X_train.columns]
X_test_scaled = preprocessor.transform(test_df)[X_test.columns]

# 6a. Train traditional ML model
trainer = ModelTrainer(model_type='xgboost')
trainer.train(X_train_scaled, y_train)
metrics = trainer.evaluate(X_test_scaled, y_test)
print(f"XGBoost Test R²: {metrics['r2']:.4f}")

# 6b. Train deep learning model (if TensorFlow available)
if HAS_DL:
    dl_trainer = DLModelTrainer(model_type='lstm')
    dl_trainer.train(X_train_scaled, y_train, epochs=30, batch_size=32)
    dl_metrics = dl_trainer.evaluate(X_test_scaled, y_test)
    print(f"LSTM Test R²: {dl_metrics['r2']:.4f}")
```

## Features

### 🔄 Data Loading

- Load from collection runs automatically
- Merge traffic, weather, and node data
- Handle multiple runs
- Data quality checks

### 🎯 Feature Engineering

- **Temporal:** Hour, day, weekend, rush hour, cyclical encoding
- **Spatial:** Coordinate differences, distances
- **Weather:** Rain indicator, categories, severity score
- **Traffic:** Speed categories, congestion indicator
- **Time Series:** Lags, rolling windows (optional)

### 🧹 Preprocessing

- Missing value imputation
- Outlier detection and removal
- Feature scaling (Standard, Robust)
- Train/val/test splitting (random or time-based)

### 🤖 Model Training

- **6 Algorithms:** Random Forest, Gradient Boosting, XGBoost, LightGBM, Ridge, Lasso
- Cross-validation support
- Hyperparameter tuning (Grid Search)
- Feature importance analysis
- Model persistence (save/load)

### 📊 Evaluation

- RMSE, MAE, R², MAPE metrics
- Prediction vs actual plots
- Residuals analysis
- Model comparison

## Available Models

| Model             | Type            | Speed  | Accuracy   | Best For                   |
| ----------------- | --------------- | ------ | ---------- | -------------------------- |
| Ridge             | Linear          | ⚡⚡⚡ | ⭐⭐       | Baseline                   |
| Lasso             | Linear          | ⚡⚡⚡ | ⭐⭐       | Feature selection          |
| Random Forest     | Ensemble        | ⚡⚡   | ⭐⭐⭐⭐   | General purpose            |
| Gradient Boosting | Ensemble        | ⚡     | ⭐⭐⭐⭐   | High accuracy              |
| **XGBoost**       | Gradient Boost  | ⚡⚡   | ⭐⭐⭐⭐⭐ | **Best overall (tabular)** |
| **LightGBM**      | Gradient Boost  | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | **Fastest**                |
| **LSTM**          | Deep Learning   | ⚡     | ⭐⭐⭐⭐   | **Time series**            |
| ASTGCN\*          | Graph Neural NN | ⚡     | ⭐⭐⭐⭐⭐ | **Graph + temporal**       |

\*ASTGCN requires graph-structured data (adjacency matrix). Use LSTM for tabular data.

## Jupyter Notebook

Use `notebooks/ML_TRAINING.ipynb` for interactive workflow:

1. Data loading and EDA
2. Feature engineering with visualization
3. Model training and comparison
4. Hyperparameter tuning
5. Model evaluation and saving

## Module Structure

```
traffic_forecast/ml/
├── __init__.py           # Public API exports
├── data_loader.py        # DataLoader class
├── preprocess.py         # DataPreprocessor, utilities
├── features.py           # Feature engineering functions
└── trainer.py            # ModelTrainer, compare_models
```

## API Reference

### DataLoader

```python
loader = DataLoader(data_dir=None)  # Auto-detect or specify path

# Methods
loader.list_runs()                    # List available runs
loader.load_traffic_data(run_idx=0)  # Load traffic edges
loader.load_weather_data(run_idx=0)  # Load weather
loader.load_nodes_data(run_idx=0)    # Load nodes
loader.load_merged_data(run_idx=0)   # Load merged (recommended)
loader.load_multiple_runs([0,1,2])   # Load multiple runs
loader.get_data_summary()             # Get statistics
```

### Feature Engineering

```python
# Build all features
df_features = build_features(
    df,
    include_temporal=True,
    include_spatial=True,
    include_weather=True,
    include_traffic=True,
    include_lags=False,
    include_rolling=False
)

# Individual functions
add_temporal_features(df)
add_spatial_features(df)
add_weather_features(df)
add_traffic_features(df)
add_lag_features(df, lags=[1,2,3])
add_rolling_features(df, windows=[3,5,10])
```

### DataPreprocessor

```python
preprocessor = DataPreprocessor(
    target_column='speed_kmh',
    scaler_type='standard',  # 'standard', 'robust', 'none'
    handle_outliers=True,
    outlier_std=3.0
)

preprocessor.fit(train_df, feature_columns=feature_list)
train_scaled = preprocessor.transform(train_df)

# Or combined
train_scaled = preprocessor.fit_transform(train_df)

# Static utilities
DataPreprocessor.remove_outliers(df, std_threshold=3.0)
DataPreprocessor.handle_missing_values(df, strategy='median')
DataPreprocessor.clip_values(df, 'speed_kmh', lower=0, upper=120)
```

### Data Splitting

```python
train_df, val_df, test_df = split_data(
    df,
    target_column='speed_kmh',
    test_size=0.2,
    val_size=0.1,
    random_state=42,
    time_based=False  # True for time-series
)

X, y = prepare_features_target(
    df,
    target_column='speed_kmh',
    drop_columns=['custom_column']  # Optional
)
```

### ModelTrainer

```python
trainer = ModelTrainer(
    model_type='xgboost',  # See available models
    params={...},          # Model hyperparameters
    model_dir=None         # Auto: PROJECT_ROOT/models
)

# Training
trainer.train(X_train, y_train, X_val, y_val)
metrics = trainer.evaluate(X_test, y_test)
predictions = trainer.predict(X_new)

# Cross-validation
cv_results = trainer.cross_validate(X, y, cv=5, scoring='r2')

# Hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15]
}
tuning_results = trainer.tune_hyperparameters(
    X_train, y_train,
    param_grid=param_grid,
    cv=3
)

# Feature importance
importance_df = trainer.get_feature_importance(top_n=20)

# Model persistence
model_path = trainer.save_model('my_model.joblib')
loaded_trainer = ModelTrainer.load_model(model_path)

# Compare multiple models
comparison_df = compare_models(
    X_train, y_train,
    X_test, y_test,
    models=['random_forest', 'xgboost', 'lightgbm']
)
```

## Performance Tips

### For Speed

1. Use `lightgbm` (fastest)
2. Reduce `n_estimators`
3. Use `n_jobs=-1` for parallelization
4. Sample large datasets

### For Accuracy

1. Use `xgboost` or `lightgbm`
2. Increase `n_estimators`
3. Tune hyperparameters with grid search
4. Create more features
5. Collect more data

### For Production

1. Use cross-validation to estimate performance
2. Save preprocessing pipeline with model
3. Monitor prediction distribution
4. Retrain periodically
5. Version models with timestamps

## Common Issues

### Low R² score

- Collect more data
- Engineer better features
- Try different models
- Check data quality

### Overfitting (high train, low test R²)

- Reduce model complexity
- Use cross-validation
- Add regularization
- Collect more data

### Memory errors

- Reduce dataset size
- Use sampling
- Use LightGBM (memory efficient)
- Process in batches

## Documentation

- **Full Guide:** `doc/getting-started/ML_PIPELINE.md`
- **Notebook:** `notebooks/ML_TRAINING.ipynb`
- **API Docs:** Inline docstrings

## Examples

### Example 1: Quick Training

```python
from traffic_forecast.ml import load_latest_data, build_features, ModelTrainer, split_data, prepare_features_target

df = load_latest_data()
df = build_features(df)
train_df, _, test_df = split_data(df)
X_train, y_train = prepare_features_target(train_df)
X_test, y_test = prepare_features_target(test_df)

trainer = ModelTrainer('xgboost')
trainer.train(X_train, y_train)
print(trainer.evaluate(X_test, y_test))
```

### Example 2: Model Comparison

```python
from traffic_forecast.ml import compare_models

comparison = compare_models(
    X_train, y_train,
    X_test, y_test,
    models=['random_forest', 'xgboost', 'lightgbm']
)
print(comparison.sort_values('r2', ascending=False))
```

### Example 3: Production Pipeline

```python
# Train
trainer = ModelTrainer('lightgbm', params={'n_estimators': 200})
trainer.train(X_train, y_train)
model_path = trainer.save_model()

# Deploy
loaded_trainer = ModelTrainer.load_model(model_path)
predictions = loaded_trainer.predict(X_new)
```

## Requirements

```bash
# Install ML dependencies
pip install xgboost lightgbm scikit-learn pandas numpy matplotlib seaborn
```

## Contributing

To add a custom model:

```python
from sklearn.ensemble import ExtraTreesRegressor

# Register in ModelTrainer.MODELS
ModelTrainer.MODELS['extra_trees'] = ExtraTreesRegressor
ModelTrainer.DEFAULT_PARAMS['extra_trees'] = {
    'n_estimators': 100,
    'random_state': 42
}

# Use it
trainer = ModelTrainer(model_type='extra_trees')
```

## License

MIT License - See project root for details.

---

**Author:** THAT Le Quang  
**Version:** 4.4.0  
**Updated:** October 27, 2025
