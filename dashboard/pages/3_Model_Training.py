"""
Model Training Page

Train and compare LSTM and ASTGCN deep learning models.

Author: thatlq1812
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import json

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from traffic_forecast.core import PipelineConfig, DataManager, TrafficForecastPipeline

st.set_page_config(page_title="Model Training", page_icon="🤖", layout="wide")

st.title("Deep Learning Model Training")
st.markdown("Train LSTM and ASTGCN models for traffic speed forecasting")

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'training_results' not in st.session_state:
    st.session_state.training_results = None

# Tabs
tab1, tab2, tab3 = st.tabs(["Configuration", "Training", "Results"])

with tab1:
    st.header("Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("LSTM Configuration")
        
        lstm_enabled = st.checkbox("Enable LSTM", value=True)
        
        with st.expander("LSTM Parameters", expanded=True):
            lstm_epochs = st.slider("Epochs", 10, 100, 50)
            lstm_batch_size = st.select_slider("Batch Size", options=[16, 32, 64, 128], value=32)
            lstm_lr = st.select_slider("Learning Rate", options=[0.0001, 0.001, 0.01], value=0.001, format_func=lambda x: f"{x:.4f}")
            lstm_sequence_length = st.slider("Sequence Length", 6, 24, 12)
            
            st.markdown(f"""
            **LSTM Architecture:**
            - Hidden Units: [64, 64]
            - Dropout: 0.2
            - Output: Dense(1)
            """)
    
    with col2:
        st.subheader("ASTGCN Configuration")
        
        astgcn_enabled = st.checkbox("Enable ASTGCN", value=True)
        
        with st.expander("ASTGCN Parameters", expanded=True):
            astgcn_epochs = st.slider("Epochs", 10, 100, 50, key="astgcn_epochs")
            astgcn_batch_size = st.select_slider("Batch Size", options=[16, 32, 64, 128], value=32, key="astgcn_batch")
            astgcn_lr = st.select_slider("Learning Rate", options=[0.0001, 0.001, 0.01], value=0.001, format_func=lambda x: f"{x:.4f}", key="astgcn_lr")
            astgcn_sequence_length = st.slider("Sequence Length", 6, 24, 12, key="astgcn_seq")
            
            st.markdown(f"""
            **ASTGCN Architecture:**
            - Hidden Units: [64, 64, 64]
            - Dropout: 0.2
            - Graph Convolution: Spatial-Temporal
            """)
    
    st.markdown("---")
    
    # Data options
    st.subheader("Data Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        use_preprocessed = st.checkbox("Use Preprocessed Data", value=True)
    
    with col2:
        test_size = st.slider("Test Size", 0.1, 0.3, 0.2)
    
    with col3:
        validation_split = st.slider("Validation Split", 0.1, 0.3, 0.2)
    
    # Save config button
    if st.button("Save Configuration", type="secondary"):
        config_dict = {
            'lstm': {
                'enabled': lstm_enabled,
                'epochs': lstm_epochs,
                'batch_size': lstm_batch_size,
                'learning_rate': lstm_lr,
                'sequence_length': lstm_sequence_length
            },
            'astgcn': {
                'enabled': astgcn_enabled,
                'epochs': astgcn_epochs,
                'batch_size': astgcn_batch_size,
                'learning_rate': astgcn_lr,
                'sequence_length': astgcn_sequence_length
            },
            'data': {
                'use_preprocessed': use_preprocessed,
                'test_size': test_size,
                'validation_split': validation_split
            }
        }
        
        config_path = PROJECT_ROOT / 'configs' / 'training_config.json'
        config_path.parent.mkdir(exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        st.success(f"Configuration saved to {config_path}")

with tab2:
    st.header("Model Training")
    
    # Training mode
    training_mode = st.radio(
        "Training Mode:",
        ["Train Selected Models", "Train All Models", "Quick Test (1 epoch)"],
        horizontal=True
    )
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("""
        ### Training Pipeline:
        
        1. **Load Data** - Load preprocessed data or process from raw
        2. **Create Sequences** - Convert to time-series sequences
        3. **Split Data** - Train/validation/test split
        4. **Train Models** - Train enabled models with configured parameters
        5. **Evaluate** - Calculate metrics on test set
        6. **Save Models** - Export trained models and metadata
        """)
    
    with col2:
        st.info(f"""
        **Models to Train:**
        - LSTM: {'Yes' if lstm_enabled else 'No'}
        - ASTGCN: {'Yes' if astgcn_enabled else 'No'}
        """)
    
    if st.button("Start Training", type="primary", use_container_width=True):
        # Create pipeline config
        config = PipelineConfig()
        config.use_preprocessed = use_preprocessed
        config.test_size = test_size
        
        # Update model configs
        from traffic_forecast.core.config import ModelConfig
        
        config.models['lstm'] = ModelConfig(
            name='lstm',
            enabled=lstm_enabled,
            epochs=1 if training_mode == "Quick Test (1 epoch)" else lstm_epochs,
            batch_size=lstm_batch_size,
            learning_rate=lstm_lr,
            sequence_length=lstm_sequence_length,
            validation_split=validation_split
        )
        
        config.models['astgcn'] = ModelConfig(
            name='astgcn',
            enabled=astgcn_enabled,
            epochs=1 if training_mode == "Quick Test (1 epoch)" else astgcn_epochs,
            batch_size=astgcn_batch_size,
            learning_rate=astgcn_lr,
            sequence_length=astgcn_sequence_length,
            validation_split=validation_split
        )
        
        # Create pipeline
        pipeline = TrafficForecastPipeline(config)
        
        # Progress tracking
        progress_container = st.container()
        
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Run pipeline
                status_text.text("Loading data...")
                progress_bar.progress(10)
                
                if use_preprocessed:
                    pipeline.data_manager.load_processed_data()
                else:
                    pipeline.data_manager.load_data()
                    pipeline.data_manager.preprocess_data()
                    pipeline.data_manager.engineer_features()
                
                status_text.text("Data loaded")
                progress_bar.progress(20)
                
                # Prepare data
                status_text.text("Preparing train/test split...")
                pipeline.data_manager.prepare_train_test()
                status_text.text("Data split complete")
                progress_bar.progress(30)
                
                # Train models
                status_text.text("Training models...")
                progress_bar.progress(40)
                
                training_results = pipeline.train_models()
                
                progress_bar.progress(70)
                
                # Evaluate
                status_text.text("Evaluating models...")
                evaluation_results = pipeline.evaluate_models()
                
                progress_bar.progress(90)
                
                # Save
                status_text.text("Saving models...")
                save_results = pipeline.save_models()
                
                progress_bar.progress(100)
                status_text.text("Training complete!")
                
                # Store results
                st.session_state.pipeline = pipeline
                st.session_state.training_results = {
                    'training': training_results,
                    'evaluation': evaluation_results,
                    'save': save_results
                }
                
                st.balloons()
                st.success("Training completed successfully!")
                
            except Exception as e:
                st.error(f"Training failed: {e}")
                st.exception(e)

with tab3:
    st.header("Training Results")
    
    if st.session_state.training_results:
        results = st.session_state.training_results
        
        # Evaluation metrics
        st.subheader("Model Performance")
        
        if 'evaluation' in results:
            eval_results = results['evaluation']
            
            # Create comparison DataFrame
            comparison_data = []
            for model_name, metrics in eval_results.items():
                if isinstance(metrics, dict) and 'mse' in metrics:
                    comparison_data.append({
                        'Model': model_name.upper(),
                        'MAE': metrics['mae'],
                        'RMSE': metrics['rmse'],
                        'MSE': metrics['mse']
                    })
            
            if comparison_data:
                df_comparison = pd.DataFrame(comparison_data)
                
                # Display table
                st.dataframe(df_comparison, use_container_width=True)
                
                # Visualize comparison
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    name='MAE',
                    x=df_comparison['Model'],
                    y=df_comparison['MAE'],
                    marker_color='#1f77b4'
                ))
                
                fig.add_trace(go.Bar(
                    name='RMSE',
                    x=df_comparison['Model'],
                    y=df_comparison['RMSE'],
                    marker_color='#ff7f0e'
                ))
                
                fig.update_layout(
                    title='Model Performance Comparison',
                    barmode='group',
                    yaxis_title='Error (km/h)',
                    xaxis_title='Model'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Best model
                best_model = df_comparison.loc[df_comparison['RMSE'].idxmin()]
                st.success(f"""
                **Best Model:** {best_model['Model']}
                - MAE: {best_model['MAE']:.4f} km/h
                - RMSE: {best_model['RMSE']:.4f} km/h
                """)
        
        # Training history
        if st.session_state.pipeline and st.session_state.pipeline.training_history:
            st.subheader("Training History")
            
            for model_name, history in st.session_state.pipeline.training_history.items():
                with st.expander(f"📊 {model_name.upper()} Training Curves", expanded=True):
                    # Create loss plot
                    epochs = list(range(1, len(history['loss']) + 1))
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=epochs,
                        y=history['loss'],
                        mode='lines+markers',
                        name='Training Loss',
                        line=dict(color='#1f77b4')
                    ))
                    
                    if 'val_loss' in history:
                        fig.add_trace(go.Scatter(
                            x=epochs,
                            y=history['val_loss'],
                            mode='lines+markers',
                            name='Validation Loss',
                            line=dict(color='#ff7f0e')
                        ))
                    
                    fig.update_layout(
                        title=f'{model_name.upper()} Training History',
                        xaxis_title='Epoch',
                        yaxis_title='Loss (MSE)',
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        # Saved models
        if 'save' in results and results['save']:
            st.subheader("Saved Models")
            
            for model_name, save_info in results['save'].items():
                if isinstance(save_info, dict) and save_info.get('status') == 'success':
                    st.success(f"""
                    **{model_name.upper()}**
                    - Model: `{save_info.get('model_path', 'N/A')}`
                    - Metadata: `{save_info.get('metadata_path', 'N/A')}`
                    """)
                elif isinstance(save_info, str):
                    st.info(f"**{model_name.upper()}**: {save_info}")
    
    else:
        st.info("Train models first to see results")

# Footer
st.markdown("---")
st.markdown("Deep Learning Training Dashboard | LSTM & ASTGCN Models")
