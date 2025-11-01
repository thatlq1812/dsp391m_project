"""
Page 4: Training Control
Start, monitor, and control STMGT model training
"""

import streamlit as st
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
import subprocess
import sys
import os
from datetime import datetime
import time

st.set_page_config(page_title="Training Control", page_icon="🎮", layout="wide")

st.title("Training Control")
st.markdown("Train and monitor STMGT models")

PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_CONDA_PATH = Path("C:/ProgramData/miniconda3/Scripts/conda.exe")


def _conda_executable() -> str:
    """Resolve conda executable path with Windows fallback."""
    env_path = os.environ.get("CONDA_EXE")
    if env_path:
        candidate = Path(env_path)
        if candidate.exists():
            return str(candidate)
    if DEFAULT_CONDA_PATH.exists():
        return str(DEFAULT_CONDA_PATH)
    return "conda"


def _training_command(config_path: Path) -> list[str]:
    """Build training launch command using the dsp environment."""
    return [
        _conda_executable(),
        "run",
        "-n",
        "dsp",
        "--no-capture-output",
        "python",
        "scripts/training/train_stmgt.py",
        "--config",
        str(config_path),
    ]

# Initialize session state
if 'training_process' not in st.session_state:
    st.session_state.training_process = None
if 'training_active' not in st.session_state:
    st.session_state.training_active = False

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Training Control",
    "Hyperparameter Tuning",
    "Monitor Progress",
    "Model Comparison",
    "Export Reports"
])

with tab1:
    st.markdown("### Training Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Model Architecture")
        
        hidden_dim = st.slider("Hidden Dimension", 32, 128, 64, 16)
        num_heads = st.slider("Attention Heads", 2, 8, 4, 2)
        num_blocks = st.slider("ST Blocks", 1, 4, 2, 1)
        mixture_components = st.slider("GMM Components (K)", 2, 5, 3, 1)
        
        st.info(f"""
        **Model Size:** ~{(hidden_dim * hidden_dim * num_blocks * 4 + hidden_dim * 62 * 3) / 1000:.1f}K parameters
        
        **Architecture:**
        - Hidden Dim: {hidden_dim}
        - Heads: {num_heads}
        - Blocks: {num_blocks}
        - GMM K: {mixture_components}
        """)
    
    with col2:
        st.markdown("#### Training Hyperparameters")
        
        batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
        learning_rate = st.selectbox("Learning Rate", [0.0001, 0.0005, 0.001, 0.005], index=2)
        # Max epochs presets with custom option
        max_epoch_choice = st.selectbox("Max Epochs", [10, 20, 50, 100, 150, 200, "Custom"], index=3)
        if max_epoch_choice == "Custom":
            max_epochs = int(st.number_input("Custom Max Epochs", min_value=1, max_value=1000, value=100, step=1))
        else:
            max_epochs = int(max_epoch_choice)
        patience = st.slider("Early Stopping Patience", 10, 50, 20, 5)
        
        st.markdown("#### Data Source")
        data_source = st.radio(
            "Training Data",
            ["Extreme Augmented (Recommended)", "Basic Augmented", "Original Combined"],
            index=0
        )
        
        data_map = {
            "Extreme Augmented (Recommended)": "all_runs_extreme_augmented.parquet",
            "Basic Augmented": "all_runs_augmented.parquet",
            "Original Combined": "all_runs_combined.parquet"
        }
    
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Start Training", width='stretch', type="primary"):
            with st.status("Initializing training...", expanded=True) as status:
                try:
                    # Create config
                    st.write("Creating training configuration...")
                    config = {
                        "model": {
                            "num_nodes": 62,
                            "hidden_dim": hidden_dim,
                            "num_heads": num_heads,
                            "num_blocks": num_blocks,
                            "mixture_components": mixture_components,
                            "seq_len": 12,
                            "pred_len": 12
                        },
                        "training": {
                            "batch_size": batch_size,
                            "learning_rate": learning_rate,
                            "max_epochs": max_epochs,
                            "patience": patience,
                            "weight_decay": 1e-4,
                            "drop_edge_p": 0.2,
                            "num_workers": 0,
                            "use_amp": True,
                            "accumulation_steps": 1,
                            "data_source": data_map[data_source]
                        }
                    }
                    
                    # Save config
                    config_dir = PROJECT_ROOT / "configs"
                    config_dir.mkdir(exist_ok=True)
                    config_path = config_dir / f"train_config_{datetime.now():%Y%m%d_%H%M%S}.json"
                    
                    with open(config_path, 'w') as f:
                        json.dump(config, f, indent=2)
                    
                    st.write(f"Config saved to `{config_path.name}`")
                    st.write("Starting training process...")
                    
                    # Start training
                    command = _training_command(config_path)
                    st.write("Launching command:")
                    st.code(" ".join(command), language="bash")

                    process = subprocess.Popen(
                        command,
                        cwd=PROJECT_ROOT,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True
                    )
                    
                    st.session_state.training_process = process
                    st.session_state.training_active = True
                    
                    st.write(f"Training process started (PID: {process.pid})")
                    st.write("\n**Initial output:**")
                    
                    # Show first few lines
                    for i, line in enumerate(process.stdout):
                        if i < 5:
                            st.code(line.strip())
                        else:
                            break
                    
                    status.update(label="Training started successfully!", state="complete")
                    st.success("Training is running! Go to 'Monitor Progress' tab to track.")
                    st.info("Training runs in background. You can close this page.")
                    
                except Exception as e:
                    status.update(label="Failed to start training", state="error")
                    st.error(f"Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    
    with col2:
        if st.button("Stop Training", width='stretch', type="secondary"):
            if st.session_state.training_process:
                st.session_state.training_process.terminate()
                st.session_state.training_active = False
                st.warning("Training stopped")
            else:
                st.info("No active training process")
    
    with col3:
        if st.button("Save Custom Config", width='stretch'):
            config = {
                "model": {
                    "num_nodes": 62,
                    "hidden_dim": hidden_dim,
                    "num_heads": num_heads,
                    "num_blocks": num_blocks,
                    "mixture_components": mixture_components,
                    "seq_len": 12,
                    "pred_len": 12
                },
                "training": {
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "max_epochs": max_epochs,
                    "patience": patience,
                    "weight_decay": 1e-4,
                    "drop_edge_p": 0.2,
                    "num_workers": 0,
                    "use_amp": True,
                    "accumulation_steps": 1,
                    "data_source": data_map[data_source]
                }
            }
            
            config_dir = PROJECT_ROOT / "configs"
            config_dir.mkdir(exist_ok=True)
            config_path = config_dir / "custom_training_config.json"
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            st.success(f"Configuration saved to {config_path}")
    
    # Training status
    st.divider()
    st.markdown("### Training Status")
    
    if st.session_state.training_active:
        st.success("Training is active")
    else:
        st.info("No active training")

with tab2:
    st.markdown("### Hyperparameter Tuning")
    
    st.info("""
    Configure grid search or random search for hyperparameter optimization.
    Launch multiple training runs with different configurations.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Search Strategy")
        search_type = st.radio("Search Type", ["Grid Search", "Random Search"])
        
        if search_type == "Grid Search":
            st.markdown("**Parameter Grids:**")
            
            batch_sizes = st.multiselect("Batch Sizes", [16, 32, 64, 128], default=[32, 64])
            learning_rates = st.multiselect("Learning Rates", [0.0001, 0.0005, 0.001, 0.005], default=[0.001, 0.0005])
            hidden_dims = st.multiselect("Hidden Dimensions", [32, 64, 96, 128], default=[64, 96])
            
            total_combos = len(batch_sizes) * len(learning_rates) * len(hidden_dims)
            st.metric("Total Combinations", total_combos)
            
            if total_combos > 20:
                st.warning("WARNING Large search space! Consider reducing combinations.")
        
        else:
            st.markdown("**Random Search Parameters:**")
            n_trials = st.slider("Number of Trials", 5, 50, 10, 5)
            
            st.info(f"Will randomly sample {n_trials} configurations")
    
    with col2:
        st.markdown("#### Parameter Ranges")
        
        st.markdown("**Batch Size:**")
        st.write("Range: [16, 32, 64, 128]")
        
        st.markdown("**Learning Rate:**")
        st.write("Range: [0.0001, 0.0005, 0.001, 0.005]")
        
        st.markdown("**Hidden Dimension:**")
        st.write("Range: [32, 64, 96, 128]")
        
        st.markdown("**Attention Heads:**")
        st.write("Range: [2, 4, 8]")
    
    st.divider()
    
    if st.button("Launch Hyperparameter Search", width='stretch', type="primary"):
        st.warning("WARNING Batch experiment management not yet implemented")
        st.info("Coming soon: Automated hyperparameter tuning with MLflow tracking")

with tab3:
    st.markdown("### Monitor Training Progress")
    
    # Auto refresh
    auto_refresh = st.checkbox("Auto-refresh every 30s", value=False)
    
    if auto_refresh:
        time.sleep(30)
        st.rerun()
    
    # Find training runs
    outputs_dir = PROJECT_ROOT / "outputs"
    
    if not outputs_dir.exists():
        st.warning("No training outputs found. Start training first.")
    else:
        run_dirs = sorted([d for d in outputs_dir.iterdir() if d.is_dir() and d.name.startswith("stmgt")],
                         key=lambda x: x.stat().st_mtime, reverse=True)
        
        if not run_dirs:
            st.warning("No STMGT training runs found")
        else:
            # Select run
            run_names = [d.name for d in run_dirs]
            selected_run = st.selectbox("Select Training Run", run_names, index=0)
            
            run_dir = outputs_dir / selected_run
            
            # Load training history
            history_file = run_dir / "training_history.csv"
            
            if history_file.exists():
                df_history = pd.read_csv(history_file)
                
                # Metrics overview
                col1, col2, col3, col4 = st.columns(4)

                latest = df_history.iloc[-1]
                best_val_mae = None
                best_val_r2 = None
                latest_val_loss = None

                if "val_mae" in df_history.columns:
                    best_val_mae = df_history["val_mae"].min()
                if "val_r2" in df_history.columns:
                    best_val_r2 = df_history["val_r2"].max()
                if "val_loss" in df_history.columns:
                    latest_val_loss = latest["val_loss"]

                with col1:
                    st.metric("Current Epoch", int(latest["epoch"]))

                with col2:
                    if best_val_mae is not None and pd.notna(best_val_mae):
                        st.metric("Best MAE", f"{best_val_mae:.3f} km/h")
                    elif "train_mae" in df_history.columns:
                        st.metric("Best MAE", f"{df_history['train_mae'].min():.3f} km/h")
                    else:
                        st.metric("Best MAE", "N/A")

                with col3:
                    if best_val_r2 is not None and pd.notna(best_val_r2):
                        st.metric("Best R²", f"{best_val_r2:.3f}")
                    elif "train_r2" in df_history.columns:
                        st.metric("Best R²", f"{df_history['train_r2'].max():.3f}")
                    else:
                        st.metric("Best R²", "N/A")

                with col4:
                    if latest_val_loss is not None and pd.notna(latest_val_loss):
                        st.metric("Val Loss", f"{latest_val_loss:.4f}")
                    else:
                        st.metric("Train Loss", f"{latest['train_loss']:.4f}")
                
                st.divider()
                
                # Loss curves
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Training & Validation Loss")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df_history['epoch'],
                        y=df_history['train_loss'],
                        mode='lines',
                        name='Train Loss',
                        line=dict(color='#1f77b4', width=2)
                    ))
                    if "val_loss" in df_history.columns:
                        fig.add_trace(go.Scatter(
                            x=df_history['epoch'],
                            y=df_history['val_loss'],
                            mode='lines',
                            name='Val Loss',
                            line=dict(color='#ff7f0e', width=2)
                        ))
                    fig.update_layout(
                        xaxis_title="Epoch",
                        yaxis_title="Loss (NLL)",
                        height=400
                    )
                    st.plotly_chart(fig, width='stretch')
                
                with col2:
                    st.markdown("#### Mean Absolute Error (MAE)")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df_history['epoch'],
                        y=df_history['train_mae'],
                        mode='lines',
                        name='Train MAE',
                        line=dict(color='#1f77b4', width=2)
                    ))
                    if "val_mae" in df_history.columns:
                        fig.add_trace(go.Scatter(
                            x=df_history['epoch'],
                            y=df_history['val_mae'],
                            mode='lines',
                            name='Val MAE',
                            line=dict(color='#ff7f0e', width=2)
                        ))
                    fig.update_layout(
                        xaxis_title="Epoch",
                        yaxis_title="MAE (km/h)",
                        height=400
                    )
                    st.plotly_chart(fig, width='stretch')
                
                # Additional metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### R² Score")
                    if "val_r2" in df_history.columns:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=df_history['epoch'],
                            y=df_history['val_r2'],
                            mode='lines+markers',
                            name='Val R²',
                            line=dict(color='#2ca02c', width=2),
                            marker=dict(size=6)
                        ))
                        fig.update_layout(
                            xaxis_title="Epoch",
                            yaxis_title="R² Score",
                            height=400
                        )
                        st.plotly_chart(fig, width='stretch')
                    else:
                        st.info("Validation R² not available in training history")
                
                with col2:
                    st.markdown("#### MAPE")
                    if 'val_mape' in df_history.columns:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=df_history['epoch'],
                            y=df_history['val_mape'],
                            mode='lines+markers',
                            name='Val MAPE',
                            line=dict(color='#d62728', width=2),
                            marker=dict(size=6)
                        ))
                        fig.update_layout(
                            xaxis_title="Epoch",
                            yaxis_title="MAPE (%)",
                            height=400
                        )
                        st.plotly_chart(fig, width='stretch')
                    else:
                        st.info("MAPE not available in training history")
                
                # Best metrics summary
                st.divider()
                st.markdown("### Best Metrics Summary")
                
                if "val_mae" in df_history.columns:
                    best_epoch = df_history.loc[df_history['val_mae'].idxmin()]
                    val_loss_display = f"{best_epoch['val_loss']:.4f}" if "val_loss" in best_epoch else "N/A"
                    val_r2_display = f"{best_epoch['val_r2']:.3f}" if "val_r2" in best_epoch else "N/A"
                    val_rmse_display = (
                        f"{best_epoch['val_rmse']:.3f} km/h" if "val_rmse" in best_epoch else "N/A"
                    )
                    metrics_data = {
                        "Metric": [
                            "Epoch",
                            "Train Loss",
                            "Val Loss",
                            "Train MAE",
                            "Val MAE",
                            "Val R²",
                            "Val RMSE",
                        ],
                        "Value": [
                            int(best_epoch['epoch']),
                            f"{best_epoch['train_loss']:.4f}",
                            val_loss_display,
                            f"{best_epoch['train_mae']:.3f} km/h",
                            f"{best_epoch['val_mae']:.3f} km/h",
                            val_r2_display,
                            val_rmse_display,
                        ],
                    }
                else:
                    best_epoch = df_history.loc[df_history['train_loss'].idxmin()]
                    metrics_data = {
                        "Metric": ["Epoch", "Train Loss", "Train MAE"],
                        "Value": [
                            int(best_epoch['epoch']),
                            f"{best_epoch['train_loss']:.4f}",
                            f"{best_epoch.get('train_mae', float('nan')):.3f} km/h"
                            if 'train_mae' in best_epoch
                            else "N/A",
                        ],
                    }

                st.dataframe(metrics_data, hide_index=True, width='stretch')
                
            else:
                st.warning(f"Training history not found for {selected_run}")
                st.info("Training may still be initializing...")

with tab4:
    st.markdown("### Model Comparison")
    
    st.info("Compare metrics across multiple training runs")
    
    outputs_dir = PROJECT_ROOT / "outputs"
    
    if outputs_dir.exists():
        run_dirs = sorted([d for d in outputs_dir.iterdir() if d.is_dir() and d.name.startswith("stmgt")],
                         key=lambda x: x.stat().st_mtime, reverse=True)
        
        if len(run_dirs) >= 2:
            # Select runs to compare
            run_names = [d.name for d in run_dirs]
            selected_runs = st.multiselect("Select Runs to Compare (max 5)", run_names, default=run_names[:2], max_selections=5)
            
            if selected_runs:
                comparison_data = []
                
                for run_name in selected_runs:
                    run_dir = outputs_dir / run_name
                    history_file = run_dir / "training_history.csv"
                    
                    if history_file.exists():
                        df_hist = pd.read_csv(history_file)
                        if 'val_mae' not in df_hist.columns:
                            continue
                        best_epoch = df_hist.loc[df_hist['val_mae'].idxmin()]

                        comparison_data.append({
                            "Run": run_name,
                            "Best Epoch": int(best_epoch['epoch']),
                            "Val MAE": f"{best_epoch['val_mae']:.3f}",
                            "Val R²": f"{best_epoch['val_r2']:.3f}" if 'val_r2' in best_epoch else "N/A",
                            "Val Loss": f"{best_epoch['val_loss']:.4f}" if 'val_loss' in best_epoch else "N/A",
                            "Val MAPE": (
                                f"{best_epoch['val_mape']:.2f}%" if 'val_mape' in best_epoch else "N/A"
                            ),
                        })
                
                if comparison_data:
                    st.dataframe(comparison_data, hide_index=True, width='stretch')
                    
                    # MAE comparison chart
                    st.markdown("#### MAE Comparison")
                    
                    fig = go.Figure()
                    
                    for run_name in selected_runs:
                        run_dir = outputs_dir / run_name
                        history_file = run_dir / "training_history.csv"
                        
                        if history_file.exists():
                            df_hist = pd.read_csv(history_file)
                            if 'val_mae' in df_hist.columns:
                                fig.add_trace(go.Scatter(
                                    x=df_hist['epoch'],
                                    y=df_hist['val_mae'],
                                    mode='lines',
                                    name=run_name[:20] + "..." if len(run_name) > 20 else run_name,
                                    line=dict(width=2)
                                ))
                    
                    fig.update_layout(
                        xaxis_title="Epoch",
                        yaxis_title="Validation MAE (km/h)",
                        height=500
                    )
                    st.plotly_chart(fig, width='stretch')
        else:
            st.info("Need at least 2 training runs for comparison")
    else:
        st.warning("No training outputs found")

with tab5:
    st.markdown("### Export Training Reports")
    
    st.info("Generate HTML reports with comprehensive training metrics and visualizations")
    
    outputs_dir = PROJECT_ROOT / "outputs"
    
    if outputs_dir.exists():
        run_dirs = sorted([d for d in outputs_dir.iterdir() if d.is_dir() and d.name.startswith("stmgt")],
                         key=lambda x: x.stat().st_mtime, reverse=True)
        
        if run_dirs:
            run_names = [d.name for d in run_dirs]
            selected_run = st.selectbox("Select Run for Report", run_names)
            
            run_dir = outputs_dir / selected_run
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Report Contents")
                st.write("- Training configuration")
                st.write("- Best metrics summary")
                st.write("- Complete training history")
                st.write("- Loss curves")
                st.write("- Performance metrics charts")
            
            with col2:
                report_name = st.text_input("Report Filename", f"train_{selected_run}.html")
                
                if st.button("Generate HTML Report", width='stretch', type="primary"):
                    report_dir = PROJECT_ROOT / "docs" / "report"
                    report_dir.mkdir(exist_ok=True, parents=True)
                    report_path = report_dir / report_name
                    
                    # Generate report
                    history_file = run_dir / "training_history.csv"
                    config_file = run_dir / "config.json"
                    
                    if history_file.exists():
                        df_history = pd.read_csv(history_file)

                        if 'val_mae' not in df_history.columns:
                            st.error("Training history does not contain validation metrics yet. Try again later.")
                        else:
                            # Load config if exists
                            config = {}
                            if config_file.exists():
                                with open(config_file, 'r') as f:
                                    config = json.load(f)

                            best_val_mae = df_history['val_mae'].min()
                            best_val_r2 = df_history['val_r2'].max() if 'val_r2' in df_history.columns else None
                            final_val_loss = df_history['val_loss'].iloc[-1] if 'val_loss' in df_history.columns else None

                            best_mae_display = f"{best_val_mae:.3f} km/h"
                            best_r2_display = f"{best_val_r2:.3f}" if best_val_r2 is not None else "N/A"
                            final_loss_display = (
                                f"{final_val_loss:.4f}"
                                if final_val_loss is not None
                                else "N/A"
                            )

                            # Generate HTML
                            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Training Report - {selected_run}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #1f77b4; }}
        h2 {{ color: #333; border-bottom: 2px solid #1f77b4; padding-bottom: 10px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #1f77b4; color: white; }}
        .metric {{ display: inline-block; margin: 10px; padding: 15px; background: #f0f0f0; border-radius: 5px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #1f77b4; }}
    </style>
</head>
<body>
    <h1>Training Report: {selected_run}</h1>
    <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <h2>Best Metrics</h2>
    <div>
        <div class="metric">
            <div>Best MAE</div>
            <div class="metric-value">{best_mae_display}</div>
        </div>
        <div class="metric">
            <div>Best R²</div>
            <div class="metric-value">{best_r2_display}</div>
        </div>
        <div class="metric">
            <div>Final Loss</div>
            <div class="metric-value">{final_loss_display}</div>
        </div>
        <div class="metric">
            <div>Total Epochs</div>
            <div class="metric-value">{len(df_history)}</div>
        </div>
    </div>
    
    <h2>Configuration</h2>
    <pre>{json.dumps(config, indent=2)}</pre>
    
    <h2>Training History</h2>
    <table>
        <thead>
            <tr>
                <th>Epoch</th>
                <th>Train Loss</th>
                <th>Val Loss</th>
                <th>Train MAE</th>
                <th>Val MAE</th>
                <th>Val R²</th>
            </tr>
        </thead>
        <tbody>
"""

                            rows_html = ""
                            for _, row in df_history.iterrows():
                                val_loss_display = (
                                    f"{row['val_loss']:.4f}"
                                    if 'val_loss' in df_history.columns and not pd.isna(row.get('val_loss'))
                                    else "N/A"
                                )
                                train_mae_display = (
                                    f"{row['train_mae']:.3f}"
                                    if 'train_mae' in df_history.columns and not pd.isna(row.get('train_mae'))
                                    else "N/A"
                                )
                                val_mae_display = (
                                    f"{row['val_mae']:.3f}"
                                    if not pd.isna(row.get('val_mae'))
                                    else "N/A"
                                )
                                val_r2_display = (
                                    f"{row['val_r2']:.3f}"
                                    if 'val_r2' in df_history.columns and not pd.isna(row.get('val_r2'))
                                    else "N/A"
                                )

                                rows_html += f"""
            <tr>
                <td>{int(row['epoch'])}</td>
                <td>{row['train_loss']:.4f}</td>
                <td>{val_loss_display}</td>
                <td>{train_mae_display}</td>
                <td>{val_mae_display}</td>
                <td>{val_r2_display}</td>
            </tr>
"""

                            html_content += rows_html

                            html_content += """
        </tbody>
    </table>
</body>
</html>
"""

                            with open(report_path, 'w', encoding='utf-8') as f:
                                f.write(html_content)

                            st.success(f"Report generated: `{report_path}`")
                            st.info(f"File size: {report_path.stat().st_size / 1024:.1f} KB")
                    else:
                        st.error("Training history not found")
            
            # List existing reports
            st.divider()
            st.markdown("#### Existing Reports")
            
            report_dir = PROJECT_ROOT / "docs" / "report"
            if report_dir.exists():
                reports = list(report_dir.glob("*.html"))
                if reports:
                    report_data = []
                    for report in sorted(reports, key=lambda x: x.stat().st_mtime, reverse=True):
                        report_data.append({
                            "Filename": report.name,
                            "Size (KB)": f"{report.stat().st_size / 1024:.1f}",
                            "Modified": datetime.fromtimestamp(report.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                        })
                    st.dataframe(report_data, hide_index=True, width='stretch')
                else:
                    st.info("No reports generated yet")
        else:
            st.warning("No training runs found")
    else:
        st.warning("No outputs directory found")

# Footer
st.divider()
st.caption("Tip: Use extreme augmented data and batch size 32 for best results on RTX 3060 6GB")
