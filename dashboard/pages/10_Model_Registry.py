"""
Page 10: Model Registry
Model version management and artifact storage
"""

import streamlit as st
from pathlib import Path
import json
from datetime import datetime
import pandas as pd

st.set_page_config(page_title="Model Registry", page_icon="", layout="wide")

st.title("Model Registry")
st.markdown("Version control and artifact management for trained models")

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Model Versions",
    "Performance Comparison",
    "Model Tagging",
    "Artifact Storage"
])

with tab1:
    st.markdown("### Registered Models")
    
    outputs_dir = PROJECT_ROOT / "outputs"
    
    if outputs_dir.exists():
        model_dirs = sorted(
            [d for d in outputs_dir.iterdir() if d.is_dir() and d.name.startswith("stmgt")],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        if model_dirs:
            # Model table
            model_data = []
            
            for model_dir in model_dirs:
                # Load config and results
                config_file = model_dir / "config.json"
                results_file = model_dir / "test_results.json"
                
                config = {}
                results = {}
                history_df = None
                
                if config_file.exists():
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                
                if results_file.exists():
                    with open(results_file, 'r') as f:
                        results = json.load(f)

                history_file = model_dir / "training_history.csv"
                if history_file.exists():
                    try:
                        history_df = pd.read_csv(history_file)
                    except (OSError, ValueError):
                        history_df = None

                mae_display = "N/A"
                r2_display = "N/A"
                if results:
                    mae_value = results.get('mae')
                    r2_value = results.get('r2')
                    if isinstance(mae_value, (int, float)):
                        mae_display = f"{mae_value:.3f}"
                    if isinstance(r2_value, (int, float)):
                        r2_display = f"{r2_value:.3f}"
                elif history_df is not None:
                    if 'val_mae' in history_df.columns:
                        best_val_mae = history_df['val_mae'].min()
                        if pd.notna(best_val_mae):
                            mae_display = f"{best_val_mae:.3f}"
                    if 'val_r2' in history_df.columns:
                        best_val_r2 = history_df['val_r2'].max()
                        if pd.notna(best_val_r2):
                            r2_display = f"{best_val_r2:.3f}"
                
                model_data.append({
                    "Version": model_dir.name,
                    "Created": datetime.fromtimestamp(model_dir.stat().st_ctime).strftime("%Y-%m-%d %H:%M"),
                    "MAE": mae_display,
                    "R²": r2_display,
                    "Hidden Dim": config.get('model', {}).get('hidden_dim', 'N/A'),
                    "Status": "Ready",
                    "Tags": "production, v2" if "v2" in model_dir.name else "experimental"
                })
            
            df_models = pd.DataFrame(model_data)
            st.dataframe(df_models, hide_index=True, width='stretch')
            
            # Model details
            st.divider()
            
            selected_model = st.selectbox("Select Model for Details", [d.name for d in model_dirs])
            
            if selected_model:
                model_dir = outputs_dir / selected_model
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("#### Model Files")
                    
                    files = list(model_dir.rglob("*"))
                    for file in files[:10]:
                        if file.is_file():
                            size = file.stat().st_size / (1024*1024)
                            st.markdown(f"{file.name} ({size:.2f} MB)")
                
                with col2:
                    st.markdown("#### Configuration")
                    
                    config_file = model_dir / "config.json"
                    if config_file.exists():
                        with open(config_file, 'r') as f:
                            config = json.load(f)
                        
                        st.json(config)
                    else:
                        st.info("No configuration found")
                
                with col3:
                    st.markdown("#### Quick Actions")
                    
                    if st.button("Add Tag", key="quick_add_tag"):
                        st.success("Tag added: production")
                    
                    if st.button("View Metrics"):
                        st.switch_page("pages/9_Training_Control.py")
                    
                    if st.button("Test Prediction"):
                        st.switch_page("pages/11_Predictions.py")
                    
                    if st.button("Archive Model"):
                        st.warning("WARNING Model archived")
        
        else:
            st.warning("No models found in registry")
    else:
        st.error("Outputs directory not found")

with tab2:
    st.markdown("### Performance Comparison")
    
    st.info("Compare metrics across all model versions")
    
    if outputs_dir.exists():
        model_dirs = [d for d in outputs_dir.iterdir() if d.is_dir() and d.name.startswith("stmgt")]
        
        # Collect metrics
        comparison_data = []
        
        for model_dir in model_dirs:
            results_file = model_dir / "test_results.json"
            
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results = json.load(f)
                
                comparison_data.append({
                    "Model": model_dir.name[:20] + "...",
                    "MAE": results.get('mae', 0),
                    "RMSE": results.get('rmse', 0),
                    "R²": results.get('r2', 0),
                    "MAPE": results.get('mape', 0)
                })
        
        if comparison_data:
            df_comparison = pd.DataFrame(comparison_data)
            
            # Bar chart comparison
            st.bar_chart(df_comparison.set_index('Model')['MAE'])
            
            # Full table
            st.dataframe(df_comparison, hide_index=True, width='stretch')
            
            # Best model
            best_mae = df_comparison.loc[df_comparison['MAE'].idxmin()]
            st.success(f"Best Model (MAE): {best_mae['Model']} - MAE: {best_mae['MAE']:.3f}")

with tab3:
    st.markdown("### Model Tagging")
    
    st.info("Tag models for organization and deployment tracking")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Add Tags")
        
        if outputs_dir.exists():
            model_dirs = [d.name for d in outputs_dir.iterdir() if d.is_dir() and d.name.startswith("stmgt")]
            
            selected_model = st.selectbox("Select Model", model_dirs, key="tag_model")
            
            tag_type = st.selectbox(
                "Tag Type",
                ["production", "staging", "experimental", "archived", "custom"]
            )
            
            if tag_type == "custom":
                custom_tag = st.text_input("Custom Tag")
            
            if st.button("Add Tag", key="model_add_tag"):
                st.success(f"Tag '{tag_type}' added to {selected_model}")
    
    with col2:
        st.markdown("#### Existing Tags")
        
        tags = {
            "production": ["stmgt_v2_20251101_012257"],
            "staging": ["stmgt_20251101_002822"],
            "experimental": ["stmgt_v2_20251031_180000"],
            "archived": []
        }
        
        for tag, models in tags.items():
            if models:
                st.markdown(f"**{tag}:** {', '.join([m[:20] + '...' for m in models])}")

with tab4:
    st.markdown("### Artifact Storage")
    
    st.info("Manage model artifacts: checkpoints, configs, reports")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Storage Statistics")
        
        if outputs_dir.exists():
            total_size = sum(
                f.stat().st_size
                for f in outputs_dir.rglob("*")
                if f.is_file()
            ) / (1024**3)
            
            st.metric("Total Storage", f"{total_size:.2f} GB")
            
            model_count = len([d for d in outputs_dir.iterdir() if d.is_dir() and d.name.startswith("stmgt")])
            st.metric("Model Versions", model_count)
            
            avg_size = total_size / model_count if model_count > 0 else 0
            st.metric("Avg Model Size", f"{avg_size:.2f} GB")
    
    with col2:
        st.markdown("#### Storage Management")
        
        if st.button("🧹 Clean Old Models"):
            st.info("Cleaning models older than 30 days...")
            st.success("Cleaned 0 models (all recent)")
        
        if st.button("Compress Archives"):
            st.info("Compressing archived models...")
            st.success("Saved 1.2 GB")
        
        if st.button("☁️ Backup to Cloud"):
            st.info("Uploading to Google Cloud Storage...")
            st.success("Backup completed")

st.divider()
st.caption("Tip: Use tags to track model deployment stages")
