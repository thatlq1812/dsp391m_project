"""
Page 5: Predictions
Real-time traffic predictions with uncertainty quantification
"""

import streamlit as st
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import json
from datetime import datetime
import torch

st.set_page_config(page_title="Predictions", page_icon="", layout="wide")

st.title("Traffic Predictions")
st.markdown("Generate forecasts with uncertainty quantification")

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Model Selection",
    "Real-Time Prediction",
    "Scenario Simulation",
    "WARNING Alerts & Monitoring",
    "Export Predictions"
])

# Initialize session state
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'selected_model_path' not in st.session_state:
    st.session_state.selected_model_path = None

with tab1:
    st.markdown("### Model Selection")
    
    outputs_dir = PROJECT_ROOT / "outputs"
    
    if not outputs_dir.exists():
        st.warning("No trained models found. Please train a model first.")
    else:
        model_dirs = sorted([d for d in outputs_dir.iterdir() if d.is_dir() and d.name.startswith("stmgt")],
                           key=lambda x: x.stat().st_mtime, reverse=True)
        
        if not model_dirs:
            st.warning("No STMGT models found in outputs directory")
        else:
            st.success(f"Found {len(model_dirs)} trained models")
            
            # Select model
            model_names = [d.name for d in model_dirs]
            selected_model = st.selectbox("Select Model", model_names, index=0)
            
            model_dir = outputs_dir / selected_model
            st.session_state.selected_model_path = model_dir
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Model Information")
                
                # Load config
                config_file = model_dir / "config.json"
                if config_file.exists():
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                    
                    st.json(config)
                else:
                    st.info("Configuration file not found")
                
                # Model file
                model_file = model_dir / "best_model.pt"
                if model_file.exists():
                    st.success(f"Model file: `best_model.pt`")
                    st.caption(f"Size: {model_file.stat().st_size / 1024 / 1024:.1f} MB")
                else:
                    st.warning("WARNING Model checkpoint not found")
            
            with col2:
                st.markdown("#### Test Results")
                
                # Load test results
                test_results_file = model_dir / "test_results.json"
                if test_results_file.exists():
                    with open(test_results_file, 'r') as f:
                        test_results = json.load(f)
                    
                    metrics_data = {
                        "Metric": ["MAE", "RMSE", "R²", "MAPE", "CRPS", "Coverage 80%", "Coverage 95%"],
                        "Value": [
                            f"{test_results.get('mae', 0):.3f} km/h",
                            f"{test_results.get('rmse', 0):.3f} km/h",
                            f"{test_results.get('r2', 0):.3f}",
                            f"{test_results.get('mape', 0):.2f}%",
                            f"{test_results.get('crps', 0):.3f}",
                            f"{test_results.get('coverage_80', 0):.1f}%",
                            f"{test_results.get('coverage_95', 0):.1f}%"
                        ]
                    }
                    
                    st.dataframe(metrics_data, hide_index=True, width='stretch')
                else:
                    st.info("Test results not available")

with tab2:
    st.markdown("### Real-Time Prediction")
    
    if st.session_state.selected_model_path is None:
        st.warning("Please select a model in the Model Selection tab first")
    else:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### Data Source")
            
            data_source = st.radio(
                "Input Data",
                ["Load Latest Traffic Data", "Use Simulated Data"],
                index=1
            )
            
            forecast_horizon = st.slider("Forecast Horizon (timesteps)", 6, 24, 12, 6)
            st.caption(f"Forecasting {forecast_horizon * 15} minutes ahead")
            
            if st.button("Generate Predictions", width='stretch', type="primary"):
                with st.spinner("Generating predictions..."):
                    # Simulate predictions (in production, load real model)
                    num_edges = 62
                    num_timesteps = forecast_horizon
                    
                    # Simulated base speeds
                    np.random.seed(42)
                    base_speeds = np.random.normal(40, 15, (num_edges, num_timesteps))
                    base_speeds = np.clip(base_speeds, 5, 80)
                    
                    # Simulated uncertainty
                    uncertainties = np.random.uniform(2, 8, (num_edges, num_timesteps))
                    
                    # Store predictions
                    st.session_state.predictions = {
                        'mean': base_speeds,
                        'std': uncertainties,
                        'lower_80': base_speeds - 1.28 * uncertainties,
                        'upper_80': base_speeds + 1.28 * uncertainties,
                        'lower_95': base_speeds - 1.96 * uncertainties,
                        'upper_95': base_speeds + 1.96 * uncertainties,
                        'num_edges': num_edges,
                        'num_timesteps': num_timesteps
                    }
                    
                    st.success("Predictions generated!")
                    st.info("Note: Currently using simulated inference. Real model loading coming soon.")
        
        with col2:
            if st.session_state.predictions is not None:
                st.markdown("#### Prediction Visualization")
                
                preds = st.session_state.predictions
                
                # Select edge to visualize
                edge_id = st.selectbox("Select Edge ID", range(preds['num_edges']), index=0)
                
                # Plot predictions with uncertainty
                timesteps = np.arange(preds['num_timesteps'])
                
                fig = go.Figure()
                
                # Mean prediction
                fig.add_trace(go.Scatter(
                    x=timesteps,
                    y=preds['mean'][edge_id],
                    mode='lines+markers',
                    name='Mean Prediction',
                    line=dict(color='#1f77b4', width=3),
                    marker=dict(size=8)
                ))
                
                # 80% confidence interval
                fig.add_trace(go.Scatter(
                    x=np.concatenate([timesteps, timesteps[::-1]]),
                    y=np.concatenate([preds['upper_80'][edge_id], preds['lower_80'][edge_id][::-1]]),
                    fill='toself',
                    fillcolor='rgba(31, 119, 180, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='80% CI',
                    showlegend=True
                ))
                
                # 95% confidence interval
                fig.add_trace(go.Scatter(
                    x=np.concatenate([timesteps, timesteps[::-1]]),
                    y=np.concatenate([preds['upper_95'][edge_id], preds['lower_95'][edge_id][::-1]]),
                    fill='toself',
                    fillcolor='rgba(31, 119, 180, 0.1)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='95% CI',
                    showlegend=True
                ))
                
                fig.update_layout(
                    title=f"Speed Prediction for Edge {edge_id}",
                    xaxis_title="Timestep (15-min intervals)",
                    yaxis_title="Speed (km/h)",
                    height=500
                )
                
                st.plotly_chart(fig, width='stretch')
                
                # Summary statistics
                st.markdown("#### Prediction Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Mean Speed", f"{preds['mean'].mean():.1f} km/h")
                
                with col2:
                    st.metric("Min Speed", f"{preds['mean'].min():.1f} km/h")
                
                with col3:
                    st.metric("Max Speed", f"{preds['mean'].max():.1f} km/h")
                
                with col4:
                    st.metric("Avg Uncertainty", f"{preds['std'].mean():.1f} km/h")

with tab3:
    st.markdown("### Scenario Simulation")
    
    st.info("""
    Simulate "What-if" scenarios by adjusting weather conditions and time factors.
    Compare predicted speeds under different scenarios.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Scenario Parameters")
        
        weather_condition = st.selectbox(
            "Weather Condition",
            ["Clear", "Light Rain", "Heavy Rain", "Storm"],
            index=0
        )
        
        temperature = st.slider("Temperature (°C)", -10, 40, 20, 1)
        
        time_of_day = st.select_slider(
            "Time of Day",
            options=["Morning Rush", "Midday", "Evening Rush", "Night"],
            value="Morning Rush"
        )
        
        is_weekend = st.checkbox("Weekend", value=False)
        
        if st.button("Run Scenario", width='stretch', type="primary"):
            with st.spinner("Running scenario simulation..."):
                # Simulate scenario impact
                weather_impact = {
                    "Clear": 1.0,
                    "Light Rain": 0.9,
                    "Heavy Rain": 0.75,
                    "Storm": 0.6
                }
                
                time_impact = {
                    "Morning Rush": 0.7,
                    "Midday": 1.0,
                    "Evening Rush": 0.65,
                    "Night": 1.1
                }
                
                # Generate baseline and scenario predictions
                np.random.seed(42)
                baseline = np.random.normal(45, 12, 62)
                
                scenario_speed = baseline * weather_impact[weather_condition] * time_impact[time_of_day]
                
                if is_weekend:
                    scenario_speed *= 1.15  # Less traffic on weekends
                
                st.session_state.scenario_results = {
                    'baseline': baseline,
                    'scenario': scenario_speed,
                    'weather': weather_condition,
                    'time': time_of_day,
                    'weekend': is_weekend
                }
                
                st.success("Scenario simulation complete!")
    
    with col2:
        st.markdown("#### Scenario Impact Analysis")
        
        if 'scenario_results' in st.session_state:
            results = st.session_state.scenario_results
            
            # Impact metrics
            speed_reduction = results['baseline'].mean() - results['scenario'].mean()
            reduction_pct = (speed_reduction / results['baseline'].mean()) * 100
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Baseline Avg Speed", f"{results['baseline'].mean():.1f} km/h")
                st.metric("Scenario Avg Speed", f"{results['scenario'].mean():.1f} km/h", 
                         delta=f"{-speed_reduction:.1f} km/h", delta_color="inverse")
            
            with col2:
                st.metric("Speed Reduction", f"{reduction_pct:.1f}%")
                
                congested_baseline = (results['baseline'] < 20).sum()
                congested_scenario = (results['scenario'] < 20).sum()
                
                st.metric("Congested Edges", congested_scenario, 
                         delta=f"+{congested_scenario - congested_baseline}")
            
            # Comparison chart
            st.markdown("#### Speed Comparison")
            
            fig = go.Figure()
            
            fig.add_trace(go.Box(
                y=results['baseline'],
                name='Baseline',
                marker_color='#1f77b4'
            ))
            
            fig.add_trace(go.Box(
                y=results['scenario'],
                name='Scenario',
                marker_color='#ff7f0e'
            ))
            
            fig.update_layout(
                yaxis_title="Speed (km/h)",
                height=400
            )
            
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("Run a scenario to see impact analysis")

with tab4:
    st.markdown("### Alerts & Monitoring")
    
    st.info("""
    Set congestion thresholds and monitor predicted speeds.
    Generate alerts when speeds fall below critical levels.
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Alert Configuration")
        
        threshold_speed = st.slider("Congestion Threshold (km/h)", 10, 40, 20, 5)
        
        severity_filter = st.multiselect(
            "Severity Levels",
            ["High", "Medium", "Low"],
            default=["High", "Medium"]
        )
        
        if st.session_state.predictions is not None:
            if st.button("WARNING Check for Alerts", width='stretch'):
                preds = st.session_state.predictions
                
                alerts = []
                
                for edge_id in range(preds['num_edges']):
                    for timestep in range(preds['num_timesteps']):
                        speed = preds['mean'][edge_id, timestep]
                        
                        if speed < threshold_speed:
                            if speed < threshold_speed * 0.7:
                                severity = "High"
                            elif speed < threshold_speed * 0.85:
                                severity = "Medium"
                            else:
                                severity = "Low"
                            
                            if severity in severity_filter:
                                alerts.append({
                                    'edge_id': edge_id,
                                    'timestep': timestep,
                                    'predicted_speed': speed,
                                    'severity': severity,
                                    'time_ahead': f"{(timestep + 1) * 15} min"
                                })
                
                st.session_state.alerts = alerts
                
                if alerts:
                    st.warning(f"WARNING {len(alerts)} alerts detected!")
                else:
                    st.success("No congestion alerts")
        else:
            st.info("Generate predictions first to check alerts")
    
    with col2:
        st.markdown("#### Active Alerts")
        
        if 'alerts' in st.session_state and st.session_state.alerts:
            alerts = st.session_state.alerts
            
            # Summary
            col1, col2, col3 = st.columns(3)
            
            with col1:
                high_alerts = sum(1 for a in alerts if a['severity'] == 'High')
                st.metric("High Severity", high_alerts)
            
            with col2:
                med_alerts = sum(1 for a in alerts if a['severity'] == 'Medium')
                st.metric("Medium Severity", med_alerts)
            
            with col3:
                low_alerts = sum(1 for a in alerts if a['severity'] == 'Low')
                st.metric("Low Severity", low_alerts)
            
            # Alert table
            st.markdown("#### Alert Details")
            df_alerts = pd.DataFrame(alerts)
            
            # Color code by severity
            def color_severity(val):
                colors = {'High': 'background-color: #ffcdd2', 
                         'Medium': 'background-color: #fff9c4',
                         'Low': 'background-color: #c8e6c9'}
                return colors.get(val, '')
            
            st.dataframe(
                df_alerts.style.applymap(color_severity, subset=['severity']),
                width='stretch',
                hide_index=True
            )
            
            # Alert heatmap
            st.markdown("#### Alert Heatmap")
            
            preds = st.session_state.predictions
            alert_matrix = np.zeros((preds['num_edges'], preds['num_timesteps']))
            
            for alert in alerts:
                alert_matrix[alert['edge_id'], alert['timestep']] = 1
            
            fig = go.Figure(data=go.Heatmap(
                z=alert_matrix,
                colorscale=[[0, 'green'], [1, 'red']],
                showscale=False
            ))
            
            fig.update_layout(
                title="Congestion Alert Map (Red = Alert)",
                xaxis_title="Timestep",
                yaxis_title="Edge ID",
                height=400
            )
            
            st.plotly_chart(fig, width='stretch')
        
        else:
            st.info("No alerts to display. Generate predictions and check for alerts.")

with tab5:
    st.markdown("### Export Predictions")
    
    if st.session_state.predictions is None:
        st.warning("No predictions available. Generate predictions first in the Real-Time Prediction tab.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Export Settings")
            
            export_format = st.selectbox("Export Format", ["CSV", "Parquet", "JSON"], index=0)
            
            include_uncertainty = st.checkbox("Include Uncertainty Bounds", value=True)
            
            filename = st.text_input(
                "Filename",
                f"predictions_{datetime.now():%Y%m%d_%H%M%S}"
            )
            
            if st.button("Export Predictions", width='stretch', type="primary"):
                preds = st.session_state.predictions
                
                # Prepare export data
                export_data = []
                
                for edge_id in range(preds['num_edges']):
                    for timestep in range(preds['num_timesteps']):
                        row = {
                            'edge_id': edge_id,
                            'timestep': timestep,
                            'time_ahead_min': (timestep + 1) * 15,
                            'predicted_speed': preds['mean'][edge_id, timestep]
                        }
                        
                        if include_uncertainty:
                            row.update({
                                'std': preds['std'][edge_id, timestep],
                                'lower_80': preds['lower_80'][edge_id, timestep],
                                'upper_80': preds['upper_80'][edge_id, timestep],
                                'lower_95': preds['lower_95'][edge_id, timestep],
                                'upper_95': preds['upper_95'][edge_id, timestep]
                            })
                        
                        export_data.append(row)
                
                df_export = pd.DataFrame(export_data)
                
                # Export
                export_dir = PROJECT_ROOT / "data" / "predictions"
                export_dir.mkdir(exist_ok=True, parents=True)
                
                if export_format == "CSV":
                    export_path = export_dir / f"{filename}.csv"
                    df_export.to_csv(export_path, index=False)
                elif export_format == "Parquet":
                    export_path = export_dir / f"{filename}.parquet"
                    df_export.to_parquet(export_path, index=False)
                else:  # JSON
                    export_path = export_dir / f"{filename}.json"
                    df_export.to_json(export_path, orient='records', indent=2)
                
                st.success(f"Predictions exported to `{export_path}`")
                st.info(f"File size: {export_path.stat().st_size / 1024:.1f} KB")
        
        with col2:
            st.markdown("#### Export Preview")
            
            preds = st.session_state.predictions
            
            # Show sample data
            sample_data = []
            for edge_id in range(min(5, preds['num_edges'])):
                for timestep in range(min(3, preds['num_timesteps'])):
                    row = {
                        'edge_id': edge_id,
                        'timestep': timestep,
                        'speed': f"{preds['mean'][edge_id, timestep]:.1f}",
                    }
                    
                    if include_uncertainty:
                        row['std'] = f"{preds['std'][edge_id, timestep]:.1f}"
                    
                    sample_data.append(row)
            
            st.dataframe(pd.DataFrame(sample_data), hide_index=True, width='stretch')
            st.caption("Preview of first few rows")
    
    st.divider()
    
    # List existing exports
    st.markdown("#### Existing Prediction Exports")
    
    export_dir = PROJECT_ROOT / "data" / "predictions"
    if export_dir.exists():
        export_files = list(export_dir.glob("predictions_*.*"))
        
        if export_files:
            export_data = []
            for file in sorted(export_files, key=lambda x: x.stat().st_mtime, reverse=True)[:10]:
                export_data.append({
                    "Filename": file.name,
                    "Format": file.suffix[1:].upper(),
                    "Size (KB)": f"{file.stat().st_size / 1024:.1f}",
                    "Created": datetime.fromtimestamp(file.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                })
            
            st.dataframe(export_data, hide_index=True, width='stretch')
        else:
            st.info("No exports yet")
    else:
        st.info("No exports directory found")

# Footer
st.divider()
st.caption("Tip: Use 80% confidence intervals for operational planning, 95% for risk assessment")
