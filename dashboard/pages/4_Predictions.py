"""
Predictions Page

Make predictions and visualize forecasts using trained models.

Author: thatlq1812
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
from datetime import datetime, timedelta

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="Predictions", page_icon="🔮", layout="wide")

st.title("Traffic Speed Predictions")
st.markdown("Make real-time predictions and visualize forecasts")

# Tabs
tab1, tab2, tab3 = st.tabs(["Single Prediction", "Batch Prediction", "Forecast Visualization"])

with tab1:
    st.header("Single Point Prediction")
    
    st.markdown("Predict traffic speed for a specific edge and time")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Parameters")
        
        # Time inputs
        prediction_date = st.date_input("Date", datetime.now())
        prediction_time = st.time_input("Time", datetime.now().time())
        
        # Location
        edge_id = st.text_input("Edge ID", placeholder="e.g., node_1_to_node_2")
        
        # Traffic features
        st.markdown("**Traffic Context:**")
        distance_km = st.number_input("Distance (km)", min_value=0.1, max_value=10.0, value=1.5)
        
        # Weather features
        st.markdown("**Weather Conditions:**")
        temperature_c = st.slider("Temperature (°C)", 20, 40, 30)
        wind_speed_kmh = st.slider("Wind Speed (km/h)", 0, 50, 10)
        precipitation_mm = st.slider("Precipitation (mm)", 0.0, 50.0, 0.0)
    
    with col2:
        st.subheader("Model Selection")
        
        selected_model = st.radio(
            "Choose Model:",
            ["LSTM", "ASTGCN", "Ensemble (Average)"],
            help="Select which model to use for prediction"
        )
        
        if st.button("Predict", type="primary", use_container_width=True):
            st.info("Making prediction...")
            
            # TODO: Implement prediction logic
            # For now, show dummy prediction
            predicted_speed = np.random.uniform(20, 50)
            confidence = np.random.uniform(0.7, 0.95)
            
            st.markdown("---")
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Predicted Speed",
                    f"{predicted_speed:.2f} km/h",
                    delta=f"{np.random.uniform(-5, 5):.2f} from avg"
                )
            
            with col2:
                st.metric(
                    "Confidence",
                    f"{confidence*100:.1f}%"
                )
            
            with col3:
                congestion = "Light" if predicted_speed > 30 else "Moderate" if predicted_speed > 20 else "Heavy"
                st.metric("Congestion Level", congestion)
            
            # Prediction range
            st.markdown("**Prediction Range:**")
            lower_bound = predicted_speed - 5
            upper_bound = predicted_speed + 5
            
            fig = go.Figure()
            
            fig.add_trace(go.Indicator(
                mode = "gauge+number+delta",
                value = predicted_speed,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Speed (km/h)"},
                delta = {'reference': 30},
                gauge = {
                    'axis': {'range': [None, 60]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 15], 'color': "red"},
                        {'range': [15, 25], 'color': "orange"},
                        {'range': [25, 35], 'color': "yellow"},
                        {'range': [35, 60], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 35
                    }
                }
            ))
            
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Batch Prediction")
    
    st.markdown("Upload a CSV file or select time range for multiple predictions")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Time range selection
        st.subheader("Time Range Selection")
        
        col_start, col_end = st.columns(2)
        
        with col_start:
            start_date = st.date_input("Start Date", datetime.now())
            start_time = st.time_input("Start Time", datetime.now().time())
        
        with col_end:
            end_date = st.date_input("End Date", datetime.now() + timedelta(days=1))
            end_time = st.time_input("End Time", datetime.now().time())
        
        interval = st.selectbox(
            "Prediction Interval",
            ["15 minutes", "30 minutes", "1 hour", "3 hours"],
            index=0
        )
    
    with col2:
        st.subheader("Options")
        
        include_confidence = st.checkbox("Include Confidence Intervals", value=True)
        export_results = st.checkbox("Export Results to CSV", value=False)
    
    if st.button("Generate Predictions", type="primary", use_container_width=True):
        st.info("Generating batch predictions...")
        
        # TODO: Implement batch prediction
        # For now, create dummy data
        
        # Generate time range
        start_datetime = datetime.combine(start_date, start_time)
        end_datetime = datetime.combine(end_date, end_time)
        
        # Create time series
        interval_minutes = {'15 minutes': 15, '30 minutes': 30, '1 hour': 60, '3 hours': 180}[interval]
        time_points = pd.date_range(start_datetime, end_datetime, freq=f'{interval_minutes}T')
        
        # Dummy predictions
        predictions_df = pd.DataFrame({
            'Timestamp': time_points,
            'Predicted Speed (km/h)': np.random.uniform(20, 50, len(time_points)),
            'Lower Bound': np.random.uniform(15, 25, len(time_points)),
            'Upper Bound': np.random.uniform(45, 55, len(time_points)),
            'Congestion Level': np.random.choice(['Light', 'Moderate', 'Heavy'], len(time_points))
        })
        
        st.success(f"Generated {len(predictions_df)} predictions")
        
        # Display table
        st.dataframe(predictions_df, use_container_width=True)
        
        # Download button
        if export_results:
            csv = predictions_df.to_csv(index=False)
            st.download_button(
                label="Download Predictions CSV",
                data=csv,
                file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

with tab3:
    st.header("Forecast Visualization")
    
    st.markdown("Visualize model predictions over time")
    
    # Model comparison
    st.subheader("Model Comparison")
    
    models_to_compare = st.multiselect(
        "Select models to compare:",
        ["LSTM", "ASTGCN", "Actual"],
        default=["LSTM", "ASTGCN", "Actual"]
    )
    
    # Generate dummy forecast data
    forecast_hours = st.slider("Forecast Horizon (hours)", 1, 24, 6)
    
    if st.button("Generate Forecast", type="primary"):
        # Create time range
        now = datetime.now()
        time_points = pd.date_range(now, now + timedelta(hours=forecast_hours), freq='15T')
        
        # Create forecast data
        forecast_data = {
            'Timestamp': time_points,
        }
        
        if "Actual" in models_to_compare:
            forecast_data['Actual'] = np.random.uniform(20, 50, len(time_points))
        
        if "LSTM" in models_to_compare:
            forecast_data['LSTM'] = np.random.uniform(20, 50, len(time_points))
        
        if "ASTGCN" in models_to_compare:
            forecast_data['ASTGCN'] = np.random.uniform(20, 50, len(time_points))
        
        df_forecast = pd.DataFrame(forecast_data)
        
        # Plot
        fig = go.Figure()
        
        colors = {'Actual': 'black', 'LSTM': '#1f77b4', 'ASTGCN': '#ff7f0e'}
        
        for model in models_to_compare:
            if model in df_forecast.columns:
                fig.add_trace(go.Scatter(
                    x=df_forecast['Timestamp'],
                    y=df_forecast[model],
                    mode='lines+markers',
                    name=model,
                    line=dict(color=colors.get(model, None), width=2)
                ))
        
        fig.update_layout(
            title=f'Traffic Speed Forecast - Next {forecast_hours} Hours',
            xaxis_title='Time',
            yaxis_title='Speed (km/h)',
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        st.markdown("### Forecast Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        if "LSTM" in df_forecast.columns:
            with col1:
                st.metric("LSTM Avg Speed", f"{df_forecast['LSTM'].mean():.2f} km/h")
        
        if "ASTGCN" in df_forecast.columns:
            with col2:
                st.metric("ASTGCN Avg Speed", f"{df_forecast['ASTGCN'].mean():.2f} km/h")
        
        if "Actual" in df_forecast.columns and "LSTM" in df_forecast.columns:
            with col3:
                mae = np.abs(df_forecast['LSTM'] - df_forecast['Actual']).mean()
                st.metric("LSTM MAE", f"{mae:.2f} km/h")
    
    # Heatmap visualization
    st.markdown("---")
    st.subheader("Traffic Heatmap")
    
    if st.button("Generate Heatmap"):
        # Create dummy heatmap data
        hours = list(range(24))
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        heatmap_data = np.random.uniform(15, 50, (len(days), len(hours)))
        
        fig = px.imshow(
            heatmap_data,
            x=hours,
            y=days,
            labels=dict(x="Hour of Day", y="Day of Week", color="Speed (km/h)"),
            color_continuous_scale='RdYlGn',
            title='Average Traffic Speed by Hour and Day'
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Prediction Dashboard | Real-time Traffic Forecasting")
