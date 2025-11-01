"""
Page 3: Data Visualization
Explore traffic patterns, distributions, and network topology
"""

import streamlit as st
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import networkx as nx
from sklearn.mixture import GaussianMixture

st.set_page_config(page_title="Visualization", page_icon="", layout="wide")

st.title("Data Visualization")
st.markdown("Explore traffic patterns and network characteristics")

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Dataset selection
dataset_options = {
    "Original Combined": "all_runs_combined.parquet",
    "Basic Augmented (23.4x)": "all_runs_augmented.parquet",
    "Extreme Augmented (48.4x)": "all_runs_extreme_augmented.parquet"
}

selected_dataset = st.selectbox("Select Dataset", list(dataset_options.keys()), index=2)
data_path = PROJECT_ROOT / "data" / "processed" / dataset_options[selected_dataset]

if not data_path.exists():
    st.error(f"Dataset not found: {data_path}")
    st.info("Please run data collection and augmentation first (see Data Overview page)")
    st.stop()

# Load data
@st.cache_data
def load_data(path):
    return pd.read_parquet(path)

with st.spinner("Loading data..."):
    df = load_data(data_path)
    st.success(f"Loaded {len(df):,} records from {selected_dataset}")

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Speed Distribution", 
    "Temporal Patterns", 
    "Spatial Analysis",
    "Weather Correlation",
    "GMM Distribution",
    "Graph Topology"
])

with tab1:
    st.markdown("### Speed Distribution Analysis")
    
    if 'speed_kmh' in df.columns:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Histogram
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=df['speed_kmh'],
                nbinsx=100,
                marker_color='#1f77b4',
                name='Speed Distribution'
            ))
            fig.update_layout(
                title="Traffic Speed Distribution",
                xaxis_title="Speed (km/h)",
                yaxis_title="Frequency",
                height=500
            )
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            st.markdown("#### Statistics")
            stats = df['speed_kmh'].describe()
            
            st.metric("Mean Speed", f"{stats['mean']:.2f} km/h")
            st.metric("Median Speed", f"{stats['50%']:.2f} km/h")
            st.metric("Std Deviation", f"{stats['std']:.2f} km/h")
            st.metric("Min Speed", f"{stats['min']:.2f} km/h")
            st.metric("Max Speed", f"{stats['max']:.2f} km/h")
            
            # Speed categories
            st.markdown("#### Speed Categories")
            congested = (df['speed_kmh'] < 20).sum()
            moderate = ((df['speed_kmh'] >= 20) & (df['speed_kmh'] < 50)).sum()
            free_flow = (df['speed_kmh'] >= 50).sum()
            
            total = len(df)
            st.write(f"Congested (<20): {congested/total*100:.1f}%")
            st.write(f"Moderate (20-50): {moderate/total*100:.1f}%")
            st.write(f"Free Flow (≥50): {free_flow/total*100:.1f}%")
        
        # Box plot by hour
        if 'timestamp' in df.columns:
            st.markdown("### Speed Distribution by Hour")
            df_temp = df.copy()
            df_temp['hour'] = pd.to_datetime(df_temp['timestamp']).dt.hour
            
            fig = go.Figure()
            for hour in range(24):
                hour_data = df_temp[df_temp['hour'] == hour]['speed_kmh']
                fig.add_trace(go.Box(
                    y=hour_data,
                    name=f"{hour:02d}:00",
                    marker_color='#1f77b4'
                ))
            
            fig.update_layout(
                xaxis_title="Hour of Day",
                yaxis_title="Speed (km/h)",
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig, width='stretch')

with tab2:
    st.markdown("### Temporal Traffic Patterns")
    
    if 'timestamp' in df.columns and 'speed_kmh' in df.columns:
        df_temp = df.copy()
        df_temp['timestamp'] = pd.to_datetime(df_temp['timestamp'])
        df_temp['hour'] = df_temp['timestamp'].dt.hour
        df_temp['dow'] = df_temp['timestamp'].dt.dayofweek
        df_temp['is_weekend'] = df_temp['dow'].isin([5, 6])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Hourly Average Speed")
            hourly_avg = df_temp.groupby('hour')['speed_kmh'].mean().reset_index()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hourly_avg['hour'],
                y=hourly_avg['speed_kmh'],
                mode='lines+markers',
                name='Avg Speed',
                line=dict(color='#ff7f0e', width=3),
                marker=dict(size=8)
            ))
            fig.update_layout(
                xaxis_title="Hour of Day",
                yaxis_title="Average Speed (km/h)",
                height=400,
                xaxis=dict(tickmode='linear', tick0=0, dtick=2)
            )
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            st.markdown("#### Weekday vs Weekend")
            weekend_avg = df_temp.groupby(['hour', 'is_weekend'])['speed_kmh'].mean().reset_index()
            
            fig = go.Figure()
            for is_weekend in [False, True]:
                data = weekend_avg[weekend_avg['is_weekend'] == is_weekend]
                fig.add_trace(go.Scatter(
                    x=data['hour'],
                    y=data['speed_kmh'],
                    mode='lines+markers',
                    name='Weekend' if is_weekend else 'Weekday',
                    line=dict(width=3)
                ))
            
            fig.update_layout(
                xaxis_title="Hour of Day",
                yaxis_title="Average Speed (km/h)",
                height=400
            )
            st.plotly_chart(fig, width='stretch')
        
        # Heatmap
        st.markdown("#### Speed Heatmap (Hour vs Day of Week)")
        heatmap_data = df_temp.pivot_table(
            values='speed_kmh',
            index='hour',
            columns='dow',
            aggfunc='mean'
        )
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
            y=list(range(24)),
            colorscale='RdYlGn',
            colorbar=dict(title="Speed (km/h)")
        ))
        fig.update_layout(
            xaxis_title="Day of Week",
            yaxis_title="Hour of Day",
            height=500
        )
        st.plotly_chart(fig, width='stretch')

with tab3:
    st.markdown("### Spatial Analysis")
    
    if 'edge_id' in df.columns and 'speed_kmh' in df.columns:
        # Edge statistics
        edge_stats = df.groupby('edge_id')['speed_kmh'].agg(['mean', 'std', 'min', 'max', 'count']).reset_index()
        edge_stats = edge_stats.sort_values('mean', ascending=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Average Speed by Edge (Top 20 Slowest)")
            top_slow = edge_stats.head(20)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=top_slow['edge_id'].astype(str),
                x=top_slow['mean'],
                orientation='h',
                marker_color='#d62728',
                text=top_slow['mean'].round(1),
                textposition='auto'
            ))
            fig.update_layout(
                xaxis_title="Average Speed (km/h)",
                yaxis_title="Edge ID",
                height=600
            )
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            st.markdown("#### Average Speed by Edge (Top 20 Fastest)")
            top_fast = edge_stats.tail(20)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=top_fast['edge_id'].astype(str),
                x=top_fast['mean'],
                orientation='h',
                marker_color='#2ca02c',
                text=top_fast['mean'].round(1),
                textposition='auto'
            ))
            fig.update_layout(
                xaxis_title="Average Speed (km/h)",
                yaxis_title="Edge ID",
                height=600
            )
            st.plotly_chart(fig, width='stretch')
        
        # Edge variability
        st.markdown("#### Speed Variability by Edge")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=edge_stats['mean'],
            y=edge_stats['std'],
            mode='markers',
            marker=dict(
                size=8,
                color=edge_stats['count'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Samples")
            ),
            text=edge_stats['edge_id'],
            hovertemplate='Edge %{text}<br>Mean: %{x:.1f} km/h<br>Std: %{y:.1f} km/h'
        ))
        fig.update_layout(
            xaxis_title="Mean Speed (km/h)",
            yaxis_title="Std Deviation (km/h)",
            height=500
        )
        st.plotly_chart(fig, width='stretch')

with tab4:
    st.markdown("### Weather Correlation Analysis")
    
    weather_cols = [col for col in df.columns if 'weather' in col.lower() or col in ['temperature_celsius', 'rain_mm', 'wind_speed_kmh']]
    
    if weather_cols and 'speed_kmh' in df.columns:
        st.markdown("#### Weather Features vs Speed")
        
        # Select weather feature
        selected_weather = st.selectbox("Select Weather Feature", weather_cols)
        
        # Scatter plot
        df_sample = df.sample(min(5000, len(df)))
        
        # Filter out rows with NaN/None values in selected columns
        df_clean = df_sample.dropna(subset=[selected_weather, 'speed_kmh'])
        
        if len(df_clean) < 10:
            st.warning(f"Not enough valid data points for {selected_weather} correlation analysis. Need at least 10 points.")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_clean[selected_weather],
                y=df_clean['speed_kmh'],
                mode='markers',
                marker=dict(
                    size=4,
                    color=df_clean['speed_kmh'],
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="Speed (km/h)")
                ),
                opacity=0.5
            ))
            
            # Add trend line only if we have enough clean data
            try:
                z = np.polyfit(df_clean[selected_weather], df_clean['speed_kmh'], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(df_clean[selected_weather].min(), df_clean[selected_weather].max(), 100)
                
                fig.add_trace(go.Scatter(
                    x=x_trend,
                    y=p(x_trend),
                    mode='lines',
                    name='Trend',
                    line=dict(color='red', width=2, dash='dash')
                ))
            except (TypeError, ValueError) as e:
                st.warning(f"Could not calculate trend line: {e}")
            
            fig.update_layout(
                xaxis_title=selected_weather,
                yaxis_title="Speed (km/h)",
                height=500
            )
            st.plotly_chart(fig, width='stretch')
        
        # Correlation matrix
        st.markdown("#### Correlation Matrix")
        corr_features = ['speed_kmh'] + weather_cols
        corr_features = [f for f in corr_features if f in df.columns]
        
        corr_matrix = df[corr_features].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values.round(2),
            texttemplate='%{text}',
            colorbar=dict(title="Correlation")
        ))
        fig.update_layout(height=500)
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("No weather features found in dataset")

with tab5:
    st.markdown("### Gaussian Mixture Model (GMM) Distribution")
    
    st.info("""
    STMGT uses a 3-component GMM for probabilistic predictions:
    - **Component 1:** Congested traffic (low speed)
    - **Component 2:** Moderate traffic (medium speed)
    - **Component 3:** Free flow (high speed)
    """)
    
    if 'speed_kmh' in df.columns:
        # Fit GMM
        speeds = df['speed_kmh'].values.reshape(-1, 1)
        gmm = GaussianMixture(n_components=3, random_state=42)
        gmm.fit(speeds)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Plot distribution with GMM overlay
            fig = go.Figure()
            
            # Histogram
            fig.add_trace(go.Histogram(
                x=df['speed_kmh'],
                nbinsx=100,
                histnorm='probability density',
                name='Data',
                marker_color='lightblue',
                opacity=0.6
            ))
            
            # GMM components
            x_range = np.linspace(df['speed_kmh'].min(), df['speed_kmh'].max(), 1000)
            colors = ['red', 'orange', 'green']
            
            for i in range(3):
                mean = gmm.means_[i][0]
                std = np.sqrt(gmm.covariances_[i][0][0])
                weight = gmm.weights_[i]
                
                y = weight * (1/(std * np.sqrt(2*np.pi))) * np.exp(-0.5*((x_range - mean)/std)**2)
                
                fig.add_trace(go.Scatter(
                    x=x_range,
                    y=y,
                    mode='lines',
                    name=f'Component {i+1}',
                    line=dict(width=3, color=colors[i])
                ))
            
            fig.update_layout(
                title="Speed Distribution with GMM Components (K=3)",
                xaxis_title="Speed (km/h)",
                yaxis_title="Density",
                height=500
            )
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            st.markdown("#### GMM Parameters")
            
            for i in range(3):
                mean = gmm.means_[i][0]
                std = np.sqrt(gmm.covariances_[i][0][0])
                weight = gmm.weights_[i]
                
                st.markdown(f"**Component {i+1}:**")
                st.write(f"- μ: {mean:.2f} km/h")
                st.write(f"- σ: {std:.2f} km/h")
                st.write(f"- Weight: {weight:.3f} ({weight*100:.1f}%)")
                st.write("")
            
            st.markdown("#### Component Assignment")
            labels = gmm.predict(speeds)
            for i in range(3):
                count = (labels == i).sum()
                pct = count / len(labels) * 100
                st.write(f"Component {i+1}: {pct:.1f}%")

with tab6:
    st.markdown("### Network Graph Topology")
    
    st.info("""
    Traffic network topology visualization:
    - **Nodes:** 62 road intersections
    - **Edges:** 144 directional road segments
    - Node size represents connectivity (degree)
    """)
    
    # Load adjacency matrix
    adj_matrix_path = PROJECT_ROOT / "cache" / "adjacency_matrix.npy"
    
    if adj_matrix_path.exists():
        adj_matrix = np.load(adj_matrix_path)
        
        # Create graph
        G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Network stats
            st.markdown("#### Network Statistics")
            st.write(f"- Nodes: {G.number_of_nodes()}")
            st.write(f"- Edges: {G.number_of_edges()}")
            st.write(f"- Density: {nx.density(G):.3f}")
            st.write(f"- Is Connected: {nx.is_weakly_connected(G)}")
            
            # Degree distribution
            degrees = [G.degree(n) for n in G.nodes()]
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=degrees,
                nbinsx=20,
                marker_color='#1f77b4'
            ))
            fig.update_layout(
                title="Node Degree Distribution",
                xaxis_title="Degree",
                yaxis_title="Count",
                height=400
            )
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            st.markdown("#### Top Connected Nodes")
            
            degree_dict = dict(G.degree())
            sorted_nodes = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:10]
            
            for node, degree in sorted_nodes:
                st.write(f"Node {node}: {degree} connections")
        
        # Network visualization (simplified)
        st.markdown("#### Network Visualization")
        st.info("Interactive network visualization - showing simplified layout")
        
        # Use spring layout
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f'Node {node}<br>Degree: {G.degree(node)}')
            node_size.append(G.degree(node) * 3)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                showscale=True,
                colorscale='YlOrRd',
                size=node_size,
                color=[G.degree(n) for n in G.nodes()],
                colorbar=dict(
                    title=dict(text="Node Degree", side='right'),
                    xanchor='left'
                )
            )
        )
        
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=0, l=0, r=0, t=0),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           height=600
                       ))
        
        st.plotly_chart(fig, width='stretch')
    
    else:
        st.warning(f"Adjacency matrix not found at {adj_matrix_path}")
        st.info("Please run data collection to generate network topology")

# Footer
st.divider()
st.caption("Tip: Use Extreme Augmented dataset for comprehensive pattern analysis")
