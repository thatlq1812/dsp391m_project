"""
Page 5: Data Visualization
Interactive analytics for traffic speed, temporal trends, and topology.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly import express as px
from sklearn.mixture import GaussianMixture

st.set_page_config(page_title="Visualization", page_icon="", layout="wide")

st.title("Data Visualization")
st.markdown("Explore traffic patterns, temporal trends, and graph structure.")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DATASET_OPTIONS = {
    "Original Combined": PROCESSED_DIR / "all_runs_combined.parquet",
    "Basic Augmented": PROCESSED_DIR / "all_runs_augmented.parquet",
    "Extreme Augmented": PROCESSED_DIR / "all_runs_extreme_augmented.parquet",
}


def _load_dataset(path: Path) -> Optional[pd.DataFrame]:
    @st.cache_data(show_spinner=False, ttl=900)
    def _cached(parquet_path: str) -> Optional[pd.DataFrame]:
        local_path = Path(parquet_path)
        if not local_path.exists():
            return None
        try:
            return pd.read_parquet(local_path)
        except Exception as exc:  # pragma: no cover - defensive
            st.error(f"Failed to read {local_path.name}: {exc}")
            return None

    return _cached(str(path))


def _apply_sample(df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    if len(df) <= sample_size:
        return df
    return df.sample(sample_size, random_state=42)


def _hourly_boxplot(df: pd.DataFrame) -> go.Figure:
    box_data = [df[df["hour"] == hour]["speed_kmh"] for hour in range(24)]
    fig = go.Figure()
    for hour, data in enumerate(box_data):
        fig.add_trace(
            go.Box(
                y=data,
                name=f"{hour:02d}:00",
                marker_color="#1f77b4",
                boxpoints="outliers",
            )
        )
    fig.update_layout(
        xaxis_title="Hour of Day",
        yaxis_title="Speed (km/h)",
        height=420,
        showlegend=False,
    )
    return fig


selected_label = st.selectbox("Dataset", list(DATASET_OPTIONS.keys()), index=2)
selected_path = DATASET_OPTIONS[selected_label]

if not selected_path.exists():
    st.error(
        f"Dataset `{selected_label}` missing. Use the Data Overview and Augmentation pages to generate it."
    )
    st.stop()

df_full = _load_dataset(selected_path)
if df_full is None:
    st.stop()

st.success(f"Loaded {len(df_full):,} rows from `{selected_label}`")

sample_max = st.slider("Rows for interactive charts", 5_000, 100_000, 25_000, step=5_000)
df = _apply_sample(df_full, sample_max)

if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["hour"] = df["timestamp"].dt.hour
    df["dow"] = df["timestamp"].dt.dayofweek

TAB_SPEED, TAB_TEMPORAL, TAB_SPATIAL, TAB_WEATHER, TAB_GMM, TAB_GRAPH = st.tabs(
    [
        "Speed Distribution",
        "Temporal Patterns",
        "Spatial Analysis",
        "Weather Correlation",
        "GMM",
        "Graph Topology",
    ]
)

with TAB_SPEED:
    st.markdown("### Speed Distribution")
    if "speed_kmh" not in df.columns:
        st.info("`speed_kmh` column missing.")
    else:
        col1, col2 = st.columns([2, 1])
        with col1:
            fig = go.Figure()
            fig.add_histogram(
                x=df["speed_kmh"],
                nbinsx=80,
                marker_color="#1f77b4",
            )
            fig.update_layout(
                xaxis_title="Speed (km/h)",
                yaxis_title="Frequency",
                height=420,
            )
            st.plotly_chart(fig, width='stretch')
        with col2:
            st.markdown("#### Summary")
            st.dataframe(
                df["speed_kmh"].describe().round(2).to_frame("Value"),
                width='stretch',
            )

        if "hour" in df.columns:
            st.markdown("#### Hourly Boxplot (sample)")
            st.plotly_chart(_hourly_boxplot(df), width='stretch')

with TAB_TEMPORAL:
    st.markdown("### Temporal Patterns")
    if {"timestamp", "speed_kmh"}.issubset(df.columns):
        hourly_avg = df.groupby("hour")["speed_kmh"].mean().reset_index()
        fig_hourly = go.Figure()
        fig_hourly.add_trace(
            go.Scatter(
                x=hourly_avg["hour"],
                y=hourly_avg["speed_kmh"],
                mode="lines+markers",
                line=dict(color="#ff7f0e", width=3),
            )
        )
        fig_hourly.update_layout(
            xaxis_title="Hour of Day",
            yaxis_title="Average Speed (km/h)",
            height=380,
        )
        st.plotly_chart(fig_hourly, width='stretch')

        weekend_avg = df.groupby(["hour", df["dow"].isin([5, 6])])["speed_kmh"].mean().reset_index()
        fig_week = go.Figure()
        for is_weekend, group in weekend_avg.groupby("dow"):
            label = "Weekend" if is_weekend else "Weekday"
            fig_week.add_trace(
                go.Scatter(
                    x=group["hour"],
                    y=group["speed_kmh"],
                    mode="lines+markers",
                    name=label,
                )
            )
        fig_week.update_layout(
            xaxis_title="Hour of Day",
            yaxis_title="Average Speed (km/h)",
            height=380,
        )
        st.plotly_chart(fig_week, width='stretch')

        pivot = df.pivot_table(
            values="speed_kmh",
            index="hour",
            columns="dow",
            aggfunc="mean",
        )
        heatmap = go.Figure(
            data=go.Heatmap(
                z=pivot.values,
                x=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                y=list(range(24)),
                colorscale="RdYlGn",
                colorbar=dict(title="km/h"),
            )
        )
        heatmap.update_layout(height=460, xaxis_title="Day of Week", yaxis_title="Hour")
        st.plotly_chart(heatmap, width='stretch')
    else:
        st.info("Timestamp data unavailable for temporal charts.")

with TAB_SPATIAL:
    st.markdown("### Spatial Analysis")
    if {"edge_id", "speed_kmh"}.issubset(df.columns):
        edge_stats = (
            df.groupby("edge_id")["speed_kmh"]
            .agg(["mean", "std", "min", "max", "count"])
            .reset_index()
        )
        slowest = edge_stats.nsmallest(20, "mean")
        fastest = edge_stats.nlargest(20, "mean")

        col1, col2 = st.columns(2)
        with col1:
            fig_slow = px.bar(
                slowest,
                x="mean",
                y="edge_id",
                orientation="h",
                title="Slowest Edges",
                labels={"mean": "Average Speed (km/h)", "edge_id": "Edge"},
                color="mean",
                color_continuous_scale="Reds",
            )
            st.plotly_chart(fig_slow, width='stretch')
        with col2:
            fig_fast = px.bar(
                fastest,
                x="mean",
                y="edge_id",
                orientation="h",
                title="Fastest Edges",
                labels={"mean": "Average Speed (km/h)", "edge_id": "Edge"},
                color="mean",
                color_continuous_scale="Greens",
            )
            st.plotly_chart(fig_fast, width='stretch')

        scatter = go.Figure(
            data=go.Scatter(
                x=edge_stats["mean"],
                y=edge_stats["std"],
                mode="markers",
                marker=dict(
                    size=8,
                    color=edge_stats["count"],
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Samples"),
                ),
                text=edge_stats["edge_id"],
                hovertemplate="Edge %{text}<br>Mean %{x:.1f} km/h<br>Std %{y:.1f} km/h",
            )
        )
        scatter.update_layout(
            xaxis_title="Mean Speed (km/h)",
            yaxis_title="Std Dev (km/h)",
            height=420,
        )
        st.plotly_chart(scatter, width='stretch')
    else:
        st.info("Edge-level statistics unavailable.")

with TAB_WEATHER:
    st.markdown("### Weather Correlation")
    weather_cols = [
        col
        for col in df.columns
        if "weather" in col.lower() or col in {"temperature_celsius", "rain_mm", "wind_speed_kmh"}
    ]
    if weather_cols and "speed_kmh" in df.columns:
        weather_feature = st.selectbox("Weather Feature", weather_cols)
        valid = df[[weather_feature, "speed_kmh"]].dropna()
        if len(valid) < 20:
            st.warning("Not enough data points after removing NaN values.")
        else:
            scatter = go.Figure()
            scatter.add_trace(
                go.Scatter(
                    x=valid[weather_feature],
                    y=valid["speed_kmh"],
                    mode="markers",
                    opacity=0.5,
                    marker=dict(size=5, color=valid["speed_kmh"], colorscale="RdYlGn", showscale=True),
                )
            )
            scatter.update_layout(
                xaxis_title=weather_feature,
                yaxis_title="Speed (km/h)",
                height=420,
            )
            st.plotly_chart(scatter, width='stretch')

            corr_cols = ["speed_kmh", weather_feature]
            corr = valid[corr_cols].corr()
            heat = go.Figure(
                data=go.Heatmap(
                    z=corr.values,
                    x=corr.columns,
                    y=corr.columns,
                    zmid=0,
                    colorscale="RdBu",
                    colorbar=dict(title="Correlation"),
                )
            )
            heat.update_layout(height=360)
            st.plotly_chart(heat, width='stretch')
    else:
        st.info("Weather columns unavailable in this dataset.")

with TAB_GMM:
    st.markdown("### Gaussian Mixture Model")
    if "speed_kmh" in df.columns:
        st.caption("Fits a 3-component GMM to the sampled dataset.")
        speeds = df["speed_kmh"].values.reshape(-1, 1)
        gmm = GaussianMixture(n_components=3, random_state=42)
        gmm.fit(speeds)
        x_axis = np.linspace(speeds.min(), speeds.max(), 500)

        fig = go.Figure()
        fig.add_histogram(
            x=df["speed_kmh"],
            nbinsx=80,
            histnorm="probability density",
            name="Data",
            marker_color="rgba(31,119,180,0.35)",
        )
        colors = ["#d62728", "#ff7f0e", "#2ca02c"]
        for idx in range(gmm.n_components):
            mean = gmm.means_[idx, 0]
            std = math.sqrt(gmm.covariances_[idx, 0, 0])
            weight = gmm.weights_[idx]
            density = weight * (1 / (std * math.sqrt(2 * math.pi))) * np.exp(
                -0.5 * ((x_axis - mean) / std) ** 2
            )
            fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=density,
                    mode="lines",
                    name=f"Component {idx + 1}",
                    line=dict(color=colors[idx], width=3),
                )
            )
        fig.update_layout(
            xaxis_title="Speed (km/h)",
            yaxis_title="Density",
            height=420,
        )
        st.plotly_chart(fig, width='stretch')

        cols = st.columns(3)
        for idx in range(gmm.n_components):
            cols[idx].metric(
                f"Comp {idx + 1} mean",
                f"{gmm.means_[idx, 0]:.1f} km/h",
                delta=f"weight {gmm.weights_[idx]:.2f}",
            )
    else:
        st.info("speed_kmh column required for GMM analysis.")

with TAB_GRAPH:
    st.markdown("### Graph Topology Preview")
    topo_json = PROJECT_ROOT / "cache" / "overpass_topology.json"
    if not topo_json.exists():
        st.warning("Topology cache missing. Rebuild it from the Data Overview page.")
    else:
        import json

        graph_data = json.loads(topo_json.read_text())
        graph = nx.node_link_graph(graph_data)
        st.metric("Nodes", graph.number_of_nodes())
        st.metric("Edges", graph.number_of_edges())
        degree_series = pd.Series(dict(graph.degree()))
        fig_degree = px.histogram(degree_series, nbins=20, labels={"value": "Degree", "count": "Frequency"})
        st.plotly_chart(fig_degree, width='stretch')

st.divider()
st.caption("Tip: Increase the sample size slider when you need higher-resolution plots.")
