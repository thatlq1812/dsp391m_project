"""
Page 8: Predictions
Prototype inference workflow with simulated outputs (real model loading TBD).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Predictions", page_icon="", layout="wide")

st.title("Traffic Predictions (Prototype)")
st.markdown(
    "This page demonstrates the intended prediction UX using simulated values. "
    "Wire the production inference script to replace the placeholders."
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

TAB_MODEL, TAB_REALTIME, TAB_SCENARIO, TAB_ALERTS, TAB_EXPORT = st.tabs(
    [
        "Model Selection",
        "Real-time Demo",
        "Scenario Simulation",
        "Alerts",
        "Export",
    ]
)

if "predictions" not in st.session_state:
    st.session_state.predictions: Optional[Dict[str, np.ndarray]] = None

with TAB_MODEL:
    st.markdown("### Choose a Model Artifact")
    if not OUTPUTS_DIR.exists():
        st.warning("No trained models available. Train a model first.")
    else:
        model_dirs = sorted(
            [d for d in OUTPUTS_DIR.iterdir() if d.is_dir() and d.name.startswith("stmgt")],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not model_dirs:
            st.warning("No STMGT runs detected.")
        else:
            run_name = st.selectbox("Select model run", [d.name for d in model_dirs])
            run_dir = OUTPUTS_DIR / run_name
            config_path = run_dir / "config.json"
            test_results = run_dir / "test_results.json"
            checkpoint = run_dir / "best_model.pt"

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Metadata")
                if config_path.exists():
                    st.json(json.loads(config_path.read_text()))
                else:
                    st.info("Config file missing.")
            with col2:
                st.markdown("#### Test Metrics")
                if test_results.exists():
                    st.json(json.loads(test_results.read_text()))
                else:
                    st.info("Run evaluation to generate `test_results.json`." )
                if checkpoint.exists():
                    st.success("Model checkpoint found: `best_model.pt`")
                else:
                    st.warning("Model checkpoint not found.")

with TAB_REALTIME:
    st.markdown("### Simulated Real-time Inference")
    st.info(
        "The section below uses randomised outputs to preview the dashboard UX. "
        "Integrate the actual inference script to replace the simulation."
    )
    horizon = st.slider("Forecast horizon (timesteps)", 6, 24, 12, step=6)
    seed = st.number_input("Simulation seed", min_value=0, value=42)

    if st.button("Generate demo predictions", type="primary"):
        np.random.seed(int(seed))
        num_edges = 62
        mean = np.clip(np.random.normal(40, 12, (num_edges, horizon)), 5, 80)
        std = np.random.uniform(2, 7, (num_edges, horizon))
        demo = {
            "mean": mean,
            "std": std,
            "lower_80": mean - 1.28 * std,
            "upper_80": mean + 1.28 * std,
            "lower_95": mean - 1.96 * std,
            "upper_95": mean + 1.96 * std,
        }
        st.session_state.predictions = demo
        st.success("Demo predictions generated. Select an edge to visualise the forecast.")

    if st.session_state.predictions is not None:
        edge = st.slider("Edge ID", 0, 61, 0)
        preds = st.session_state.predictions
        x_axis = np.arange(preds["mean"].shape[1])
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=x_axis, y=preds["mean"][edge], mode="lines+markers", name="Mean")
        )
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([x_axis, x_axis[::-1]]),
                y=np.concatenate([preds["upper_80"][edge], preds["lower_80"][edge][::-1]]),
                fill="toself",
                fillcolor="rgba(31,119,180,0.25)",
                line=dict(color="rgba(255,255,255,0)"),
                name="80% CI",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([x_axis, x_axis[::-1]]),
                y=np.concatenate([preds["upper_95"][edge], preds["lower_95"][edge][::-1]]),
                fill="toself",
                fillcolor="rgba(31,119,180,0.12)",
                line=dict(color="rgba(255,255,255,0)"),
                name="95% CI",
            )
        )
        fig.update_layout(
            xaxis_title="Timestep (15 min)",
            yaxis_title="Speed (km/h)",
            height=420,
        )
        st.plotly_chart(fig, width='stretch')

with TAB_SCENARIO:
    st.markdown("### Scenario Simulation")
    st.info("Estimate how traffic might react to weather or demand changes using heuristic modifiers.")

    weather = st.selectbox("Weather", ["Clear", "Light Rain", "Heavy Rain", "Storm"], index=0)
    time_of_day = st.selectbox("Time of day", ["Morning Rush", "Midday", "Evening Rush", "Night"], index=0)
    weekend = st.checkbox("Weekend", value=False)

    if st.button("Run scenario", type="secondary"):
        np.random.seed(7)
        baseline = np.random.normal(45, 10, 62)
        weather_scale = {"Clear": 1.0, "Light Rain": 0.9, "Heavy Rain": 0.75, "Storm": 0.6}[weather]
        time_scale = {"Morning Rush": 0.7, "Midday": 1.0, "Evening Rush": 0.65, "Night": 1.1}[time_of_day]
        weekend_scale = 1.15 if weekend else 1.0
        scenario = baseline * weather_scale * time_scale * weekend_scale
        st.session_state.scenario = {
            "baseline": baseline,
            "scenario": scenario,
            "weather": weather,
            "time": time_of_day,
            "weekend": weekend,
        }
        st.success("Scenario computed. Review the comparison below.")

    if "scenario" in st.session_state:
        results = st.session_state.scenario
        delta = results["scenario"].mean() - results["baseline"].mean()
        col1, col2 = st.columns(2)
        col1.metric("Baseline mean", f"{results['baseline'].mean():.1f} km/h")
        col2.metric(
            "Scenario mean",
            f"{results['scenario'].mean():.1f} km/h",
            delta=f"{delta:.1f}",
            delta_color="inverse",
        )
        congestion = (results["scenario"] < 20).sum()
        st.metric("Edges under congestion", congestion)
        fig = go.Figure()
        fig.add_trace(go.Box(y=results["baseline"], name="Baseline"))
        fig.add_trace(go.Box(y=results["scenario"], name="Scenario"))
        st.plotly_chart(fig, width='stretch')

with TAB_ALERTS:
    st.markdown("### Alert Prototype")
    if st.session_state.predictions is None:
        st.info("Generate demo predictions to simulate alerts.")
    else:
        threshold = st.slider("Congestion threshold (km/h)", 10, 40, 20, step=5)
        preds = st.session_state.predictions
        alerts = []
        for edge_id in range(preds["mean"].shape[0]):
            for ts, value in enumerate(preds["mean"][edge_id]):
                if value < threshold:
                    severity = "High" if value < threshold * 0.7 else "Medium"
                    alerts.append({
                        "edge": edge_id,
                        "timestep": ts,
                        "speed": round(value, 1),
                        "severity": severity,
                    })
        if alerts:
            st.warning(f"{len(alerts)} alerts triggered")
            st.dataframe(alerts, hide_index=True, width='stretch')
        else:
            st.success("No edges predicted below the threshold in the current simulation.")

with TAB_EXPORT:
    st.markdown("### Export Guidance")
    st.info(
        "When real inference is wired up, log predictions to `outputs/predictions/<timestamp>.parquet` "
        "and offer download buttons here. For now, copy scenario results manually if needed."
    )
    if st.session_state.predictions is not None:
        st.caption("Use the data frame export menu above each table to download CSV snapshots.")

st.divider()
st.caption("Tip: Replace the simulation by calling `scripts/inference/run_prediction.py --model <run>` once available.")
