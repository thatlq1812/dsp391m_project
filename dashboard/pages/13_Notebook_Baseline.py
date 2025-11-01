"""Page 13: ASTGCN
Expose the original notebook workflow inside the dashboard."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit import components

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "astgcn"
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "training" / "train_astcgn.py"

st.title("ASTGCN")
st.markdown(
    "This page replays the exact notebook steps without modifications so the baseline remains intact."
)

col_run, col_info = st.columns([1, 3])

with col_run:
    if st.button("Run ASTGCN", type="primary"):
        command = [
            "conda",
            "run",
            "-n",
            "dsp",
            "--no-capture-output",
            "python",
            str(SCRIPT_PATH.relative_to(PROJECT_ROOT)),
        ]
        try:
            subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            st.success("Baseline run completed successfully.")
        except subprocess.CalledProcessError as exc:
            st.error("Baseline run failed. See details below.")
            st.code(exc.stderr or str(exc))

with col_info:
    st.markdown(
        "Outputs are stored under `outputs/astgcn/<timestamp>` following the original notebook flow."
    )

st.divider()

if OUTPUT_ROOT.exists():
    run_dirs = sorted(
        [d for d in OUTPUT_ROOT.iterdir() if d.is_dir()],
        key=lambda path: path.name,
        reverse=True,
    )
else:
    run_dirs = []

if run_dirs:
    run_options = [run.name for run in run_dirs]
    selected = st.selectbox("Select baseline run", run_options)
    selected_dir = OUTPUT_ROOT / selected

    summary_file = selected_dir / "summary.txt"
    describe_file = selected_dir / "descriptive_stats.csv"
    dist_file = selected_dir / "distribution_plots.png"
    scatter_file = selected_dir / "temperature_precipitation_scatter.png"
    heatmap_file = selected_dir / "correlation_heatmap.png"
    node_file = selected_dir / "node_average.csv"
    map_file = selected_dir / "traffic_status_map.html"
    congestion_plot = selected_dir / "congested_road_types.png"
    congestion_data = selected_dir / "congested_edges.csv"

    st.markdown("### Notebook Outputs")

    if summary_file.exists():
        st.markdown("#### Summary")
        st.code(summary_file.read_text(), language="text")

    if describe_file.exists():
        st.markdown("#### Descriptive Statistics")
        describe_df = pd.read_csv(describe_file)
        st.dataframe(describe_df, hide_index=True, use_container_width=True)

    if dist_file.exists():
        st.markdown("#### Distribution Charts")
        st.image(str(dist_file))

    if scatter_file.exists():
        st.markdown("#### Temperature vs Speed")
        st.image(str(scatter_file))

    if heatmap_file.exists():
        st.markdown("#### Correlation Heatmap")
        st.image(str(heatmap_file))

    if congestion_plot.exists():
        st.markdown("#### Congested Road Types")
        st.image(str(congestion_plot))

    if node_file.exists():
        st.markdown("#### Node Averages")
        node_df = pd.read_csv(node_file)
        st.dataframe(node_df, hide_index=True, use_container_width=True)

    if map_file.exists():
        st.markdown("#### Traffic Status Map")
        components.v1.html(map_file.read_text(), height=600, scrolling=False)

    if congestion_data.exists():
        st.markdown("#### Congested Edges")
        congestion_df = pd.read_csv(congestion_data)
        st.dataframe(congestion_df.head(200), use_container_width=True)
else:
    st.info("Run the ASTGCN to populate visualization artifacts.")
