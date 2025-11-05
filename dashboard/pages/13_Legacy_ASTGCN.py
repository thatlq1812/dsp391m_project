"""
Page 13: ASTGCN Baseline
Expose the legacy notebook workflow via copyable commands.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit import components

from dashboard.utils.command_blocks import show_command_block

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "astgcn"
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "training" / "train_astcgn.py"

st.set_page_config(page_title="ASTGCN Baseline", page_icon="", layout="wide")

st.title("ASTGCN Baseline")
st.markdown(
    "This page preserves the original notebook pipeline. Use the command below to rerun the baseline."
)

if st.button("Prepare ASTGCN command", type="primary", width='stretch'):
    show_command_block(
        [
            "conda",
            "run",
            "-n",
            "dsp",
            "--no-capture-output",
            "python",
            str(SCRIPT_PATH.relative_to(PROJECT_ROOT)),
        ],
        cwd=PROJECT_ROOT,
        description="Execute the command in a terminal to run the ASTGCN baseline.",
        success_hint="Outputs will be written to `outputs/astgcn/<timestamp>/`.",
    )
    st.success("Command prepared. Run it manually to reproduce the notebook baseline.")

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
    selected = st.selectbox("Select run", [run.name for run in run_dirs])
    run_dir = OUTPUT_ROOT / selected

    summary_file = run_dir / "summary.txt"
    describe_file = run_dir / "descriptive_stats.csv"
    dist_file = run_dir / "distribution_plots.png"
    scatter_file = run_dir / "temperature_precipitation_scatter.png"
    heatmap_file = run_dir / "correlation_heatmap.png"
    node_file = run_dir / "node_average.csv"
    map_file = run_dir / "traffic_status_map.html"
    congestion_plot = run_dir / "congested_road_types.png"
    congestion_data = run_dir / "congested_edges.csv"

    st.markdown("### Outputs")

    if summary_file.exists():
        st.markdown("#### Summary")
        st.code(summary_file.read_text(), language="text")

    if describe_file.exists():
        st.markdown("#### Descriptive statistics")
        st.dataframe(pd.read_csv(describe_file), hide_index=True, width='stretch')

    image_pairs = [
        ("Distribution charts", dist_file),
        ("Temperature vs Speed", scatter_file),
        ("Correlation heatmap", heatmap_file),
        ("Congested road types", congestion_plot),
    ]
    for title, image_path in image_pairs:
        if image_path.exists():
            st.markdown(f"#### {title}")
            st.image(str(image_path))

    if node_file.exists():
        st.markdown("#### Node averages")
        st.dataframe(pd.read_csv(node_file), hide_index=True, width='stretch')

    if map_file.exists():
        st.markdown("#### Traffic map")
        components.v1.html(map_file.read_text(), height=600, scrolling=False)

    if congestion_data.exists():
        st.markdown("#### Congested edges")
        st.dataframe(pd.read_csv(congestion_data).head(200), width='stretch')
else:
    st.info("Run the baseline command once to populate output artifacts.")

st.divider()
st.caption("Tip: keep the legacy baseline for regression checks when introducing major model changes.")
