"""
Page 2: Data Overview
Monitor collection outputs, parquet artifacts, and maintenance tasks.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.utils.command_blocks import show_command_block, show_command_list

st.set_page_config(page_title="Data Overview", page_icon="", layout="wide")

st.title("Data Overview")
st.markdown("Monitor data collections, processed parquet artifacts, and maintenance tasks")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_ROOT / "data" / "runs"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DATASETS: Dict[str, Path] = {
    "Original Combined": PROCESSED_DIR / "all_runs_combined.parquet",
    "Basic Augmented": PROCESSED_DIR / "all_runs_augmented.parquet",
    "Extreme Augmented": PROCESSED_DIR / "all_runs_extreme_augmented.parquet",
}


def _load_dataset(path: Path) -> Optional[pd.DataFrame]:
    """Load a parquet dataset with caching and graceful error handling."""

    @st.cache_data(show_spinner=False, ttl=900)
    def _cached_load(parquet_path: str) -> Optional[pd.DataFrame]:
        if not Path(parquet_path).exists():
            return None
        try:
            df = pd.read_parquet(parquet_path)
        except Exception as exc:  # pragma: no cover - defensive path
            st.error(f"Failed to read parquet file {parquet_path}: {exc}")
            return None
        return df

    return _cached_load(str(path))


def _run_count(df: pd.DataFrame) -> int:
    """Return unique run count if column exists, otherwise fallback to row count."""
    if "run_id" in df.columns:
        return int(df["run_id"].nunique())
    return int(len(df))


def _summarize_dataset(name: str, path: Path, baseline_runs: Optional[int]) -> Dict[str, Any]:
    if not path.exists():
        return {
            "Dataset": name,
            "Status": "Missing",
            "Rows": pd.NA,
            "Columns": pd.NA,
            "Size (MB)": pd.NA,
            "Modified": "-",
            "Run Ratio": pd.NA,
        }

    df = _load_dataset(path)
    if df is None:
        return {
            "Dataset": name,
            "Status": "Unreadable",
            "Rows": pd.NA,
            "Columns": pd.NA,
            "Size (MB)": pd.NA,
            "Modified": "-",
            "Run Ratio": pd.NA,
        }

    size_mb = round(path.stat().st_size / 1024 / 1024, 1)
    run_ratio: Optional[float] = None
    run_total = _run_count(df)
    if baseline_runs:
        run_ratio = run_total / baseline_runs

    return {
        "Dataset": name,
        "Status": "Available",
        "Rows": run_total if "run_id" not in df.columns else len(df),
        "Columns": len(df.columns),
        "Size (MB)": size_mb,
        "Modified": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
        "Run Ratio": round(run_ratio, 1) if run_ratio else pd.NA,
    }



def _delta_label(current: Optional[int], baseline: Optional[int]) -> str:
    if not current or not baseline or baseline == 0:
        return ""
    ratio = current / baseline
    if ratio == 1:
        return "1.0×"
    return f"{ratio:.1f}×"


# Tabs
COLLECTION_TAB, STATS_TAB, MANAGEMENT_TAB = st.tabs([
    "Data Collections",
    "Statistics",
    "Data Management",
])

with COLLECTION_TAB:
    st.markdown("### Collection Snapshot")

    original_df = _load_dataset(DATASETS["Original Combined"])
    baseline_runs = _run_count(original_df) if original_df is not None else None

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Raw Runs")
        run_dirs: Iterable[Path] = [d for d in RUNS_DIR.iterdir()] if RUNS_DIR.exists() else []
        run_dirs = [d for d in run_dirs if d.is_dir()]
        st.metric("Total Runs", len(run_dirs))
        if run_dirs:
            latest_run = max(run_dirs, key=lambda path: path.stat().st_mtime)
            st.caption(f"Latest run: {latest_run.name}")
        else:
            st.caption("No raw run directories detected")

    with col2:
        st.markdown("#### Basic Augmentation")
        basic_df = _load_dataset(DATASETS["Basic Augmented"])
        if basic_df is not None:
            run_total = _run_count(basic_df)
            st.metric(
                "Augmented Runs",
                run_total,
                delta=_delta_label(run_total, baseline_runs),
            )
            st.caption(f"File size {DATASETS['Basic Augmented'].stat().st_size / 1024 / 1024:.1f} MB")
        else:
            st.metric("Augmented Runs", 0)
            st.warning("Run `augment_data_advanced.py` to build this dataset")

    with col3:
        st.markdown("#### Extreme Augmentation")
        extreme_df = _load_dataset(DATASETS["Extreme Augmented"])
        if extreme_df is not None:
            run_total = _run_count(extreme_df)
            st.metric(
                "Extreme Runs",
                run_total,
                delta=_delta_label(run_total, baseline_runs),
            )
            st.caption(f"File size {DATASETS['Extreme Augmented'].stat().st_size / 1024 / 1024:.1f} MB")
        else:
            st.metric("Extreme Runs", 0)
            st.warning("Run `augment_extreme.py` to build this dataset")

    st.divider()

    st.markdown("### Dataset Summary")
    summary_rows = [
        _summarize_dataset(name, path, baseline_runs) for name, path in DATASETS.items()
    ]
    summary_df = pd.DataFrame(summary_rows)

    st.dataframe(summary_df, hide_index=True, width='stretch')

with STATS_TAB:
    st.markdown("### Dataset Explorer")

    dataset_choice = st.selectbox("Select dataset", list(DATASETS.keys()), index=2)
    dataset_path = DATASETS[dataset_choice]
    df = _load_dataset(dataset_path)

    if df is None:
        st.warning(
            f"Dataset `{dataset_choice}` is missing. Use the Data Management tab to rebuild it."
        )
    else:
        col1, col2, col3, col4 = st.columns(4)
        st.metric("Total Records", f"{len(df):,}")
        st.metric("Unique Runs", df["run_id"].nunique() if "run_id" in df.columns else "-")
        st.metric("Unique Edges", df["edge_id"].nunique() if "edge_id" in df.columns else "-")
        if "timestamp" in df.columns:
            timestamps = pd.to_datetime(df["timestamp"], errors="coerce")
            if timestamps.notna().any():
                span = timestamps.max() - timestamps.min()
                st.metric("Coverage", f"{span.days} days")
            else:
                st.metric("Coverage", "-")
        else:
            st.metric("Coverage", "-")

        st.divider()

        max_rows = st.slider("Sample size for charts", 5000, 50000, 10000, step=5000)
        df_sample = df.sample(min(max_rows, len(df)), random_state=42) if len(df) > max_rows else df

        if "speed_kmh" in df_sample.columns:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Speed Distribution")
                fig = go.Figure()
                fig.add_histogram(
                    x=df_sample["speed_kmh"],
                    nbinsx=60,
                    marker_color="#1f77b4",
                )
                fig.update_layout(
                    xaxis_title="Speed (km/h)",
                    yaxis_title="Frequency",
                    height=380,
                )
                st.plotly_chart(fig, width='stretch')

            with col2:
                st.markdown("#### Speed Statistics")
                stats = df_sample["speed_kmh"].describe()
                st.dataframe(
                    {
                        "Metric": stats.index.tolist(),
                        "Value": [f"{value:.2f}" for value in stats.values],
                    },
                    hide_index=True,
                    width='stretch',
                )

        if {"timestamp", "speed_kmh"}.issubset(df_sample.columns):
            st.markdown("#### Hourly Traffic Pattern (sample)")
            copy_df = df_sample.copy()
            copy_df["hour"] = pd.to_datetime(copy_df["timestamp"], errors="coerce").dt.hour
            hourly_avg = copy_df.groupby("hour")["speed_kmh"].mean().reset_index()
            hourly_fig = go.Figure()
            hourly_fig.add_scatter(
                x=hourly_avg["hour"],
                y=hourly_avg["speed_kmh"],
                mode="lines+markers",
                line=dict(color="#ff7f0e", width=3),
            )
            hourly_fig.update_layout(
                xaxis_title="Hour of Day",
                yaxis_title="Average Speed (km/h)",
                height=380,
            )
            st.plotly_chart(hourly_fig, width='stretch')

with MANAGEMENT_TAB:
    st.markdown("### Maintenance Commands")
    st.info(
        "Run the commands below from a terminal in the project root. The Conda variant is recommended; "
        "the fallback command uses the active Python interpreter."
    )

    if st.button("Combine all runs", width='stretch'):
        show_command_list(
            [
                [
                    "conda",
                    "run",
                    "-n",
                    os.environ.get("CONDA_ENV", "dsp"),
                    "--no-capture-output",
                    "python",
                    "scripts/data/combine_runs.py",
                ],
                [
                    "python",
                    "scripts/data/combine_runs.py",
                ],
            ],
            description="Merge raw run folders into `all_runs_combined.parquet`.",
            cwd=PROJECT_ROOT,
        )
        st.success("Command prepared. Execute it manually to begin the merge job.")

    st.divider()

    cache_script = PROJECT_ROOT / "scripts" / "data" / "rebuild_cache.py"
    if cache_script.exists():
        if st.button("Rebuild cache", width='stretch'):
            show_command_block(
                [
                    "conda",
                    "run",
                    "-n",
                    os.environ.get("CONDA_ENV", "dsp"),
                    "--no-capture-output",
                    "python",
                    str(cache_script.relative_to(PROJECT_ROOT)),
                ],
                cwd=PROJECT_ROOT,
                description="Regenerate adjacency matrix and topology caches.",
            )
            st.success("Command prepared. Run it in a terminal to rebuild the cache.")
    else:
        st.warning("Cache rebuild script not found at `scripts/data/rebuild_cache.py`.")

st.divider()
st.caption("Tip: Monitor the extreme augmented dataset for the most complete coverage of traffic scenarios.")
