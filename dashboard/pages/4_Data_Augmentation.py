"""
Page 4: Data Augmentation
Configure augmentation parameters, compare datasets, and prepare commands.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.utils.command_blocks import show_command_block
from traffic_forecast.utils.conda import resolve_conda_executable

st.set_page_config(page_title="Data Augmentation", page_icon="", layout="wide")

st.title("Data Augmentation")
st.markdown("Tune augmentation hyperparameters, validate quality, and launch jobs.")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CONFIG_PATH = PROJECT_ROOT / "configs" / "augmentation_config.json"
ENV_NAME = os.environ.get("CONDA_ENV", "dsp")

DATASETS = {
    "Original": PROCESSED_DIR / "all_runs_combined.parquet",
    "Basic": PROCESSED_DIR / "all_runs_augmented.parquet",
    "Extreme": PROCESSED_DIR / "all_runs_extreme_augmented.parquet",
}


def _conda_run_args(script: str) -> list[str]:
    """Build the conda run invocation for the dsp environment."""
    return [
        resolve_conda_executable(),
        "run",
        "-n",
        ENV_NAME,
        "python",
        script,
    ]


def _load_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text())
        except json.JSONDecodeError:
            st.warning("Existing augmentation_config.json is invalid JSON. Reverting to defaults.")
    return {
        "basic": {
            "noise_std_speed": 2.0,
            "noise_std_weather": 0.1,
            "interpolation_steps": 2,
            "gmm_samples_per_run": 10,
            "target_multiplier": 20,
        },
        "extreme": {
            "noise_std_speed": 3.0,
            "noise_std_weather": 0.2,
            "interpolation_steps": 3,
            "gmm_samples_per_run": 20,
            "target_multiplier": 45,
        },
    }


def _cached_read(path: Path) -> Optional[pd.DataFrame]:
    @st.cache_data(show_spinner=False, ttl=900)
    def _read(parquet_path: str) -> Optional[pd.DataFrame]:
        local_path = Path(parquet_path)
        if not local_path.exists():
            return None
        try:
            return pd.read_parquet(local_path)
        except Exception as exc:  # pragma: no cover - defensive
            st.error(f"Failed to read {local_path.name}: {exc}")
            return None

    return _read(str(path))


config = _load_config()

# Tabs
CONF_TAB, COMPARE_TAB, QUALITY_TAB, RUN_TAB = st.tabs(
    [
        "Configuration",
        "Strategy Comparison",
        "Quality Validation",
        "Run Augmentation",
    ]
)

with CONF_TAB:
    st.markdown("### Augmentation Parameters")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Basic Augmentation")
        config["basic"]["noise_std_speed"] = st.slider(
            "Speed Noise (km/h)", 0.5, 5.0, float(config["basic"]["noise_std_speed"]), 0.1
        )
        config["basic"]["noise_std_weather"] = st.slider(
            "Weather Noise", 0.05, 0.5, float(config["basic"]["noise_std_weather"]), 0.05
        )
        config["basic"]["interpolation_steps"] = st.slider(
            "Interpolation Steps", 1, 5, int(config["basic"]["interpolation_steps"])
        )
        config["basic"]["gmm_samples_per_run"] = st.slider(
            "GMM Samples per Run", 5, 30, int(config["basic"]["gmm_samples_per_run"])
        )
        config["basic"]["target_multiplier"] = st.slider(
            "Target Multiplication Factor", 10, 30, int(config["basic"]["target_multiplier"])
        )
        st.caption(
            f"Approximate output: ~{38 * config['basic']['target_multiplier']:,} samples (from 38 base runs)."
        )

    with col2:
        st.markdown("#### Extreme Augmentation")
        config["extreme"]["noise_std_speed"] = st.slider(
            "Speed Noise (km/h)", 1.0, 10.0, float(config["extreme"]["noise_std_speed"]), 0.5
        )
        config["extreme"]["noise_std_weather"] = st.slider(
            "Weather Noise", 0.1, 1.0, float(config["extreme"]["noise_std_weather"]), 0.1
        )
        config["extreme"]["interpolation_steps"] = st.slider(
            "Interpolation Steps", 2, 10, int(config["extreme"]["interpolation_steps"])
        )
        config["extreme"]["gmm_samples_per_run"] = st.slider(
            "GMM Samples per Run", 10, 50, int(config["extreme"]["gmm_samples_per_run"])
        )
        config["extreme"]["target_multiplier"] = st.slider(
            "Target Multiplication Factor", 30, 60, int(config["extreme"]["target_multiplier"])
        )
        st.caption(
            f"Approximate output: ~{38 * config['extreme']['target_multiplier']:,} samples (from 38 base runs)."
        )

    st.divider()

    config_preview = json.dumps(config, indent=2)
    st.markdown("#### Preview JSON")
    st.code(config_preview, language="json")

    col_save, col_reset = st.columns([2, 1])
    with col_save:
        if st.button("Save Configuration", type="primary", width='stretch'):
            CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            CONFIG_PATH.write_text(config_preview)
            st.success(f"Configuration saved to `{CONFIG_PATH.relative_to(PROJECT_ROOT)}`")
    with col_reset:
        if st.button("Reset to Defaults", width='stretch'):
            config.clear()
            config.update(_load_config())
            st.experimental_rerun()

with COMPARE_TAB:
    st.markdown("### Strategy Comparison")

    basic_df = _cached_read(DATASETS["Basic"])
    extreme_df = _cached_read(DATASETS["Extreme"])

    col1, col2 = st.columns(2)

    def _render_summary(label: str, df: Optional[pd.DataFrame], path: Path) -> None:
        st.markdown(f"#### {label}")
        if df is None:
            st.warning(f"Dataset not found at `{path.relative_to(PROJECT_ROOT)}`")
            return
        runs = df["run_id"].nunique() if "run_id" in df.columns else len(df)
        st.metric("Runs", runs)
        st.metric("Records", f"{len(df):,}")
        st.metric("File Size", f"{path.stat().st_size / 1024 / 1024:.1f} MB")
        if "speed_kmh" in df.columns:
            stats = df["speed_kmh"].describe().to_dict()
            st.markdown(
                "\n".join(
                    [
                        f"- Mean: {stats['mean']:.2f} km/h",
                        f"- Std: {stats['std']:.2f} km/h",
                        f"- Min: {stats['min']:.2f} km/h",
                        f"- Max: {stats['max']:.2f} km/h",
                    ]
                )
            )

    with col1:
        _render_summary("Basic Augmentation", basic_df, DATASETS["Basic"])
    with col2:
        _render_summary("Extreme Augmentation", extreme_df, DATASETS["Extreme"])

    st.divider()

    if basic_df is not None and extreme_df is not None and "speed_kmh" in basic_df.columns:
        st.markdown("#### Speed Distribution Comparison")
        basic_sample = basic_df.sample(min(10000, len(basic_df)), random_state=42)
        extreme_sample = extreme_df.sample(min(10000, len(extreme_df)), random_state=42)
        fig = go.Figure()
        fig.add_histogram(
            x=basic_sample["speed_kmh"],
            name="Basic",
            opacity=0.6,
            nbinsx=50,
            marker_color="#1f77b4",
        )
        fig.add_histogram(
            x=extreme_sample["speed_kmh"],
            name="Extreme",
            opacity=0.6,
            nbinsx=50,
            marker_color="#ff7f0e",
        )
        fig.update_layout(
            barmode="overlay",
            xaxis_title="Speed (km/h)",
            yaxis_title="Frequency",
            height=420,
        )
        st.plotly_chart(fig, width='stretch')

with QUALITY_TAB:
    st.markdown("### Quality Validation")
    st.info(
        "Compare augmented datasets against the original distribution using statistical summaries and a "
        "Kolmogorov-Smirnov test."
    )

    original_df = _cached_read(DATASETS["Original"])
    if original_df is None:
        st.warning("Original dataset missing. Combine runs on the Data Overview page before validating.")
        st.stop()

    target_choice = st.radio(
        "Dataset to validate",
        ["Basic Augmented", "Extreme Augmented"],
        index=1,
    )
    candidate_df = basic_df if target_choice.startswith("Basic") else extreme_df

    if candidate_df is None:
        st.warning("Selected augmentation dataset is missing. Run augmentation first.")
    else:
        cols = st.columns(2)
        with cols[0]:
            st.markdown("#### Original Data Stats")
            st.write(original_df["speed_kmh"].describe()[["mean", "std", "min", "max"]])
        with cols[1]:
            st.markdown(f"#### {target_choice} Stats")
            st.write(candidate_df["speed_kmh"].describe()[["mean", "std", "min", "max"]])

        st.divider()
        st.markdown("#### Kolmogorov-Smirnov Test")
        try:
            from scipy import stats  # type: ignore

            ks_stat, p_value = stats.ks_2samp(
                original_df["speed_kmh"].sample(min(2000, len(original_df)), random_state=42),
                candidate_df["speed_kmh"].sample(min(2000, len(candidate_df)), random_state=99),
            )
            c1, c2, c3 = st.columns(3)
            c1.metric("KS Statistic", f"{ks_stat:.4f}")
            c2.metric("P-value", f"{p_value:.4f}")
            verdict = "Pass" if p_value > 0.05 else "Review"
            c3.metric("Verdict", verdict, delta=f"p>{0.05}" if verdict == "Pass" else "p<=0.05")
        except ModuleNotFoundError:
            st.warning("scipy not installed. Install it in the dsp environment to run the KS test.")

with RUN_TAB:
    st.markdown("### Launch Augmentation Jobs")

    original_ready = DATASETS["Original"].exists()
    if not original_ready:
        st.error(
            "Original combined dataset missing. Generate it on the Data Overview page before running augmentation."
        )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Basic Augmentation")
        st.caption("Balanced data boost. Runtime ≈ 5-10 minutes.")
        if st.button(
            "Prepare Basic Command",
            type="primary",
            width='stretch',
            disabled=not original_ready,
        ):
            show_command_block(
                _conda_run_args("scripts/data/augment_data_advanced.py"),
                cwd=PROJECT_ROOT,
                description="Run this command in a terminal to start basic augmentation.",
                success_hint="Keep the terminal open; progress logs stream there.",
            )
            st.success("Command prepared. Execute it manually to launch augmentation.")

    with col2:
        st.markdown("#### Extreme Augmentation")
        st.caption("Maximum diversity. Runtime ≈ 10-20 minutes.")
        if st.button(
            "Prepare Extreme Command",
            type="primary",
            width='stretch',
            disabled=not original_ready,
        ):
            show_command_block(
                _conda_run_args("scripts/data/augment_extreme.py"),
                cwd=PROJECT_ROOT,
                description="Run this command in a terminal to start extreme augmentation.",
                success_hint="Progress may take several minutes before the first log line appears.",
            )
            st.success("Command prepared. Execute it manually to launch augmentation.")

    st.divider()
    st.markdown("#### Post-run Checklist")
    st.markdown(
        "- Verify parquet files under `data/processed/`\n"
        "- Inspect `outputs/data_analysis_results.md` for summary statistics\n"
        "- Run the Quality Validation tab to confirm distributions"
    )

st.divider()
st.caption("Tip: Extreme augmentation is recommended before training production models.")
