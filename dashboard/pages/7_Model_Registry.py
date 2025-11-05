"""
Page 7: Model Registry
Inspect trained STMGT artifacts and supply manual management steps.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st

from dashboard.utils.command_blocks import show_command_block

st.set_page_config(page_title="Model Registry", page_icon="", layout="wide")

st.title("Model Registry")
st.markdown("Review training outputs, tag releases, and manage storage.")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

TAB_MODELS, TAB_COMPARE, TAB_TAGS, TAB_STORAGE = st.tabs(
    [
        "Model Versions",
        "Performance Comparison",
        "Model Tagging",
        "Artifact Storage",
    ]
)

with TAB_MODELS:
    st.markdown("### Registered Models")
    if not OUTPUTS_DIR.exists():
        st.info("No outputs directory detected. Train a model to populate this page.")
    else:
        model_dirs: List[Path] = sorted(
            [d for d in OUTPUTS_DIR.iterdir() if d.is_dir() and d.name.startswith("stmgt")],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not model_dirs:
            st.warning("No STMGT run folders found.")
        else:
            rows = []
            for model_dir in model_dirs:
                config_path = model_dir / "config.json"
                results_path = model_dir / "test_results.json"
                history_path = model_dir / "training_history.csv"
                mae = r2 = "-"
                if results_path.exists():
                    try:
                        metrics = json.loads(results_path.read_text())
                        if isinstance(metrics.get("mae"), (int, float)):
                            mae = f"{metrics['mae']:.3f}"
                        if isinstance(metrics.get("r2"), (int, float)):
                            r2 = f"{metrics['r2']:.3f}"
                    except json.JSONDecodeError:
                        st.warning(f"Invalid JSON in {results_path.name}")
                elif history_path.exists():
                    history_df = pd.read_csv(history_path)
                    if "val_mae" in history_df.columns and history_df["val_mae"].notna().any():
                        mae = f"{history_df['val_mae'].min():.3f}"
                    if "val_r2" in history_df.columns and history_df["val_r2"].notna().any():
                        r2 = f"{history_df['val_r2'].max():.3f}"

                rows.append(
                    {
                        "Run": model_dir.name,
                        "Created": datetime.fromtimestamp(model_dir.stat().st_ctime).strftime("%Y-%m-%d %H:%M"),
                        "MAE": mae,
                        "R²": r2,
                        "HiddenDim": json.loads(config_path.read_text()).get("model", {}).get("hidden_dim", "-")
                        if config_path.exists()
                        else "-",
                    }
                )
            st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')

            st.divider()
            selected = st.selectbox("Inspect model", [d.name for d in model_dirs])
            model_path = OUTPUTS_DIR / selected
            st.markdown("#### Files")
            files = [f for f in model_path.glob("**/*") if f.is_file()]
            if files:
                for f in files[:12]:
                    size_mb = f.stat().st_size / 1024 / 1024
                    st.write(f"{f.relative_to(PROJECT_ROOT)} ({size_mb:.2f} MB)")
            else:
                st.info("No files found in this run directory.")

with TAB_COMPARE:
    st.markdown("### Performance Comparison")
    if not OUTPUTS_DIR.exists():
        st.info("Train models to enable comparison.")
    else:
        rows = []
        for model_dir in OUTPUTS_DIR.iterdir():
            if not model_dir.is_dir() or not model_dir.name.startswith("stmgt"):
                continue
            results_path = model_dir / "test_results.json"
            if results_path.exists():
                try:
                    results = json.loads(results_path.read_text())
                except json.JSONDecodeError:
                    continue
                rows.append(
                    {
                        "Run": model_dir.name,
                        "MAE": results.get("mae"),
                        "RMSE": results.get("rmse"),
                        "R²": results.get("r2"),
                        "MAPE": results.get("mape"),
                    }
                )
        if rows:
            df_compare = pd.DataFrame(rows)
            st.bar_chart(df_compare.set_index("Run")["MAE"])
            st.dataframe(df_compare, hide_index=True, width='stretch')
        else:
            st.info("No `test_results.json` files available yet.")

with TAB_TAGS:
    st.markdown("### Model Tagging Guidance")
    st.info(
        "Tagging is managed through Git and release notes. Use the commands below to record production-ready models."
    )

    if st.button("Prepare tagging command", type="primary", width='stretch'):
        show_command_block(
            [
                "git",
                "tag",
                "-a",
                "model-v2",
                "<commit-hash>",
            ],
            cwd=PROJECT_ROOT,
            description="Create an annotated tag pointing to the commit associated with the trained model.",
            success_hint="Replace `<commit-hash>` with the commit that produced the run.",
        )
        st.success("Command prepared. Execute it in a terminal after updating the hash.")

    st.markdown(
        "- Document deployment status in `docs/CHANGELOG.md`\n"
        "- Store evaluation charts alongside each run in `outputs/<run>/plots/`\n"
        "- For promotion, copy artifacts to long-term storage (e.g., GCS bucket)"
    )

with TAB_STORAGE:
    st.markdown("### Storage Overview")
    if not OUTPUTS_DIR.exists():
        st.info("No outputs to measure yet.")
    else:
        total_size = sum(f.stat().st_size for f in OUTPUTS_DIR.rglob("*") if f.is_file()) / (1024 ** 3)
        model_count = len([d for d in OUTPUTS_DIR.iterdir() if d.is_dir() and d.name.startswith("stmgt")])
        st.metric("Total Size", f"{total_size:.2f} GB")
        st.metric("Model Runs", model_count)
        average = total_size / model_count if model_count else 0
        st.metric("Average Size", f"{average:.2f} GB")

        st.divider()
        st.markdown("#### Cleanup Command")
        if st.button("Prepare cleanup command", width='stretch'):
            show_command_block(
                [
                    "conda",
                    "run",
                    "-n",
                    os.environ.get("CONDA_ENV", "dsp"),
                    "--no-capture-output",
                    "python",
                    "scripts/data/cleanup_runs.py",
                    "--days",
                    "30",
                ],
                cwd=PROJECT_ROOT,
                description="Remove run directories older than 30 days.",
            )
            st.success("Command prepared. Execute it manually to prune old runs.")

st.divider()
st.caption("Tip: Sync the best model artifacts to cloud storage immediately after evaluation.")
