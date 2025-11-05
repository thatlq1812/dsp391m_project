"""
Page 6: Training Control
Launch STMGT training runs, monitor progress, and compare results.
"""

from __future__ import annotations

import copy
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.utils.command_blocks import show_command_block
from pydantic import ValidationError

from traffic_forecast.core.artifacts import save_run_config
from traffic_forecast.core.config_loader import ModelConfig, RunConfig, TrainingConfig, get_data_root
from traffic_forecast.core.registry import load_model_registry
from traffic_forecast.utils.conda import resolve_conda_executable

st.set_page_config(page_title="Training Control", page_icon="🎮", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CONFIG_DIR = PROJECT_ROOT / "configs"
ENV_NAME = os.environ.get("CONDA_ENV", "dsp")
MODEL_REGISTRY_PATH = PROJECT_ROOT / "configs" / "model_registry.json"
PROCESSED_DIR = get_data_root()

DATASET_OPTIONS = [
    {
        "label": "Extreme Augmented (recommended)",
        "value": "all_runs_extreme_augmented.parquet",
        "note": "Max coverage, 48.4x multiplier",
    },
    {
        "label": "Basic Augmented",
        "value": "all_runs_augmented.parquet",
        "note": "Balanced noise & interpolation",
    },
    {
        "label": "Original Combined",
        "value": "all_runs_combined.parquet",
        "note": "Raw merged collections",
    },
]

st.title("🎮 Training Control")
st.markdown("Configure, launch, and monitor STMGT training runs with interactive hyperparameter tuning.")

# Quick stats at top
def _is_run_directory(path: Path) -> bool:
    if not path.is_dir():
        return False
    for candidate in ("training_history.csv", "config.json", "test_results.json"):
        if (path / candidate).exists():
            return True
    return False

if OUTPUTS_DIR.exists():
    run_dirs = sorted(
        [d for d in OUTPUTS_DIR.iterdir() if _is_run_directory(d)],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Runs", len(run_dirs))
    
    best_mae = float('inf')
    best_r2 = -float('inf')
    for run_dir in run_dirs:
        results = run_dir / "test_results.json"
        if results.exists():
            try:
                data = json.loads(results.read_text())
                if data.get("mae"): best_mae = min(best_mae, data["mae"])
                if data.get("r2"): best_r2 = max(best_r2, data["r2"])
            except: pass
    
    col2.metric("Best MAE", f"{best_mae:.3f} km/h" if best_mae != float('inf') else "N/A")
    col3.metric("Best R²", f"{best_r2:.3f}" if best_r2 != -float('inf') else "N/A")
    st.divider()
else:
    st.info("No training runs yet. Configure and launch your first training below!")
    st.divider()


def _build_run_config(payload: Dict[str, Any]) -> RunConfig:
    model_cfg = ModelConfig(**payload.get("model", {}))
    training_cfg = TrainingConfig(**payload.get("training", {}))
    metadata = payload.get("metadata", {}).copy()
    return RunConfig(model=model_cfg, training=training_cfg, metadata=metadata)


def _build_training_command(script_path: str, config_path: Path) -> List[str]:
    return [
        resolve_conda_executable(),
        "run",
        "-n",
        ENV_NAME,
        "--no-capture-output",
        "python",
        script_path,
        "--config",
        str(config_path.relative_to(PROJECT_ROOT)),
    ]


def _load_model_registry() -> List[Dict[str, Any]]:
    if not MODEL_REGISTRY_PATH.exists():
        st.error(
            f"Model registry not found at `{MODEL_REGISTRY_PATH.relative_to(PROJECT_ROOT)}`."
        )
        return []

    try:
        registry = load_model_registry(MODEL_REGISTRY_PATH)
    except (json.JSONDecodeError, ValidationError) as exc:
        st.error(f"Failed to load model registry: {exc}")
        return []

    return [entry.model_dump() for entry in registry.models]


def _find_default_index(options: List[Any], default: Any) -> int:
    for idx, option in enumerate(options):
        if isinstance(option, float) and isinstance(default, float):
            if abs(option - default) < 1e-9:
                return idx
        if option == default:
            return idx
    return 0


def _set_nested(config: Dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor: Dict[str, Any] = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def _dataset_label(option: Dict[str, str]) -> str:
    dataset_path = PROCESSED_DIR / option["value"]
    status = "available" if dataset_path.exists() else "missing"
    return f"{option['label']} ({option['value']}) [{status}]"


def _render_hyperparameter(model_key: str, param: Dict[str, Any]) -> Any:
    param_type = param.get("type", "text")
    label = param.get("label", param["key"])
    widget_key = f"{model_key}_{param['key']}"
    default = param.get("default")

    if param_type == "int_slider":
        return st.slider(
            label,
            min_value=int(param.get("min", 0)),
            max_value=int(param.get("max", 100)),
            value=int(default if default is not None else param.get("min", 0)),
            step=int(param.get("step", 1)),
            key=widget_key,
        )

    if param_type == "float_slider":
        return st.slider(
            label,
            min_value=float(param.get("min", 0.0)),
            max_value=float(param.get("max", 1.0)),
            value=float(default if default is not None else param.get("min", 0.0)),
            step=float(param.get("step", 0.1)),
            key=widget_key,
        )

    if param_type == "select":
        options = param.get("options", [])
        if not options:
            st.warning(f"Parameter `{param['key']}` has no options defined.")
            return default
        index = _find_default_index(options, default)
        return st.selectbox(
            label,
            options,
            index=index,
            format_func=lambda opt: str(opt),
            key=widget_key,
        )

    if param_type == "dataset_select":
        option_labels = [_dataset_label(option) for option in DATASET_OPTIONS]
        values = [option["value"] for option in DATASET_OPTIONS]
        target = default if default in values else values[0]
        index = _find_default_index(values, target)
        selection = st.selectbox(label, option_labels, index=index, key=widget_key)
        label_to_value = {label_text: value for label_text, value in zip(option_labels, values)}
        return label_to_value[selection]

    if param_type == "bool":
        return st.checkbox(label, value=bool(default), key=widget_key)

    if param_type == "text":
        return st.text_input(label, value=str(default or ""), key=widget_key)

    st.warning(f"Unsupported parameter type `{param_type}` for `{param['key']}`.")
    return default


if "latest_training_command" not in st.session_state:
    st.session_state.latest_training_command = None
if "latest_training_config" not in st.session_state:
    st.session_state.latest_training_config = None

TAB_CONTROL, TAB_TUNING, TAB_MONITOR, TAB_COMPARE, TAB_EXPORT = st.tabs(
    [
        "Training Control",
        "Hyperparameter Tuning",
        "Monitor Progress",
        "Model Comparison",
        "Export Reports",
    ]
)

with TAB_CONTROL:
    st.markdown("### Training Configuration")

    registry_models = _load_model_registry()

    if not registry_models:
        st.info(
            "Add entries to `configs/model_registry.json` to enable registry-driven training guidance."
        )
    else:
        display_map: Dict[str, Dict[str, Any]] = {
            model["display_name"]: model for model in registry_models
        }
        display_names = list(display_map.keys())
        selected_display = st.selectbox("Model", display_names, index=0)
        selected_model = display_map[selected_display]

        if selected_model.get("description"):
            st.caption(selected_model["description"])

        hyperparameters = selected_model.get("hyperparameters", [])
        selected_values: Dict[str, Any] = {}

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for param in hyperparameters:
            grouped.setdefault(param.get("group", "Parameters"), []).append(param)

        for group_name, params in grouped.items():
            st.markdown(f"#### {group_name}")
            if not params:
                continue
            columns = st.columns(min(2, len(params)))
            for idx, param in enumerate(params):
                target_column = columns[idx % len(columns)]
                with target_column:
                    selected_values[param["key"]] = _render_hyperparameter(
                        selected_model["key"], param
                    )

        config_payload: Dict[str, Any] = copy.deepcopy(
            selected_model.get("train", {}).get("config", {})
        )
        for dotted_key, value in selected_values.items():
            _set_nested(config_payload, dotted_key, value)

        run_config = _build_run_config(config_payload)
        run_config.metadata.setdefault("created_by", "dashboard")
        run_config.metadata["model_key"] = selected_model["key"]

        st.markdown("#### Generated Configuration")
        config_preview = json.dumps(run_config.to_dict(), indent=2)
        st.code(config_preview, language="json")

        train_script = selected_model.get("train", {}).get("script")
        if train_script:
            st.markdown("#### Command Template")
            st.code(f"python {train_script} --config <config_path.json>", language="bash")
        else:
            st.error("Selected model does not define a training script path.")

        st.divider()

        col_prepare, col_stop, col_save = st.columns(3)
        with col_prepare:
            disabled = not train_script
            if st.button(
                "Prepare Training Command",
                type="primary",
                width='stretch',
                disabled=disabled,
                key=f"prepare_{selected_model['key']}",
            ):
                CONFIG_DIR.mkdir(parents=True, exist_ok=True)
                config_filename = (
                    f"{selected_model['key']}_train_{datetime.now():%Y%m%d_%H%M%S}.json"
                )
                config_path = CONFIG_DIR / config_filename
                save_run_config(run_config, config_path)
                st.session_state.latest_training_config = config_path
                command = _build_training_command(train_script, config_path)
                st.session_state.latest_training_command = command
                show_command_block(
                    command,
                    cwd=PROJECT_ROOT,
                    description="Run the command below to launch training.",
                    success_hint="Keep the terminal open; logs stream continuously.",
                )
                st.success(
                    f"Config saved to `{config_path.relative_to(PROJECT_ROOT)}`. Command ready to copy."
                )

        with col_stop:
            if st.button(
                "How do I stop training?",
                width='stretch',
                key=f"stop_{selected_model['key']}",
            ):
                st.info(
                    "Focus the terminal running the command and press Ctrl+C once. Allow PyTorch to shut down cleanly."
                )

        with col_save:
            if st.button(
                "Save Preset",
                width='stretch',
                key=f"save_{selected_model['key']}",
            ):
                CONFIG_DIR.mkdir(parents=True, exist_ok=True)
                preset_path = CONFIG_DIR / f"{selected_model['key']}_custom_config.json"
                save_run_config(run_config, preset_path)
                st.success(f"Preset saved to `{preset_path.relative_to(PROJECT_ROOT)}`")

        st.divider()
        if st.session_state.latest_training_command:
            st.success("Latest training command prepared. Paste it into a terminal to start training.")
        else:
            st.info("Prepare a training command to begin.")

with TAB_TUNING:
    st.markdown("### Hyperparameter Tuning (Roadmap)")
    st.warning("Automated tuning is not yet wired up. Use the controls above to generate per-run configs.")
    st.markdown(
        "- Record configs and metrics in `outputs/<run>/config.json` and `training_history.csv`\n"
        "- Use external tools (e.g., MLflow, Optuna) if large sweeps are required"
    )

with TAB_MONITOR:
    st.markdown("### Monitor Training Progress")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        auto_refresh = st.checkbox("Auto-refresh every 10s", value=False)
    with col2:
        if st.button("Refresh Now", use_container_width=True):
            st.rerun()
    
    if auto_refresh:
        time.sleep(10)
        st.rerun()

    if not OUTPUTS_DIR.exists():
        st.info("No outputs directory yet. Launch a training run to populate it.")
    else:
        run_dirs = sorted(
            [d for d in OUTPUTS_DIR.iterdir() if _is_run_directory(d)],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not run_dirs:
            st.info("No training runs with `training_history.csv` detected yet.")
        else:
            selected_run = st.selectbox("Training run", [d.name for d in run_dirs])
            run_dir = OUTPUTS_DIR / selected_run
            history_file = run_dir / "training_history.csv"
            if not history_file.exists():
                st.warning(
                    f"`training_history.csv` missing in `{selected_run}`. Training may still be in progress or logging disabled."
                )
            else:
                df_history = pd.read_csv(history_file)
                if df_history.empty:
                    st.warning("History file is empty. Wait for the next logging interval.")
                else:
                    latest = df_history.iloc[-1]
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Epoch", f"{int(latest.get('epoch', 0))}/{run_config.get('training', {}).get('max_epochs', '?') if (config_path := run_dir / 'config.json').exists() and (run_config := json.loads(config_path.read_text())) else '?'}")
                    if "val_mae" in df_history.columns and df_history["val_mae"].notna().any():
                        best_mae = df_history['val_mae'].min()
                        current_mae = latest.get('val_mae', float('nan'))
                        delta_mae = current_mae - best_mae
                        col2.metric("Val MAE", f"{current_mae:.3f} km/h", delta=f"{delta_mae:+.3f}", delta_color="inverse")
                    elif "train_mae" in df_history.columns:
                        col2.metric("Train MAE", f"{latest.get('train_mae', 0):.3f} km/h")
                    else:
                        col2.metric("MAE", "N/A")
                    if "val_r2" in df_history.columns and df_history["val_r2"].notna().any():
                        col3.metric("Best Val R²", f"{df_history['val_r2'].max():.3f}")
                    elif "train_r2" in df_history.columns:
                        col3.metric("Best Train R²", f"{df_history['train_r2'].max():.3f}")
                    else:
                        col3.metric("R²", "N/A")
                    col4.metric(
                        "Current Loss",
                        f"{latest.get('val_loss', latest.get('train_loss', float('nan'))):.4f}",
                    )

                    st.divider()
                    col_loss, col_mae = st.columns(2)
                    with col_loss:
                        fig_loss = go.Figure()
                        fig_loss.add_scatter(
                            x=df_history["epoch"],
                            y=df_history["train_loss"],
                            mode="lines",
                            name="Train Loss",
                        )
                        if "val_loss" in df_history.columns:
                            fig_loss.add_scatter(
                                x=df_history["epoch"],
                                y=df_history["val_loss"],
                                mode="lines",
                                name="Val Loss",
                            )
                        fig_loss.update_layout(xaxis_title="Epoch", yaxis_title="Loss", height=380)
                        st.plotly_chart(fig_loss, width='stretch')

                    with col_mae:
                        if "train_mae" in df_history.columns:
                            fig_mae = go.Figure()
                            fig_mae.add_scatter(
                                x=df_history["epoch"],
                                y=df_history["train_mae"],
                                mode="lines",
                                name="Train MAE",
                            )
                            if "val_mae" in df_history.columns:
                                fig_mae.add_scatter(
                                    x=df_history["epoch"],
                                    y=df_history["val_mae"],
                                    mode="lines",
                                    name="Val MAE",
                                )
                            fig_mae.update_layout(xaxis_title="Epoch", yaxis_title="MAE (km/h)", height=380)
                            st.plotly_chart(fig_mae, width='stretch')
                        else:
                            st.info("MAE metrics unavailable in this history file.")

                    test_results = run_dir / "test_results.json"
                    if test_results.exists():
                        st.markdown("#### Test Metrics")
                        st.json(json.loads(test_results.read_text()))
                    else:
                        st.caption("Test metrics not generated yet. Run `scripts/training/evaluate.py` after training.")

with TAB_COMPARE:
    st.markdown("### 🏆 Model Comparison")
    model_dirs = (
        [d for d in OUTPUTS_DIR.iterdir() if _is_run_directory(d)] if OUTPUTS_DIR.exists() else []
    )
    if not model_dirs:
        st.info("Train at least one model to populate comparison data.")
    else:
        rows = []
        for model_dir in model_dirs:
            results_path = model_dir / "test_results.json"
            config_path = model_dir / "config.json"
            history_path = model_dir / "training_history.csv"
            
            mae = r2 = epochs = None
            if results_path.exists():
                try:
                    results = json.loads(results_path.read_text())
                except json.JSONDecodeError:
                    results = {}
                mae = results.get("mae")
                r2 = results.get("r2")
            
            if history_path.exists():
                try:
                    df_h = pd.read_csv(history_path)
                    epochs = len(df_h)
                    if mae is None and "val_mae" in df_h.columns:
                        mae = df_h["val_mae"].min()
                    if r2 is None and "val_r2" in df_h.columns:
                        r2 = df_h["val_r2"].max()
                except: pass

            config_data: Dict[str, Any] = {}
            if config_path.exists():
                try:
                    config_data = json.loads(config_path.read_text())
                except json.JSONDecodeError:
                    config_data = {}

            dataset = config_data.get("training", {}).get("data_source", "-")
            model_details: Dict[str, Any] = config_data.get("model", {})
            detail_parts = []
            for field in ("hidden_dim", "num_heads", "num_blocks", "mixture_components"):
                if isinstance(model_details.get(field), (int, float)):
                    detail_parts.append(f"{field}={model_details[field]}")
            summary = ", ".join(detail_parts) if detail_parts else "-"
            
            lr = config_data.get("training", {}).get("learning_rate", "-")
            batch = config_data.get("training", {}).get("batch_size", "-")
            
            rows.append(
                {
                    "Run": model_dir.name,
                    "Created": datetime.fromtimestamp(model_dir.stat().st_ctime).strftime("%Y-%m-%d %H:%M"),
                    "Epochs": epochs if epochs else "-",
                    "MAE": f"{mae:.3f}" if isinstance(mae, (int, float)) else "-",
                    "R²": f"{r2:.3f}" if isinstance(r2, (int, float)) else "-",
                    "LR": f"{lr:.4f}" if isinstance(lr, (int, float)) else "-",
                    "Batch": batch if batch != "-" else "-",
                    "Dataset": dataset.replace("all_runs_", "").replace(".parquet", ""),
                    "Model": summary,
                }
            )
        
        df_compare = pd.DataFrame(rows)
        # Highlight best
        st.dataframe(
            df_compare.style.apply(
                lambda x: ['background-color: #90EE90' if v == df_compare['MAE'].min() else '' for v in x], 
                subset=['MAE'], 
                axis=0
            ).apply(
                lambda x: ['background-color: #90EE90' if v == df_compare['R²'].max() else '' for v in x], 
                subset=['R²'], 
                axis=0
            ),
            hide_index=True, 
            use_container_width=True
        )

with TAB_EXPORT:
    st.markdown("### Export Reports")
    st.info(
        "Training exports remain manual. Copy artifacts from `outputs/<run>/` into reports or notebooks as needed."
    )
    if st.session_state.latest_training_config:
        st.markdown("#### Latest Config Path")
        st.code(str(st.session_state.latest_training_config.relative_to(PROJECT_ROOT)))

st.divider()
st.caption("Tip: Always verify validation curves plateau before promoting a model to production.")
