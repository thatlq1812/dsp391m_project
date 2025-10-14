"""
Preprocess feature datasets to prepare model-ready train/validation splits.

The pipeline loads the feature snapshot (default: ``data/features_nodes_v2.json``),
cleans and scales the feature columns, produces train/validation parquet files,
and persists the fitted scaler together with metadata used during inference.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
try:  # Optional plotting support
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover - matplotlib optional
    plt = None
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from traffic_forecast import PROJECT_ROOT

CONFIG_PATH = PROJECT_ROOT / "configs" / "project_config.yaml"


@dataclass
class PreprocessArtifacts:
    train_path: Path
    val_path: Path
    metadata_path: Path
    scaler_path: Path
    feature_columns: List[str]
    target_column: str
    summary_path: Path
    report_path: Path
    samples_path: Path


def load_config(config_path: Optional[Path] = None) -> dict:
    cfg_path = config_path or CONFIG_PATH
    with cfg_path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _load_features(features_path: Path) -> pd.DataFrame:
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")

    with features_path.open(encoding="utf-8") as fh:
        data = json.load(fh)

    if not data:
        raise ValueError(f"Features file {features_path} is empty.")

    df = pd.DataFrame(data)

    if "feature_vector" in df.columns:
        vector_df = pd.DataFrame(df.pop("feature_vector").tolist())
        vector_df.columns = [f"feature_{i}" for i in range(vector_df.shape[1])]
        df = pd.concat([df, vector_df], axis=1)

    return df


def _resolve_feature_columns(
    df: pd.DataFrame,
    configured_features: Optional[Sequence[str]],
    target_column: str,
) -> List[str]:
    if configured_features:
        # ensure column exists; create placeholder if missing to keep pipeline consistent
        for col in configured_features:
            if col not in df.columns:
                df[col] = np.nan
        return list(configured_features)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [col for col in numeric_cols if col != target_column]


def _clean_dataframe(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    drop_threshold: float,
) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=[target_column])

    if drop_threshold and feature_columns:
        ratio = df[feature_columns].notna().mean(axis=1)
        df = df.loc[ratio >= drop_threshold]

    return df


def _impute_features(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    imputation_values: Dict[str, float] = {}
    for col in feature_columns:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        median = series.median()
        if pd.isna(median):
            median = 0.0
        df[col] = series.fillna(float(median))
        imputation_values[col] = float(median)
    return df, imputation_values


def _generate_reports(
    df_original: pd.DataFrame,
    df_imputed: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    output_dir: Path,
) -> Tuple[Path, Path, Path]:
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    samples_path = output_dir / "samples_head.csv"
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    def _missing_ratio(frame: pd.DataFrame) -> Dict[str, float]:
        return {
            col: float(frame[col].isna().mean())
            for col in feature_columns + [target_column]
            if col in frame.columns
        }

    summary = {
        "rows_before_clean": int(len(df_original)),
        "rows_after_clean": int(len(df_imputed)),
        "feature_columns": list(feature_columns),
        "target_column": target_column,
        "missing_ratio_before": _missing_ratio(df_original),
        "missing_ratio_after": _missing_ratio(df_imputed),
    }

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    head_df = df_imputed.head(10)
    head_df.to_csv(samples_path, index=False)

    lines = [
        "# Preprocess Report",
        "",
        f"*Rows before cleaning:* {summary['rows_before_clean']}",
        f"*Rows after cleaning:* {summary['rows_after_clean']}",
        "",
        "## Missing Ratio (Before)",
    ]
    for col, ratio in summary["missing_ratio_before"].items():
        lines.append(f"- {col}: {ratio:.2%}")

    lines.append("\n## Missing Ratio (After)")
    for col, ratio in summary["missing_ratio_after"].items():
        lines.append(f"- {col}: {ratio:.2%}")

    lines.append("\n## Sample Rows")
    lines.append("```")
    lines.append(head_df.to_string(index=False))
    lines.append("```")

    report_path.write_text("\n".join(lines), encoding="utf-8")

    if plt is not None and target_column in df_imputed.columns:
        try:
            fig, ax = plt.subplots(figsize=(6, 4))
            df_imputed[target_column].dropna().hist(ax=ax, bins=30)
            ax.set_title("Target Distribution")
            ax.set_xlabel(target_column)
            ax.set_ylabel("Frequency")
            fig.tight_layout()
            fig.savefig(plots_dir / "target_distribution.png")
            plt.close(fig)
        except Exception:  # pragma: no cover - plotting best effort
            pass

    return summary_path, report_path, samples_path


def _split_datasets(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    keep_columns: Sequence[str],
    test_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features = df[feature_columns].to_numpy(dtype=float)
    target = df[target_column].to_numpy(dtype=float)

    if len(df) < 2:
        raise ValueError("Not enough samples to perform train/validation split.")

    idx_train, idx_val = train_test_split(
        np.arange(len(df)),
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    train_df = df.iloc[idx_train].copy()
    val_df = df.iloc[idx_val].copy()

    X_train = features[idx_train]
    X_val = features[idx_val]
    y_train = target[idx_train]
    y_val = target[idx_val]

    keep_data = {col: (train_df[col].values, val_df[col].values) for col in keep_columns if col in df.columns}

    train_meta = {col: values[0] for col, values in keep_data.items()}
    val_meta = {col: values[1] for col, values in keep_data.items()}

    return train_df, val_df, X_train, X_val, y_train, y_val, train_meta, val_meta


def _build_processed_frames(
    X_scaled: np.ndarray,
    y: np.ndarray,
    feature_columns: Sequence[str],
    target_column: str,
    meta_columns: Dict[str, np.ndarray],
) -> pd.DataFrame:
    df = pd.DataFrame(X_scaled, columns=feature_columns)
    df[target_column] = y
    for col, values in meta_columns.items():
        df[col] = values
    return df


def run_pipeline(config: Optional[dict] = None) -> PreprocessArtifacts:
    """Execute the preprocessing pipeline and return artifact locations."""
    full_config = config or load_config()
    preprocess_cfg = full_config.get("pipelines", {}).get("preprocess", {})

    features_path = PROJECT_ROOT / preprocess_cfg.get("features_file", "data/features_nodes_v2.json")
    output_dir = PROJECT_ROOT / preprocess_cfg.get("output_dir", "data/processed")
    scaler_path = PROJECT_ROOT / preprocess_cfg.get("scaler_path", "models/feature_scaler.pkl")
    metadata_path = PROJECT_ROOT / preprocess_cfg.get("metadata_path", "data/processed/metadata.json")

    target_column = preprocess_cfg.get("target_column", "avg_speed_kmh")
    configured_features = preprocess_cfg.get("feature_columns")
    keep_columns = preprocess_cfg.get("keep_columns", ["node_id", "ts"])
    drop_threshold = preprocess_cfg.get("drop_na_threshold", 0.6)
    test_size = preprocess_cfg.get("test_size", 0.2)
    random_state = preprocess_cfg.get("random_state", 42)

    df = _load_features(features_path)
    feature_columns = _resolve_feature_columns(df, configured_features, target_column)
    df = _clean_dataframe(df, feature_columns, target_column, drop_threshold)

    if df.empty:
        raise ValueError("No usable samples after cleaning; check feature inputs.")

    df_imputed, imputation_values = _impute_features(df, feature_columns)

    summary_path, report_path, samples_path = _generate_reports(
        df_original=df,
        df_imputed=df_imputed,
        feature_columns=feature_columns,
        target_column=target_column,
        output_dir=output_dir,
    )

    (
        _,
        _,
        X_train,
        X_val,
        y_train,
        y_val,
        train_meta,
        val_meta,
    ) = _split_datasets(
        df_imputed,
        feature_columns,
        target_column,
        keep_columns,
        test_size,
        random_state,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    train_processed = _build_processed_frames(X_train_scaled, y_train, feature_columns, target_column, train_meta)
    val_processed = _build_processed_frames(X_val_scaled, y_val, feature_columns, target_column, val_meta)

    output_dir.mkdir(parents=True, exist_ok=True)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"

    train_processed.to_parquet(train_path, index=False)
    val_processed.to_parquet(val_path, index=False)

    joblib.dump(scaler, scaler_path)

    metadata = {
        "created_at": datetime.utcnow().isoformat(),
        "feature_columns": list(feature_columns),
        "target_column": target_column,
        "keep_columns": [col for col in keep_columns if col in train_processed.columns],
        "imputation_values": imputation_values,
        "scaler_path": str(scaler_path.relative_to(PROJECT_ROOT)),
        "train_rows": int(len(train_processed)),
        "val_rows": int(len(val_processed)),
        "source_features_file": str(features_path.relative_to(PROJECT_ROOT)),
        "summary_file": str(summary_path.relative_to(PROJECT_ROOT)),
        "report_file": str(report_path.relative_to(PROJECT_ROOT)),
        "samples_file": str(samples_path.relative_to(PROJECT_ROOT)),
    }

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    print(f"[preprocess] Train samples: {len(train_processed)}, Validation samples: {len(val_processed)}")
    print(f"[preprocess] Artifacts written to {output_dir}")
    print(f"[preprocess] Scaler stored at {scaler_path}")

    return PreprocessArtifacts(
        train_path=train_path,
        val_path=val_path,
        metadata_path=metadata_path,
        scaler_path=scaler_path,
        feature_columns=list(feature_columns),
        target_column=target_column,
        summary_path=summary_path,
        report_path=report_path,
        samples_path=samples_path,
    )


def main(argv: Optional[Sequence[str]] = None) -> PreprocessArtifacts:
    parser = argparse.ArgumentParser(description="Preprocess feature snapshots for model training.")
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional override path to project_config.yaml",
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.config) if args.config else None
    return run_pipeline(cfg)


if __name__ == "__main__":
    main()
