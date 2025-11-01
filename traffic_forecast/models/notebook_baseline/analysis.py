"""Pipeline that mirrors the original notebook analysis without modifications."""

from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import folium
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


@dataclass
class NotebookBaselineConfig:
    data_path: Path = Path("data/processed/all_runs_combined.parquet")
    output_root: Path = Path("outputs/astgcn")
    congestion_threshold: float = 20.0


class NotebookBaselineRunner:
    """Replays the exact notebook workflow against the project dataset."""

    def __init__(self, config: NotebookBaselineConfig | None = None) -> None:
        self.config = config or NotebookBaselineConfig()
        self.config.output_root.mkdir(parents=True, exist_ok=True)

    def run(self) -> Dict[str, Path]:
        """Execute every notebook cell sequentially and save artifacts."""

        df_raw = self._load_data()
        df = self._align_columns(df_raw.copy())

        run_dir = self._prepare_run_directory()
        outputs: Dict[str, Path] = {}

        # --- 1. Đọc dữ liệu ---
        summary_path = self._export_summary(df, run_dir)
        outputs["summary"] = summary_path

        # --- 2. Thống kê mô tả ---
        describe_path = self._export_descriptive_stats(df, run_dir)
        outputs["describe"] = describe_path

        # --- 3. Phân tích phân phối tốc độ và thời tiết ---
        dist_path = self._plot_distributions(df, run_dir)
        outputs["distributions"] = dist_path

        # --- 4. Mối quan hệ giữa tốc độ và thời tiết ---
        scatter_path = self._plot_relationship(df, run_dir)
        outputs["scatter"] = scatter_path

        # --- 5. Ma trận tương quan ---
        heatmap_path = self._plot_heatmap(df, run_dir)
        outputs["heatmap"] = heatmap_path

        # --- 6. Trạng thái trung bình của các node ---
        node_stats, node_stats_path = self._export_node_average(df, run_dir)
        outputs["node_stats"] = node_stats_path

        # --- 7. Bản đồ Folium ---
        map_path = self._build_map(node_stats, run_dir)
        outputs["map"] = map_path

        # --- 8. Xác định tắc nghẽn ---
        congestion_plot, congestion_data = self._analyze_congestion(df, run_dir)
        outputs["congestion_plot"] = congestion_plot
        outputs["congestion_data"] = congestion_data

        return outputs

    def _load_data(self) -> pd.DataFrame:
        if self.config.data_path.suffix.lower() == ".csv":
            return pd.read_csv(self.config.data_path)
        if self.config.data_path.suffix.lower() in {".parquet", ".pq"}:
            return pd.read_parquet(self.config.data_path)
        raise ValueError(f"Unsupported data format: {self.config.data_path}")

    def _align_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        rename_map = {
            "temperature_c": "temperature_c_avg",
            "wind_speed_kmh": "wind_speed_kmh_avg",
            "precipitation_mm": "precipitation_mm_avg",
            "lat_a": "node_a_lat_x",
            "lon_a": "node_a_lon_x",
        }
        df = df.rename(columns=rename_map)
        if "duration_sec" not in df.columns and "duration_min" in df.columns:
            df["duration_sec"] = df["duration_min"] * 60
        if "road_type" not in df.columns:
            df["road_type"] = "Unknown"
        expected = [
            "speed_kmh",
            "distance_km",
            "duration_sec",
            "temperature_c_avg",
            "precipitation_mm_avg",
            "wind_speed_kmh_avg",
        ]
        missing: List[str] = [col for col in expected if col not in df.columns]
        if missing:
            raise KeyError(f"Dataset is missing expected columns: {missing}")
        return df

    def _prepare_run_directory(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.config.output_root / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _export_summary(self, df: pd.DataFrame, run_dir: Path) -> Path:
        summary_path = run_dir / "summary.txt"
        buffer = io.StringIO()
        df.info(buf=buffer)
        contents = ["Dữ liệu tổng hợp:", buffer.getvalue(), df.head(3).to_string()]
        summary_path.write_text("\n\n".join(contents), encoding="utf-8")
        return summary_path

    def _export_descriptive_stats(self, df: pd.DataFrame, run_dir: Path) -> Path:
        describe_path = run_dir / "descriptive_stats.csv"
        cols = [
            "speed_kmh",
            "distance_km",
            "duration_sec",
            "temperature_c_avg",
            "precipitation_mm_avg",
            "wind_speed_kmh_avg",
        ]
        df[cols].describe().to_csv(describe_path)
        return describe_path

    def _plot_distributions(self, df: pd.DataFrame, run_dir: Path) -> Path:
        sns.set(style="whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle("Phân phối tốc độ và điều kiện thời tiết", fontsize=16)

        sns.histplot(df["speed_kmh"], kde=True, ax=axes[0, 0], color="skyblue")
        axes[0, 0].set_title("Phân phối tốc độ trung bình (km/h)")

        sns.boxplot(x=df["speed_kmh"], ax=axes[0, 1], color="skyblue")
        axes[0, 1].set_title("Boxplot tốc độ trung bình")

        sns.histplot(df["temperature_c_avg"], kde=True, ax=axes[1, 0], color="orange")
        axes[1, 0].set_title("Phân phối nhiệt độ trung bình (°C)")

        sns.histplot(df["precipitation_mm_avg"], kde=True, ax=axes[1, 1], color="green")
        axes[1, 1].set_title("Phân phối lượng mưa (mm)")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        output_path = run_dir / "distribution_plots.png"
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return output_path

    def _plot_relationship(self, df: pd.DataFrame, run_dir: Path) -> Path:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            data=df,
            x="temperature_c_avg",
            y="speed_kmh",
            hue="precipitation_mm_avg",
            ax=ax,
        )
        ax.set_title("Mối quan hệ giữa Nhiệt độ, Mưa và Tốc độ", fontsize=15)
        ax.set_xlabel("Nhiệt độ trung bình (°C)")
        ax.set_ylabel("Tốc độ trung bình (km/h)")
        output_path = run_dir / "temperature_precipitation_scatter.png"
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return output_path

    def _plot_heatmap(self, df: pd.DataFrame, run_dir: Path) -> Path:
        corr_cols = [
            "speed_kmh",
            "distance_km",
            "duration_sec",
            "temperature_c_avg",
            "precipitation_mm_avg",
            "wind_speed_kmh_avg",
        ]
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df[corr_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.set_title("Ma trận tương quan giữa các biến chính")
        output_path = run_dir / "correlation_heatmap.png"
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return output_path

    def _export_node_average(self, df: pd.DataFrame, run_dir: Path) -> tuple[pd.DataFrame, Path]:
        df_node_avg = df.groupby("node_a_id").agg(
            avg_speed=("speed_kmh", "mean"),
            avg_temp=("temperature_c_avg", "mean"),
            avg_rain=("precipitation_mm_avg", "mean"),
            lat=("node_a_lat_x", "first"),
            lon=("node_a_lon_x", "first"),
        ).reset_index()

        node_stats_path = run_dir / "node_average.csv"
        df_node_avg.to_csv(node_stats_path, index=False)
        return df_node_avg, node_stats_path

    def _build_map(self, df_node_avg: pd.DataFrame, run_dir: Path) -> Path:
        map_center = [df_node_avg["lat"].mean(), df_node_avg["lon"].mean()]
        traffic_map = folium.Map(location=map_center, zoom_start=13, tiles="cartodbpositron")

        def get_color(speed: float) -> str:
            if speed < 25:
                return "red"
            if speed < 40:
                return "orange"
            return "green"

        for _, row in df_node_avg.iterrows():
            popup_html = (
                f"<b>Node:</b> {row['node_a_id']}<br>"
                f"<b>Tốc độ TB:</b> {row['avg_speed']:.2f} km/h<br>"
                f"<b>Nhiệt độ TB:</b> {row['avg_temp']:.1f}°C<br>"
                f"<b>Lượng mưa TB:</b> {row['avg_rain']:.1f} mm"
            )
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=4,
                color=get_color(row["avg_speed"]),
                fill=True,
                fill_opacity=0.8,
                popup=popup_html,
            ).add_to(traffic_map)

        output_path = run_dir / "traffic_status_map.html"
        traffic_map.save(output_path)
        return output_path

    def _analyze_congestion(self, df: pd.DataFrame, run_dir: Path) -> tuple[Path, Path]:
        congested_edges = df[df["speed_kmh"] < self.config.congestion_threshold]
        congestion_path = run_dir / "congested_edges.csv"
        congested_edges.to_csv(congestion_path, index=False)

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.countplot(
            x="road_type",
            data=congested_edges,
            order=df["road_type"].value_counts().index if "road_type" in df.columns else None,
            ax=ax,
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_title("Phân bố loại đường trong các cạnh tắc nghẽn (<20 km/h)")
        output_plot = run_dir / "congested_road_types.png"
        fig.savefig(output_plot, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return output_plot, congestion_path


def run_astgcn(config: NotebookBaselineConfig | None = None) -> Dict[str, Path]:
    runner = NotebookBaselineRunner(config)
    return runner.run()
