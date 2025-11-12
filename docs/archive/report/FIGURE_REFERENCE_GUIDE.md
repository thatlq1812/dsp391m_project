# Quick Figure Reference Guide

Use this for inserting figures into the LaTeX document.

## LaTeX Figure Code Templates

### Standard Figure

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.8\textwidth]{figures/fig01_speed_distribution.png}
    \caption{Speed distribution across all traffic edges showing multi-modal patterns.}
    \label{fig:speed_distribution}
\end{figure}
```

### Two-Column Figure

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.95\textwidth]{figures/fig15_model_comparison.png}
    \caption{Performance comparison of STMGT V3 against baseline models.}
    \label{fig:model_comparison}
\end{figure}
```

### Full-Width Figure

```latex
\begin{figure*}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figures/fig02_network_topology.png}
    \caption{HCMC road network topology with 62 nodes and 144 edges.}
    \label{fig:network_topology}
\end{figure*}
```

## Figure Placement by Section

### Section 5: Data Collection & Preprocessing

- `fig01_speed_distribution.png` - Speed distribution histogram
- `fig02_network_topology.png` - Road network graph visualization
- `fig03_preprocessing_flow.png` - Data pipeline flowchart
- `fig04_normalization.png` - Before/after normalization

### Section 6: Exploratory Data Analysis

- `fig05_eda_speed_hist.png` - Speed histogram with GMM
- `fig06_hourly_pattern.png` - Hourly speed patterns
- `fig07_weekly_pattern.png` - Day-of-week boxplots
- `fig08_spatial_corr.png` - Spatial correlation heatmap
- `fig09_temp_speed.png` - Temperature vs speed scatter
- `fig10_weather_box.png` - Weather condition boxplots

### Section 7: Model Architecture

- `fig11_architecture.png` - **NEED TO CREATE** - STMGT architecture diagram
- `fig12_attention.png` - **NEED TO CREATE** - Attention mechanism

### Section 10: Evaluation & Section 11: Results

- `fig13_training_curves.png` - Training/validation loss and MAE
- `fig14_ablation_study.png` - Component ablation results
- `fig15_model_comparison.png` - Model performance comparison
- `fig16_good_prediction.png` - Best case prediction example
- `fig17_bad_prediction.png` - Challenging case prediction

## Figure Captions (Copy-Paste Ready)

```latex
% Figure 1
\caption{Speed distribution across all traffic edges in HCMC dataset, showing multi-modal patterns with peaks at approximately 15 km/h (congested) and 35 km/h (free-flow).}

% Figure 2
\caption{Road network topology extracted from GPS trajectories. The network consists of 62 major intersection nodes and 144 directed edges, color-coded by node degree.}

% Figure 3
\caption{End-to-end data preprocessing pipeline, from raw GPS trajectories to normalized features ready for model training.}

% Figure 4
\caption{Effect of Z-score normalization on speed distribution. Left: raw data with mean 21.4 km/h. Right: normalized data with mean 0 and standard deviation 1.}

% Figure 5
\caption{Speed histogram fitted with Gaussian Mixture Model (GMM) showing three distinct traffic regimes: congested, transitional, and free-flow.}

% Figure 6
\caption{Average traffic speed by hour of day, revealing morning (7-9 AM) and evening (5-7 PM) rush hour patterns.}

% Figure 7
\caption{Speed distribution by day of week. Weekdays show lower average speeds during commute hours, while weekends exhibit higher speeds.}

% Figure 8
\caption{Spatial correlation heatmap between traffic edges, demonstrating strong correlations among nearby road segments.}

% Figure 9
\caption{Relationship between ambient temperature and traffic speed. Linear regression shows weak negative correlation (R²=0.023), indicating temperature has minimal direct impact.}

% Figure 10
\caption{Traffic speed distribution under different weather conditions. Clear weather shows highest average speeds, while rainy conditions lead to reduced speeds.}

% Figure 13
\caption{STMGT V3 training curves showing convergence of loss and MAE over 100 epochs. Early stopping activated at epoch 47 based on validation performance.}

% Figure 14
\caption{Ablation study results quantifying the contribution of each model component. Weather features provide 11\% MAE improvement, while spatial attention contributes 24\%.}

% Figure 15
\caption{Performance comparison of STMGT V3 against baseline models on test set. STMGT V3 achieves lowest MAE (3.08 km/h) and highest R² (0.817).}

% Figure 16
\caption{Example of accurate speed prediction during normal traffic conditions. STMGT V3 captures the temporal dynamics with MAE of 1.8 km/h.}

% Figure 17
\caption{Challenging prediction case during sudden congestion event. The model struggles to predict abrupt speed drops, achieving MAE of 6.5 km/h in this scenario.}
```

## Markdown Reference (for report)

```markdown
![Speed Distribution](figures/fig01_speed_distribution.png)
_Figure 1: Speed distribution across all traffic edges_

![Network Topology](figures/fig02_network_topology.png)
_Figure 2: HCMC road network topology (62 nodes, 144 edges)_
```

## All Figure Files

```
docs/final_report/figures/
├── fig01_speed_distribution.png      (158 KB)
├── fig02_network_topology.png         (1.3 MB)
├── fig03_preprocessing_flow.png       (338 KB)
├── fig04_normalization.png            (145 KB)
├── fig05_eda_speed_hist.png          (248 KB)
├── fig06_hourly_pattern.png          (228 KB)
├── fig07_weekly_pattern.png          (175 KB)
├── fig08_spatial_corr.png            (1.7 MB)
├── fig09_temp_speed.png              (677 KB)
├── fig10_weather_box.png             (95 KB)
├── fig13_training_curves.png         (224 KB)
├── fig14_ablation_study.png          (169 KB)
├── fig15_model_comparison.png        (165 KB)
├── fig16_good_prediction.png         (275 KB)
└── fig17_bad_prediction.png          (282 KB)
```

**Total**: 15 PNG files, ~7.5 MB, 300 DPI
