# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# STMGT Data Input & Output Reference

Comprehensive description of the data structures consumed and produced by the STMGT (Spatio-Temporal Multi-Graph Transformer) workflow.

## Notation

- **B** – batch size
- **N** – number of graph nodes (unique traffic edges)
- **T** – history sequence length (input timesteps)
- **P** – prediction horizon (output timesteps)
- **E** – number of directed edges in the graph

Default configuration: `T = 12`, `P = 12`, `N ≈ 62`.

## Raw Dataset Schema

Parquet files under `data/processed/` must include the following columns:

| Column                   | Type       | Description                                                      |
| ------------------------ | ---------- | ---------------------------------------------------------------- |
| `run_id`                 | string/int | Unique ID per collection run (one timestep).                     |
| `timestamp`              | datetime   | Collection time (UTC+7).                                         |
| `node_a_id`, `node_b_id` | string     | Directed edge endpoints used to build the graph.                 |
| `speed_kmh`              | float      | Average traffic speed (km/h) for the edge.                       |
| `congestion_level`       | float      | Optional congestion indicator.                                   |
| `temperature_c`          | float      | Ambient temperature (°C).                                        |
| `wind_speed_kmh`         | float      | Wind speed (km/h).                                               |
| `precipitation_mm`       | float      | Rainfall (mm).                                                   |
| other features           | optional   | Additional engineered signals; ignored unless mapped in configs. |

Temporal fields (`hour`, `dow`, `is_weekend`) are derived during dataset preparation.

## Dataset (\_\_getitem\_\_) Output

`traffic_forecast.data.stmgt_dataset.STMGTDataset` returns a dictionary per sample:

| Key          | Shape       | Description                                         |
| ------------ | ----------- | --------------------------------------------------- |
| `x_traffic`  | `[N, T, 1]` | Speed history per node and timestep.                |
| `x_weather`  | `[T, 3]`    | Weather history (temperature, wind, precipitation). |
| `hour`       | `[T]`       | Hour-of-day (0–23).                                 |
| `dow`        | `[T]`       | Day-of-week (0–6, Monday=0).                        |
| `is_weekend` | `[T]`       | Weekend mask (0/1).                                 |
| `y_target`   | `[N, P]`    | Future speeds to predict.                           |

An accompanying `edge_index` tensor `[2, E]` is stored on the dataset and passed via the collate function.

## Collated Batch Structure

`collate_fn_stmgt` stacks samples into batch tensors consumed by the model:

| Key                            | Shape          | Notes                                            |
| ------------------------------ | -------------- | ------------------------------------------------ |
| `x_traffic`                    | `[B, N, T, 1]` | History per batch item.                          |
| `x_weather`                    | `[B, T, 3]`    | Weather features broadcast across nodes.         |
| `edge_index`                   | `[2, E]`       | Shared graph connectivity (unchanged per batch). |
| `temporal_features.hour`       | `[B, T]`       | Integers 0–23.                                   |
| `temporal_features.dow`        | `[B, T]`       | Integers 0–6.                                    |
| `temporal_features.is_weekend` | `[B, T]`       | Binary mask.                                     |
| `y_target`                     | `[B, N, P]`    | Ground-truth future speeds.                      |

## Model Input Contract

`traffic_forecast.models.stmgt.STMGT.forward` expects:

```python
pred_params = model(
    x_traffic=batch['x_traffic'],          # [B, N, T, 1]
    edge_index=batch['edge_index'],        # [2, E]
    x_weather=batch['x_weather'],          # [B, T, 3]
    temporal_features=batch['temporal_features']
)
```

`temporal_features` is a dictionary with the three tensors listed above.

## Model Output

STMGT produces Gaussian mixture parameters for each node and prediction step:

| Key      | Shape          | Description                                                 |
| -------- | -------------- | ----------------------------------------------------------- |
| `means`  | `[B, N, P, K]` | Component means (default `K = 3`).                          |
| `stds`   | `[B, N, P, K]` | Component standard deviations (bounded between 0.1 and 10). |
| `logits` | `[B, N, P, K]` | Mixture logits (softmax → weights).                         |

The downstream toolkit derives point forecasts by expectation or component selection and can compute uncertainty bands using the mixture distribution.

## Loss and Metrics

- **Training loss:** `mixture_nll_loss` (negative log likelihood with diversity and entropy regularizers).
- **Dashboard metrics:** MAE, MAPE, R², coverage statistics computed from `training_history.csv` within each run directory.

## Data Flow Summary

1. **Collection:** raw runs stored under `data/runs/` with metadata (per scheduler config).
2. **Processing:** merged parquet files exported to `data/processed/`.
3. **Dataset:** sliding windows in `STMGTDataset` convert runs to tensors; graph topology inferred from edge pairs.
4. **Model:** `STMGT` consumes batched tensors via PyTorch Lightning-style training loop (custom script).
5. **Outputs:** results saved in `outputs/stmgt_*`, including predictions, checkpoints, and reports for dashboard consumption.

## Extending the Schema

- To add features, append columns to processed parquet files and update:
  - `traffic_forecast.data.stmgt_dataset.STMGTDataset` for ingestion.
  - `traffic_forecast.models.stmgt.STMGT` encoders (e.g., increase `in_dim` for traffic or weather inputs).
- Ensure new tensors maintain `[B, N, T, feature_dim]` or `[B, T, feature_dim]` formats and update the collate function accordingly.

For real-time inference, replicate the same tensor shapes with `B = 1` and reuse trained checkpoints via `traffic_forecast.models.stmgt.STMGT.load_state_dict`.
