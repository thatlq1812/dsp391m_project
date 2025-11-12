# Model Comparison Results

|              |   MAE (km/h) |   RMSE (km/h) |    R² |   MAPE (%) | CRPS   |
|:-------------|-------------:|--------------:|------:|-----------:|:-------|
| STMGT V3     |         3.08 |          4.5  | 0.817 |      19.68 | 2.23   |
| GCN          |         3.91 |          5.2  | 0.72  |      25    | -      |
| LSTM         |         4.35 |          5.86 | 0.302 |      26.23 | -      |
| GraphWaveNet |        11.04 |         12.5  | 0.4   |      35    | -      |

**Best Model**: STMGT V3
- Lowest MAE, RMSE, and MAPE
- Highest R² score
- Provides uncertainty quantification (CRPS)
