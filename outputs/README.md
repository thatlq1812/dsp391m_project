# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# Training Outputs

This directory contains training run outputs and saved models.

## Latest Run

Check the most recent `stmgt_v2_*` directory for:

- `best_model.pt` - Best model checkpoint
- `config.json` - Training configuration
- `history.json` - Training history
- `training_history.csv` - Epoch metrics used by the dashboard registry
- `test_results.json` (optional) - Final evaluation metrics

## Usage

```python
import torch
from traffic_forecast.models.stmgt import STMGT

model = STMGT(...)
model.load_state_dict(torch.load('outputs/stmgt_v2_*/best_model.pt'))
```

All Phase 1 training jobs are automatically validated:

- Training metadata is synchronized with the Streamlit registry (`dashboard/pages/10_Model_Registry.py`).
- Dataset provenance (parquet path, validation summary) is embedded into each `config.json`.
- Use `traffic_forecast.core.config_loader.resolve_output_path` when scripting custom evaluations.

Old training runs are archived in `models/training_runs/`.
