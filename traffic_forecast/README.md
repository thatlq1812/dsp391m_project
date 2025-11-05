# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# Traffic Forecast Library

Core implementation of the STMGT traffic forecasting model.

## Structure

- **models/** – STMGT architecture components and inference helpers
- **data/** – Dataset builders, validators, and dataloader factories
- **augmentation/** – Augmentation strategy primitives (legacy baseline)
- **utils/** – Shared utilities (logging, time features, math helpers)
- **core/** – Config loader, registry schema, artifact manager

## Usage

```python
from traffic_forecast.models.stmgt import STMGT
from traffic_forecast.data.stmgt_dataset import create_stmgt_dataloaders

# Create model
model = STMGT(num_nodes=62, mixture_components=3, ...)

# Load data
train_loader, val_loader, test_loader, num_nodes, edge_index = create_stmgt_dataloaders(...)
```

See `docs/STMGT_ARCHITECTURE.md` and `docs/WORKFLOW.md` for deeper guidance.
