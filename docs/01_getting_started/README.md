# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# Getting Started

Quick start guides for using the STMGT Traffic Forecasting System.

---

## 📖 Available Guides

### [CLI Guide](CLI.md)
Command-line interface for training, evaluation, and prediction.

**Use Cases:**
- Training models from scratch
- Running evaluations on test sets
- Making predictions via terminal

**Quick Example:**
```bash
# Train STMGT
traffic-forecast train --config configs/train_normalized_v3.json

# Evaluate model
traffic-forecast evaluate --model outputs/stmgt_v3/model.pt

# Make predictions
traffic-forecast predict --input data/test.parquet
```

### [API Guide](API.md)
REST API endpoints for integration with other systems.

**Use Cases:**
- Production deployments
- Real-time predictions
- Integration with dashboards

**Quick Example:**
```bash
# Start API server
uvicorn traffic_api.main:app --host 0.0.0.0 --port 8000

# Make prediction request
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"edge_id": "edge_123", "timestamp": "2024-01-01T10:00:00"}'
```

### [Deployment Guide](DEPLOYMENT.md)
Production deployment instructions for various platforms.

**Use Cases:**
- Docker containerization
- Cloud deployment (AWS, GCP, Azure)
- Kubernetes orchestration

**Quick Example:**
```bash
# Build Docker image
docker build -t traffic-forecast:latest .

# Run container
docker run -p 8000:8000 traffic-forecast:latest
```

---

## 🚦 Quick Start Workflow

### 1. Installation

```bash
# Clone repository
git clone https://github.com/thatlq1812/dsp391m_project.git
cd dsp391m_project

# Create environment
conda env create -f environment.yml
conda activate dsp

# Install package
pip install -e .
```

### 2. Data Preparation

```bash
# Download or generate dataset
python scripts/data/generate_super_dataset.py \
  --config configs/super_dataset_config.yaml \
  --output data/processed/dataset.parquet
```

### 3. Train Model

```bash
# Train STMGT
python scripts/training/train_stmgt.py \
  --config configs/train_normalized_v3.json \
  --output-dir outputs/my_model
```

### 4. Evaluate

```bash
# Evaluate on test set
python scripts/evaluation/evaluate_model.py \
  --model outputs/my_model/model.pt \
  --data data/processed/dataset.parquet
```

### 5. Deploy (Optional)

```bash
# Start API server
python -m traffic_api.main
```

---

## 📚 Next Steps

**For New Users:**
- Read [Data Overview](../02_data/DATA.md) to understand dataset structure
- Review [Model Overview](../03_models/MODEL.md) to understand architectures
- Check [Training Workflow](../03_models/TRAINING_WORKFLOW.md) for best practices

**For Developers:**
- Explore [STMGT Architecture](../03_models/architecture/STMGT_ARCHITECTURE.md)
- Review [API Guide](API.md) for integration details
- Check [Deployment Guide](DEPLOYMENT.md) for production setup

**For Researchers:**
- Read [Final Report](../05_final_report/final_report.pdf)
- Review [Super Dataset Design](../02_data/super_dataset/SUPER_DATASET_DESIGN.md)
- Analyze [Metrics Verification](../04_evaluation/METRICS_VERIFICATION_ALL_MODELS.md)

---

## 🆘 Troubleshooting

**Common Issues:**

1. **Import errors:** Make sure to `pip install -e .` in project root
2. **CUDA errors:** Check PyTorch installation matches your CUDA version
3. **Memory errors:** Reduce batch size in config files
4. **Data not found:** Verify dataset paths in config files

**Getting Help:**
- Check [CHANGELOG](../CHANGELOG.md) for recent updates
- Review error logs in `outputs/*/logs/`
- Open an issue on GitHub

---

**Last Updated:** November 15, 2025
