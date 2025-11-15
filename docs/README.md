# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# Documentation Index

Comprehensive documentation for STMGT Traffic Forecasting System.

**Project:** Multi-Modal Traffic Speed Forecasting System  
**Tech Stack:** PyTorch, PyTorch Geometric, FastAPI, TensorFlow/Keras  
**Current Version:** STMGT V3 + Super Dataset 1-Year

---

## 📚 Documentation Structure

### [01. Getting Started](01_getting_started/)
Quick start guides for using the system.

- **[CLI Guide](01_getting_started/CLI.md)** - Command-line interface usage
- **[API Guide](01_getting_started/API.md)** - REST API endpoints and examples
- **[Deployment](01_getting_started/DEPLOYMENT.md)** - Production deployment guide

### [02. Data](02_data/)
Dataset documentation and preprocessing.

- **[Data Overview](02_data/DATA.md)** - Dataset structure and specifications
- **[Augmentation](02_data/AUGMENTATION.md)** - Data augmentation strategies
- **[Super Dataset](02_data/super_dataset/)** - 1-year simulation dataset
  - [Design Document](02_data/super_dataset/SUPER_DATASET_DESIGN.md)
  - [Quick Start](02_data/super_dataset/SUPER_DATASET_QUICKSTART.md)
  - [Generation Report](02_data/super_dataset/SUPER_DATASET_GENERATION_COMPLETE.md)

### [03. Models](03_models/)
Model architecture and training documentation.

- **[Model Overview](03_models/MODEL.md)** - STMGT, GraphWaveNet, LSTM baselines
- **[Training Workflow](03_models/TRAINING_WORKFLOW.md)** - Training pipeline and best practices
- **[Architecture Details](03_models/architecture/)**
  - [STMGT Architecture](03_models/architecture/STMGT_ARCHITECTURE.md)
  - [Data I/O Flow](03_models/architecture/STMGT_DATA_IO.md)
  - [ELI5 Explanation](03_models/architecture/ARCHITECTURE_ELI5.md)

### [04. Evaluation](04_evaluation/)
Model evaluation, verification, and comparison reports.

- **[Metrics Verification](04_evaluation/METRICS_VERIFICATION_ALL_MODELS.md)** - Cross-model comparison
- **[GraphWaveNet Investigation](04_evaluation/graphwavenet/)** - Autocorrelation exploitation analysis
  - [Final Investigation](04_evaluation/graphwavenet/GRAPHWAVENET_INVESTIGATION_FINAL.md)
  - [Verification Report](04_evaluation/graphwavenet/GRAPHWAVENET_VERIFICATION_REPORT.md)
  - [Audit Summary](04_evaluation/graphwavenet/GRAPHWAVENET_AUDIT_SUMMARY.md)
  - [Fix Summary](04_evaluation/graphwavenet/GRAPHWAVENET_FIX_SUMMARY.md)
- **[ASTGCN Verification](04_evaluation/astgcn/)**
  - [Verification Report](04_evaluation/astgcn/ASTGCN_VERIFICATION_REPORT.md)

### [05. Final Report](05_final_report/)
Official project report in IEEE conference format.

- **[Final Report PDF](05_final_report/final_report.pdf)** - Complete project documentation
- **[LaTeX Source](05_final_report/)** - Modular LaTeX sections
- **[Build Guide](05_final_report/BUILD_GUIDE.md)** - Compilation instructions

### [CHANGELOG](CHANGELOG.md)
Complete project changelog with version history and updates.

### [Archive](archive/)
Deprecated and outdated documentation files.

---

## 🚀 Quick Navigation

**For New Users:**
1. Start with [CLI Guide](01_getting_started/CLI.md)
2. Read [Data Overview](02_data/DATA.md)
3. Review [Model Overview](03_models/MODEL.md)

**For Developers:**
1. [Training Workflow](03_models/TRAINING_WORKFLOW.md)
2. [STMGT Architecture](03_models/architecture/STMGT_ARCHITECTURE.md)
3. [API Guide](01_getting_started/API.md)

**For Researchers:**
1. [Final Report](05_final_report/final_report.pdf)
2. [Super Dataset Design](02_data/super_dataset/SUPER_DATASET_DESIGN.md)
3. [Metrics Verification](04_evaluation/METRICS_VERIFICATION_ALL_MODELS.md)

---

## 📊 Project Status

**Latest Achievements:**
- ✅ Super Dataset 1-Year generated (7.5M samples, 247.9 MB)
- ✅ GraphWaveNet autocorrelation issue identified and documented
- ✅ STMGT V3 architecture optimized (680K params, MAE 3.08 km/h on easy dataset)
- 🔄 Training all models on challenging super dataset (in progress)

**Expected Results on Super Dataset:**
- GraphWaveNet: MAE 4-6 km/h (no autocorrelation shortcuts)
- LSTM: MAE 5-7 km/h (temporal only)
- STMGT: MAE 3-4 km/h (maintains spatial-temporal advantage)

---

## 📝 Documentation Standards

All documentation follows these guidelines:

1. **Maintainer Profile** at the top of each file
2. **Clear section hierarchy** with H1 for title, H2 for major sections
3. **American English** spelling throughout
4. **No emojis** in documentation (except this README for navigation)
5. **Code examples** with proper syntax highlighting
6. **Tables** for structured data
7. **Cross-references** using relative links

---

## 🔗 Related Resources

- **Main README:** [../README.md](../README.md) - Project overview
- **Scripts:** [../scripts/](../scripts/) - Training and analysis scripts
- **Configs:** [../configs/](../configs/) - Model and training configurations
- **Outputs:** [../outputs/](../outputs/) - Training results and models

---

**Last Updated:** November 15, 2025  
**Maintainer:** THAT Le Quang ([thatlq1812])
