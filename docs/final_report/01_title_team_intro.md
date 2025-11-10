# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# Section 1-3: Title, Team Members & Introduction

## 1. Project Title

**Multi-Modal Spatio-Temporal Graph Transformer for Real-Time Traffic Speed Forecasting in Ho Chi Minh City**

**Short Title:** STMGT Traffic Forecasting System

---

## 2. Team Members

<!-- TODO: UPDATE with actual team member information -->

| Name                       | Role                   | Responsibilities                                                             |
| -------------------------- | ---------------------- | ---------------------------------------------------------------------------- |
| THAT Le Quang (thatlq1812) | Developer & Researcher | Model architecture design, implementation, training pipeline, API deployment |

---

## 3. Introduction and Background

### 3.1 Objective

This project aims to develop an accurate and reliable traffic speed forecasting system for Ho Chi Minh City using deep learning techniques. The primary objectives are:

1. **Accurate Short-Term Forecasting:** Predict traffic speeds for the next 15 minutes to 3 hours with high precision (target MAE < 5 km/h)

2. **Uncertainty Quantification:** Provide confidence intervals for predictions to support risk-aware decision-making

3. **Multi-Modal Integration:** Incorporate weather conditions and temporal patterns alongside spatial road network structure

4. **Real-Time Deployment:** Deploy a production-ready API capable of serving predictions with low latency (<500ms)

5. **Comparative Analysis:** Benchmark against established baseline models (LSTM, GCN, GraphWaveNet, ASTGCN) to validate architectural improvements

### 3.2 Motivation

Traffic congestion is a critical challenge in rapidly urbanizing cities, particularly in Ho Chi Minh City, Vietnam. According to recent studies:

- **Economic Impact:** Traffic congestion costs approximately **$1.2 billion USD annually** in lost productivity and fuel consumption
- **Travel Time:** Average commute times have increased by **35% over the past 5 years**
- **Environmental Cost:** Congestion contributes to increased CO2 emissions and air pollution
- **Quality of Life:** Extended commute times negatively impact citizen well-being and urban livability

Accurate traffic forecasting can enable:

- **Intelligent Route Planning:** Help drivers avoid congested routes, reducing travel time by 15-20%
- **Traffic Management:** Allow authorities to implement proactive traffic control measures
- **Public Transportation Optimization:** Improve bus scheduling and route planning
- **Emergency Response:** Enable faster emergency vehicle routing during critical situations
- **Urban Planning:** Provide data-driven insights for infrastructure development decisions

### 3.3 Why Deep Learning & Graph Neural Networks?

Traditional traffic forecasting methods (ARIMA, Kalman filters) struggle with:

- **Non-linear patterns** in traffic flow
- **Complex spatial dependencies** across road networks
- **Multi-modal interactions** (weather, events, accidents)

**Graph Neural Networks (GNNs)** address these challenges by:

- Modeling road networks as graphs (nodes = intersections, edges = road segments)
- Capturing spatial dependencies through message passing
- Learning adaptive representations of network topology

**Transformers** enhance temporal modeling through:

- Self-attention mechanisms for long-range dependencies
- Parallel processing of time sequences
- Better handling of irregular temporal patterns

### 3.4 Background Information

#### 3.4.1 Ho Chi Minh City Context

- **Population:** ~9 million (metropolitan area: ~13 million)
- **Road Network:** 3,200+ km of roads, 15,000+ intersections
- **Traffic Volume:** 8+ million motorcycles, 600,000+ cars
- **Peak Hours:** 7-9 AM, 5-7 PM (severe congestion)
- **Weather Impact:** Tropical monsoon climate with heavy rainfall affecting traffic patterns

#### 3.4.2 Data Collection Infrastructure

Our system leverages:

- **Google Directions API:** Real-time traffic speed data
- **OpenWeatherMap API:** Weather conditions (temperature, humidity, rainfall)
- **OpenStreetMap/Overpass API:** Road network topology
- **Collection Frequency:** Every 15 minutes during peak hours
- **Spatial Coverage:** 62 key intersections, 144 road segments

#### 3.4.3 Technical Challenge

The core challenge is predicting **spatially-dependent, temporally-evolving** traffic speeds while accounting for:

- **Spatial correlations:** Speed on adjacent roads influences each other
- **Temporal dynamics:** Traffic patterns vary by time-of-day, day-of-week
- **External factors:** Weather, accidents, events
- **Uncertainty:** Inherent randomness in driver behavior

### 3.5 Research Questions

This project addresses the following research questions:

1. **Can a unified spatio-temporal architecture outperform separate spatial and temporal processing?**

   - Hypothesis: Parallel spatial-temporal blocks with learned fusion achieve better performance

2. **How effective is Gaussian Mixture Modeling for traffic speed uncertainty quantification?**

   - Hypothesis: K=3-5 mixture components capture multi-modal speed distributions

3. **Does weather cross-attention provide meaningful improvements over simple concatenation?**

   - Hypothesis: State-dependent weather effects require cross-attention mechanism

4. **What is the realistic performance ceiling for a 62-node network with limited training data?**

   - Hypothesis: R² = 0.45-0.55 achievable with ~16K samples (vs R² = 0.80+ for large-scale datasets)

5. **Can the model generalize to unseen traffic patterns after deployment?**
   - Hypothesis: Robust regularization enables real-world deployment with retraining every 1-2 weeks

---

## Key Contributions

1. **Novel Architecture:** First application of parallel spatio-temporal processing with Gaussian mixture outputs for traffic forecasting in Vietnam

2. **Multi-Modal Fusion:** Weather cross-attention mechanism for context-aware predictions

3. **Production Deployment:** End-to-end system from data collection to API serving with <400ms inference latency

4. **Comprehensive Benchmarking:** Systematic comparison against 4 baseline architectures (LSTM, GCN, GraphWaveNet, ASTGCN)

5. **Open Source Implementation:** Fully documented codebase with training scripts, API, and deployment guides

---

**Next:** [Literature Review →](02_literature_review.md)
