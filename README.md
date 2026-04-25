# 🚀 NASA CMAPSS Turbofan Engine RUL Prediction
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-latest-green.svg)](https://scikit-learn.org)
[![HuggingFace Space](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Deployment-yellow)](https://huggingface.co/spaces)

> **Research-Grade Predictive Maintenance Pipeline**  
> Predicting Remaining Useful Life (RUL) and detecting anomalies in aerospace engines using the NASA CMAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset.

---

## 🌟 Project Overview
This project transforms the standard CMAPSS dataset into a production-ready research platform. We implement advanced feature engineering, multi-model ensemble strategies, and **real-world sensor noise simulation** to benchmark model robustness in hostile environments.

### Key Features:
- **Condition-Based Normalization**: Automated K-Means clustering for operating regimes (FD002/FD004).
- **Hybrid Modeling**: LSTM (Temporal Pattern Recognition) + Random Forest (Feature-based Baseline).
- **Anomaly Detection**: Unsupervised Autoencoder for reconstruction-error-based health monitoring.
- **Noise Injection Suite**: Simulate Gaussian noise, sensor dropout, drift, and outlier spikes.
- **Interactive Dashboard**: Premium Plotly/JS dashboard for real-time performance visualization.

---

## 📊 Research Benchmark Results

| Dataset | Best Model | RMSE | F1 Score | Warning Rate | NASA Score |
|---------|------------|------|----------|--------------|------------|
| **FD001** | Ensemble | 16.38 | 0.8186 | 80% | 5.1e+04 |
| **FD002** | Ensemble | 18.67 | 0.8132 | 92% | 1.5e+05 |
| **FD003** | Ensemble | 20.47 | 0.8416 | 79% | 9.8e+04 |
| **FD004** | Ensemble | 19.13 | 0.7273 | 72% | 3.4e+05 |

### Robustness Finding:
The **LSTM** model exhibits significantly higher resilience to sensor noise than traditional ML models. Under "High" noise simulation (drift + dropouts), LSTM maintained a **27% more stable** RUL prediction than Random Forest, proving the value of temporal context in predictive maintenance.

---

## 🛠️ Installation & Usage

### 1. Setup Environment
```bash
pip install -r requirements.txt
```

### 2. Run Full Pipeline (Training + Eval)
```bash
python src/evaluate.py --all
```

### 3. Run Robustness Test
```bash
python src/robustness_eval.py
```

### 4. Launch Local Dashboard
Simply open `results/dashboard.html` in any browser or run:
```bash
python src/dashboard_gen.py
```

---

## 🏗️ Project Structure
```text
turbofan-rul-prediction/
├── data/               # Raw CMAPSS text files
├── src/
│   ├── data_prep.py    # Ingestion & Clustering
│   ├── features.py     # Feature Engineering & Sequences
│   ├── models/         # LSTM, RF, AE, Ensemble
│   ├── noise_injection.py # Sensor Degradation Logic
│   ├── robustness_eval.py # Graceful Degradation Testing
│   └── evaluate.py     # NASA Metrics & Pipeline Orchestration
├── results/
│   ├── metrics/        # CSV benchmarks & findings
│   ├── plots/          # Training curves & comparisons
│   └── dashboard.html  # Interactive UI
└── README.md
```

---

## 📜 Findings
Detailed research analysis, including hardware requirements and deployment strategies, can be found in [results/findings.md](./results/findings.md).

## 🚀 Deployment
This model is ready for deployment to **HuggingFace Spaces** using Streamlit. See `huggingface_deploy/` for assets.

---
*Created by Antigravity AI Assistant.*
