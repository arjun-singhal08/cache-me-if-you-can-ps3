# NebulaX 2026 Hackathon — Problem Statement 3 (PS3)
## Cache Me If You Can — SHM Subsystem Solution

Welcome to the **Cache Me If You Can** repository for NebulaX 2026 Problem Statement 3 (Train Condition Monitoring).

This repository contains the machine learning pipeline for **Subsystem 1: Structural Health Monitoring (SHM)**.

---

### 📊 Performance Summary
* **Validation Strategy**: 5-Fold Cross Validation
* **Primary Metric**: Hackathon SHM Score $\text{Score} = \max(0, 1 - \text{MAPE})$
* **Cross-Validation Score**: **0.9231 (92.31%)**
* **Average Error (MAPE)**: **7.69%**

---

### 🛠 Repository Structure

```
cache-me-if-you-can-ps3/
├── src/
│   ├── extract_features.py   # Signal processing, Rainflow cycle counting, FFT features
│   └── train.py              # Multi-model cross-validation & training pipeline
├── predict.py                # Official submission inference interface
├── requirements.txt          # Python dependencies
└── README.md                 # Documentation
```

---

### 🚀 How to Run

#### 1. Install Dependencies
```bash
python -m pip install -r requirements.txt
```

#### 2. Feature Extraction
Extracts physics-informed, statistical, spectral, and Rainflow fatigue features from raw stress CSVs:
```bash
python src/extract_features.py
```

#### 3. Train Model
Trains models with 5-fold cross-validation and saves the best model (`models/shm_model.pkl`):
```bash
python src/train.py
```

#### 4. Run Inference (Generate Predictions)
```bash
python predict.py --input "path/to/test/csvs" --output "path/to/output/dir"
```

---

### 🔬 Technical Approach
1. **Extrema Filtering**: Fast prominence-based turning point extraction reduces signal points to true stress reversals, enabling rapid Rainflow counting (ASTM E1049-85).
2. **Fatigue Accumulators**: Computes power-law stress range accumulators $\sum n_i (\Delta \sigma)^m$ for $m \in [3, 3.5, 4, 5]$ corresponding to material S-N curves.
3. **Log-Target Relative Error Optimization**: Trains tree regressors on $\log(1 + y)$ target transformation to directly minimize Mean Absolute Percentage Error (MAPE).
