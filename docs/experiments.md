# Subsystem 1: Structural Health Monitoring (SHM) — Experiment Log

## 1. Data Inspection & Understanding

- **Data Domain**: Dynamic stress time-series from train bogie/carbody strain gauge sensors.
- **Data Shape**: 
  - Train: 64 CSV files (`train01.csv` to `train64.csv`), each with 581,120 rows (single-column float measurements). Total: ~37.2 million measurements.
  - Test: 16 CSV files (`test01.csv` to `test16.csv`), unlabelled.
  - Target: `damage` in `Train_Labels.csv` (cumulative fatigue damage).
- **Target Distribution**:
  - Min: 0.0286
  - Median: 0.1113
  - Max: 0.8947
  - Distribution is right-skewed with many small values.
- **Physical Ground Truth Mechanism**:
  - Material fatigue follows **Miner's Linear Cumulative Damage Rule**:
    $$D = \sum_{i} \frac{n_i}{N_i}$$
  - S-N Curve relationship: $\sigma_a^m \cdot N = C \implies D \propto \sum n_i (\Delta \sigma_i)^m$, where $m \in [2.5, 5.0]$ is the material Basquin slope exponent.

---

## 2. Feature Engineering (`src/shm/features.py`)

Every raw signal (581,120 samples) is converted into a structured feature vector:
1. **Statistical & Time-Domain Moments**: Mean, Standard Deviation, Min, Max, Peak-to-Peak (PTP), Root Mean Square (RMS), Crest Factor, Peak Factor, Skewness, Kurtosis, Mean Absolute Deviation (MAD), Quantiles ($1\%, 5\%, 25\%, 50\%, 75\%, 95\%, 99\%$), Interquartile Range (IQR), Signal Energy ($\sum x^2$).
2. **Prominence-Gated Rainflow Cycle Counting (ASTM E1049-85)**:
   - Peak/valley turning point filter reducing 580k points to key extrema in $<0.4$s without loss of fatigue precision.
   - Cycle count, total cycles, maximum stress range, weighted mean stress range, stress range standard deviation.
   - **S-N Damage Accumulators**: $\sum n_i (\Delta\sigma)^m$ for $m \in \{2.5, 3.0, 3.5, 4.0, 4.5, 5.0\}$.
   - 10-bin stress range histogram.
3. **Spectral / Frequency Domain (FFT)**: Mean, Std, Max, Skewness, Kurtosis of FFT magnitude spectrum, plus 5 spectral sub-band energies.

---

## 3. Cross-Validation & Modeling Benchmark (5-Fold CV)

Evaluation Metric: **Hackathon SHM Score** $\max(0, 1 - \text{MAPE})$ where $\text{MAPE} = \frac{1}{N}\sum |\frac{y - \hat{y}}{y}|$.

| Model | Loss Function / Objective | Target Space | Out-Of-Fold MAPE | Official SHM Score |
| :--- | :--- | :--- | :--- | :--- |
| LightGBM Baseline | L2 / MSE | Linear $y$ | 60.14% | 0.3986 |
| LightGBM + Log Target | MAPE | $\log(1+y)$ | 36.42% | 0.6358 |
| XGBoost | Absolute Error | $\log(1+y)$ | 11.16% | 0.8884 |
| RandomForest | Absolute Error | $\log(1+y)$ | 10.14% | 0.8986 |
| GradientBoosting | Absolute Error | $\log(1+y)$ | 8.86% | 0.9114 |
| **ExtraTrees Regressor (Final)** | **Absolute Error** | **$\log(1+y)$** | **`6.28%`** | **`0.9372` (93.72%)** |

---

## 4. Final Model Artifacts

- **Trained Model**: `models/shm_model.joblib` (trained on 100% of the 64 training files).
- **Test Predictions**: `predictions/shm_predictions.csv` (16 test files).
- **Validation**: Verified against official example submission schema (`file_id,prediction`).
