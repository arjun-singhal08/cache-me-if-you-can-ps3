# Subsystem 1: Structural Health Monitoring (SHM) — Experiment Log & Benchmark

## 1. Problem Formulation & Physics Grounding
* **Objective**: Predict single-value cumulative fatigue damage ($D$) on railway bogie/carbody dynamic stress time-series.
* **Physics Mechanism**: Dynamic stress cycles accumulate irreversible fatigue damage following **Miner's Linear Cumulative Damage Rule** ($D = \sum \frac{n_i}{N_i}$) and **Basquin's S-N Power-Law** ($\sigma_a^m \cdot N = C \implies D \propto \sum n_i (\Delta \sigma_i)^m$).
* **Metric**: $\text{Score} = \max(0, 1 - \text{MAPE})$, where $\text{MAPE} = \frac{1}{N}\sum \frac{|y - \hat{y}|}{y}$.

---

## 2. Feature Engineering Architecture (`src/shm/features.py`)
Tabular representation extracted from raw 581,120-sample stress time series:
1. **Statistical Moments**: Mean, variance, std, RMS, skewness, kurtosis, peak-to-peak amplitude, crest factor, peak factor, mean absolute difference, max absolute difference.
2. **Percentiles & Energy**: 1st, 5th, 25th, 50th, 75th, 90th, 95th, 99th percentiles, IQR, total energy ($\sum x^2$).
3. **ASTM E1049-85 Rainflow Cycle Proxies**:
   - Prominence-gated turning point filtering ($<0.3$s per file).
   - Total cycle count, max range, weighted mean range, std range.
   - Cycle amplitude statistics: mean amplitude, max amplitude, 90th/95th/99th percentile amplitude.
   - High-amplitude cycle counts exceeding stress thresholds ($>10, >20, >40, >60\text{ MPa}$).
   - Multi-exponent S-N damage accumulators: $\sum n_i (\Delta \sigma)^m$ and $\sum n_i (\sigma_a)^m$ for $m \in \{2.5, 3.0, 3.5, 4.0, 4.5, 5.0\}$.
   - 10-bin cycle amplitude distribution histogram.
4. **Spectral Dynamics (FFT)**: Mean, std, max, skewness, kurtosis of FFT magnitude spectrum, plus 5 sub-band energy accumulators.

---

## 3. 5-Fold Cross-Validation Benchmark Results

All regressors were trained on log-transformed targets $\log(1 + \text{damage})$ to mathematically align the optimization objective with the competition MAPE evaluation metric:

| Model | Loss Objective | 5-Fold Out-of-Fold MAPE | Official Validation Score |
| :--- | :--- | :--- | :--- |
| Ridge Regression | L2 Regularized Linear | 7.09% | 0.9291 |
| ElasticNet | L1 + L2 Coordinate Descent | 6.37% | 0.9363 |
| RandomForest Regressor | Absolute Error | 9.25% | 0.9075 |
| GradientBoosting Regressor | Absolute Error | 8.47% | 0.9153 |
| XGBoost Regressor | `reg:absoluteerror` | 10.92% | 0.8908 |
| **ExtraTrees Regressor (Best Model)** | **Absolute Error** | **`5.07%`** | **`0.9493` (94.93%)** |

---

## 4. Final Production Artifacts
* **Saved Model Artifact**: `models/shm_best_model.joblib` (trained on 100% of the 64 training files).
* **Official Prediction CSV**: `predictions/shm_predictions.csv` (16 test files, valid bounds $[0, 1]$, zero nulls).
* **Schema Verification**: Fully validated with `scripts/validate_submission.py`.
