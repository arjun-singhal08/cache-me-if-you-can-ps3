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

---

# Subsystem 3: Rail Corrugation — Experiment Log & Benchmark

## 1. Problem Formulation & Physics Grounding
* **Objective**: Predict rail corrugation defect class: `Normal`, `Side I`, or `Side II` from 1-second 10,000 Hz axle-box multi-channel recordings.
* **Physics Mechanism**: Periodic surface wear on rail running surface provokes high-frequency wheel-rail contact resonance (pinned-pinned resonance 300–800 Hz and P2 resonance 100–300 Hz) measured by vertical accelerometers on 8 cars across 8 axle boxes:
  - Positions 1, 3, 5, 7: Side I (Left rail)
  - Positions 2, 4, 6, 8: Side II (Right rail)
* **Metric**: **Macro F1** across all 3 classes (unweighted average of class F1 scores) and **Balanced Accuracy**, ensuring detection of rare fault classes (`Side I`: 14 samples, `Side II`: 24 samples out of 272 training recordings).

---

## 2. Feature Engineering Architecture (`src/rail/features.py`)
139 physics-grounded features extracted from 129 channels (10,000 samples per channel):
1. **Operating Condition**: Rotating speed mean, std, max, and pulse transition frequency (toothed wheel tachometer).
2. **Left/Right Time-Domain Moments**: Mean, std, variance, RMS, peak-to-peak amplitude, skewness, kurtosis, crest factor, shape factor, impulse factor for Side I Vibration, Side II Vibration, Side I Shock, and Side II Shock.
3. **Bilateral Asymmetry Metrics**:
   - RMS and energy ratios ($\text{RMS}_{s1} / \text{RMS}_{s2}$, $E_{s1} / E_{s2}$).
   - RMS differentials and peak-to-peak differentials.
   - Normalized Asymmetry Index: $(S_1 - S_2) / (S_1 + S_2)$ bounded in $[-1, +1]$.
   - Shock energy asymmetry ratios and differentials.
4. **Per-Car Localized Asymmetry & Cross-Correlation**:
   - Individual car vibration and shock RMS values for Side I vs Side II.
   - Per-car asymmetry ratios ($\max_{car}$, $\min_{car}$, differential).
   - Pairwise Pearson cross-correlations across paired left and right axle-box sensors (mean and min).
5. **Spectral & Band Imbalance (Welch PSD)**:
   - Dominant peak frequencies for Side I vs Side II.
   - Spectral centroids for Side I vs Side II.
   - Energy and bilateral imbalance in 5 critical resonance bands:
     - Band 1 (20–100 Hz): Track & sleeper passing
     - Band 2 (100–300 Hz): P2 wheel-rail resonance
     - Band 3 (300–800 Hz): Pinned-pinned corrugation resonance
     - Band 4 (800–1500 Hz): Corrugation wear / high-frequency harmonics
     - Band 5 (1500–3000 Hz): Wheel-rail shock / impact

---

## 3. 5-Fold Stratified Cross-Validation Benchmark Results

| Model | Class Balancing | 5-Fold Macro F1 | Balanced Accuracy | Overall Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| GradientBoosting Classifier | None | 59.49% | 58.19% | 88.97% |
| ExtraTrees Classifier | Balanced | 58.12% | 56.66% | 88.60% |
| LightGBM Classifier | Balanced | 68.78% | 67.00% | 89.71% |
| **RandomForest Classifier (Best Model)** | **Balanced** | **`69.19%`** | **`68.81%`** | **`90.81%`** |

### Winning Model (RandomForest) Out-of-Fold Matrix:
```
              Pred Normal  Pred Side I  Pred Side II
True Normal           224            5             5
True Side I             8            5             1
True Side II            3            3            18
```

---

## 4. Final Production Artifacts
* **Saved Model Artifact**: `models/rail_best_model.joblib` & `models/rail_model.joblib` (trained on 100% of the 272 training files).
* **Official Prediction CSV**: `predictions/rail_predictions.csv` (68 test files: 59 Normal, 5 Side I, 4 Side II).
* **FastAPI Service**: `src/rail/api.py` with `POST /predict_rail` and `/health`.
* **Schema Verification**: 100% compliance certified by `scripts/validate_submission.py`.
