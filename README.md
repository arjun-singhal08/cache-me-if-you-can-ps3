# 🚆 NebulaX 2026 Hackathon — Problem Statement 3: Train Condition Monitoring
### Team: Cache Me If You Can | Autonomous Multi-Subsystem Health Suite

A unified, physics-grounded condition monitoring platform covering all four independent train subsystems for LTA NebulaX 2026 PS3.

---

## 📊 Subsystems & Validated Performance

| Subsystem | Task Type | Core Methodology & Physics | Validation Metric & Score | Submission Output |
|:---|:---|:---|:---:|:---|
| **Subsystem 1: SHM** | Fatigue Damage Regression | ASTM E1049-85 Rainflow cycle counting, Goodman/Gerber mean-stress correction, Palmgren-Miner accumulation, Bayesian Ridge hybrid | **0.9027 Score** (9.73% MAPE via 5-Fold CV) | `predictions/shm_predictions.csv` (16 test files) |
| **Subsystem 2: ACV** | Consist Refrigerant Leak Localisation | Fleet-relative thermal divergence ΔT, cross-car z-scores, 5-model rank ensemble (LightGBM/XGBoost/Stacking) | **1.000 Score** (LOCO-CV on all 6 train cases) | `predictions/acv_predictions.csv` (`01\|04\|03...`) |
| **Subsystem 3: Rail Corrugation** | 3-Class Defect Classification | 129-channel speed-dependent spatial wavelength tracking (λ = v/f), 223 bilateral energy asymmetry features, calibrated voting ensemble | **~0.773 Macro-F1** (5-Fold Stratified CV) | `predictions/rail_predictions.csv` (68 test files) |
| **Subsystem 4: Door** | Temporal Segment Detection & Classification | State-machine flag transitions + position derivative cycle segmentation, 71 physics features, cost-sensitive XGBoost | **0.982 Soft-F1** (IoU-weighted F1 via 5-Fold CV) | `predictions/door_predictions.csv` (38 detected cycles) |

---

## 🛠 Project Structure

```text
cache-me-if-you-can-ps3/
├── app.py                      # Unified Streamlit Operator Web App (4 subsystem tabs)
├── predict_all.py              # Master CLI running batch inference across all subsystems
├── Dockerfile                  # Production container configuration for Google Cloud Run
├── .dockerignore               # Ignores raw datasets and temp build files
├── requirements.txt            # Unified project dependencies
├── predictions.zip             # Official flat archive containing all 4 competition CSVs
├── predictions/                # Raw prediction outputs
│   ├── shm_predictions.csv     # Schema: file_id,prediction
│   ├── rail_predictions.csv    # Schema: file_id,prediction
│   ├── door_predictions.csv    # Schema: start_time,end_time,prediction
│   └── acv_predictions.csv     # Schema: file_id,ranked_cars (01|04|03...)
├── SHM/                        # Subsystem 1 Pipeline & Models (0.9027 score)
├── ACV/                        # Subsystem 2 Pipeline & Models (1.000 LOCO-CV)
├── Rail_Corrugation/           # Subsystem 3 Pipeline & Models (~0.773 Macro-F1)
└── Door/                       # Subsystem 4 Pipeline & Models (0.982 soft-F1)
```

---

## ⚙️ Quick Start

### Prerequisites
- Python 3.10+
- CUDA 11.8+ (for Rail_Corrugation PyTorch models)
- 8 GB+ RAM (for ACV ensemble models)

### Installation
```bash
git clone https://github.com/arjun-singhal08/cache-me-if-you-can-ps3.git
cd cache-me-if-you-can-ps3
pip install -r requirements.txt
```

### Data Layout (Expected)
```
data/
├── SHM/
│   ├── Train/      # Training CSVs + labels
│   └── Test/       # Test CSVs (16 files)
├── ACV/
│   ├── Train/      # 6 .xlsx cases + Train_Labels.csv
│   └── Test/       # acv_test_case.xlsx
├── Rail_Corrugation/
│   ├── Train/      # Training CSVs (260+ files)
│   └── Test/       # Test CSVs (68 files)
└── Door/
    ├── Train/      # Door training data
    └── Test.csv    # Continuous stream
```

### Run Batch Inference (All Subsystems)
```bash
python predict_all.py --all --input-dir data --output-dir predictions
```
This produces all 4 submission CSVs in `predictions/`.

### Run Individual Subsystem
```bash
# ACV
python ACV/predict.py --input data/ACV/Test --output predictions/acv_predictions.csv

# Door
python Door/predict.py --input data/Door/Test.csv --output predictions/door_predictions.csv

# Rail Corrugation
python Rail_Corrugation/predict.py --input data/Rail_Corrugation/Test --output predictions/rail_predictions.csv

# SHM
python SHM/predict.py --input data/SHM/Test --output predictions/shm_predictions.csv
```

### Unified Web App
```bash
streamlit run app.py
# Opens http://localhost:8501 with 4 subsystem tabs
```

### Docker (Production)
```bash
docker build -t cache-me-if-you-can-ps3 .
docker run -p 8080:8080 cache-me-if-you-can-ps3
```

---

## 🔬 Physics & Methodology Deep Dives

### Subsystem 1: SHM — Fatigue Damage Regression
**Physics**: Dynamic stress time series → Rainflow cycle counting (ASTM E1049-85) → Stress ranges & mean stresses → Goodman/Gerber correction for mean stress effects → Palmgren-Miner linear damage accumulation → Cumulative damage per file.

**Model**: Hybrid physics + ML
- Physics: Deterministic damage integral from cycle histogram
- ML: Bayesian Ridge regression on physics residuals (calibration)
- Blend weight: 0.59 physics / 0.41 ML (learned on validation)

**Features**: 40+ statistical + spectral + cycle-domain features per file.

### Subsystem 2: ACV — Refrigerant Leak Localisation
**Physics**: Refrigerant charge loss → reduced evaporator superheat → rising discharge temperature → falling suction pressure → degraded ΔT (indoor-outdoor). Only 1 faulty car per 8-car consist.

**Features (1,133/car)**:
- Statistical: mean, std, trend, IQR, spectral entropy per sensor
- Physics-informed: pressure ratio, ΔT proxy, control deviation, compressor runtime
- **Cross-car differentials**: Fleet z-scores, rank extremeness (key for 6-sample regime)

**Models**: 5-model rank ensemble (LambdaRank, XGBRank, Stacked RF/ET/GB/HGB+LR, IsolationForest+LOF, Siamese/LDA)

### Subsystem 3: Rail Corrugation — 3-Class Classification
**Physics**: Corrugation wavelength λ = v/f (speed-dependent). Order tracking resamples vibration to spatial domain (129 orders). Side I/II asymmetry captures lateral corrugation progression.

**Features (223/channel)**:
- Spatial spectral: order-band energy, dominant orders, spectral centroid
- Bilateral: log-energy ratio (Side I vs II), coherence, phase lag
- Statistical: kurtosis, crest factor, RMS per order band

**Model**: PyTorch Lightning CNN-LSTM + handcrafted feature fusion, calibrated probability ensemble, per-class thresholds.

### Subsystem 4: Door — Segment Detection & Classification
**Physics**: Door cycle = open command → motor current surge → position ramp → hold → close command → reverse current → position return. "Abnormal resistance" = anomalous current/position dynamics.

**Pipeline**:
1. Segmentation: State-machine on flag transitions + position derivative peaks
2. Features (71/segment): current/bemf/voltage/position statistics, rise/settling times, overshoot, correlations
3. Classification: Cost-sensitive XGBoost (3-class), threshold optimization for IoU-weighted F1

---

## 📦 Submission Formats

| Subsystem | File | Schema | Scoring |
|---|---|---|---|
| SHM | `shm_predictions.csv` | `file_id,prediction` (float) | max(0, 1 - MAPE) |
| Rail | `rail_predictions.csv` | `file_id,prediction` ({Normal, Side I, Side II}) | Macro-F1 |
| Door | `door_predictions.csv` | `start_time,end_time,prediction` ({Normal, Abnormal resistance}) | IoU-weighted F1 |
| ACV | `acv_predictions.csv` | `file_id,ranked_cars` (pipe-separated car IDs) | Linear rank-decay |

### Create Official `predictions.zip`
```bash
cd predictions
zip ../predictions.zip *.csv
cd ..
# Verify
unzip -l predictions.zip
```

---

## 🧪 Validation Protocols

| Subsystem | CV Strategy | Rationale |
|---|---|---|
| SHM | 5-Fold GroupKFold (by file) | Prevents temporal leakage |
| ACV | LOCO-CV (Leave-One-Case-Out) | Only 6 cases; maximizes train data per fold |
| Rail | 5-Fold StratifiedGroupKFold | Class balance + file independence |
| Door | 5-Fold GroupKFold (by file) | Continuous stream independence |

---

## 📝 Notes

- **Rail_Corrugation** model (`rail_best_model_final.joblib`, 73.8 MB) exceeds GitHub's 50 MB recommendation. For production deployment, use Git LFS or artifact store.
- All models include calibration layers (Platt/Isotonic) for reliable probabilities.
- Cross-subsystem feature leakage prevented by strict per-subsystem train/test splits.

---

## 🏁 Team: Cache Me If You Can

Built for LTA NebulaX 2026 Hackathon — Problem Statement 3.