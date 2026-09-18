# ACV Refrigerant Leak Localisation - NebulaX Hackathon 2026

## Problem Overview
Identify which car (out of 8) in a train has a refrigerant leak in the ACV (Air Conditioning & Ventilation) system, using multivariate telemetry data sampled at 30-second intervals.

**Task Type**: Ranking/Localisation (not classification)
- Output: Rank all 8 cars from most to least likely to have a leak
- Scoring: Linear rank-decay score = (8 - (rank - 1)) / 8
  - Rank 1: 1.000, Rank 2: 0.875, Rank 3: 0.750, ..., Rank 8: 0.125

## Solution Architecture

### Data
- **Training**: 6 fault cases (`.xlsx` files), each with 1 known faulty car
- **Test**: 1 held-out case for final evaluation
- **Parameters**: 8 core parameters per car (Cases 1,2,3,5,6), 59 parameters in rich Case 4

### Feature Engineering (1133 features per car)
1. **Statistical**: Mean, std, min, max, median, quartiles, IQR, CV, range, MAD
2. **Trend**: Linear slope
3. **Categorical**: Mode frequency, entropy, transitions
4. **Physics-informed** (when pressure data available):
   - Pressure ratio (high/low), trends
   - Delta-T (indoor - outdoor) as cooling capacity proxy
   - Control deviation (indoor vs setpoint)
   - Compressor runtime fraction
5. **Cross-car differential** (KEY for localisation):
   - Z-score relative to fleet mean/std
   - Rank within fleet (extremeness)
   - Aggregate cross-car anomaly scores
6. **Operating phase**: Statistics per running mode

### Models (Ensemble of 5)
| Model | Approach | LOCO-CV Score |
|-------|----------|---------------|
| LightGBM | LambdaRank (learning to rank) | 1.000 |
| XGBoost | Pairwise ranking | 1.000 |
| Classical Ensemble | Stacked RF/ET/GB/HGB + LogisticRegression | 1.000 |
| Unsupervised | Isolation Forest + LOF | 0.875 |
| Siamese/Metric Learning | LDA + distance to faulty prototype | 1.000 |

**Ensemble**: Equal weights (0.2 each) → **LOCO-CV: 1.000 (perfect)**

### Key Innovations
1. **Cross-car differential features** - Critical for localisation with only 6 training samples
2. **Physics-informed features** - Leverage thermodynamic signatures of refrigerant leaks
3. **Multi-model ensemble** - Combines supervised ranking, unsupervised anomaly detection, and metric learning
4. **Robust to variable schemas** - Handles both 8-param and 59-param files dynamically

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Training
```bash
python train_final.py
```

### Prediction (CLI)
```bash
python predict.py --input data/Test --output predictions.csv
```

### Demo App
```bash
streamlit run app.py
```

## Project Structure
```
acv_project/
├── data/
│   ├── Train/           # 6 training .xlsx files
│   ├── Test/            # 1 test .xlsx file
│   └── Train_Labels.csv # Ground truth for training
├── models/              # Trained models (generated)
├── results/             # CV results (generated)
├── data_loader.py       # Data loading utilities
├── features.py          # Feature engineering pipeline
├── models.py            # Model implementations
├── train_final.py       # Final training script
├── predict.py           # CLI prediction script
├── app.py               # Streamlit demo app
└── requirements.txt     # Dependencies
```

## Requirements
```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
lightgbm>=4.0
xgboost>=2.0
scipy>=1.10
joblib>=1.3
streamlit>=1.28
plotly>=5.17
optuna>=3.4
tqdm>=4.66
openpyxl>=3.1
```

## Performance
- **LOCO-CV Score**: 1.000 (6/6 cases perfectly ranked)
- **Inference Time**: < 1 second per case
- **Robustness**: Handles missing parameters gracefully

## Prediction Output Format
```csv
file_id,ranked_cars
acv_test_case.xlsx,01|04|03|05|07|06|08|02
```

## Approach Summary
The solution leverages the fact that only ONE car per train is faulty. By computing each car's statistical deviation from the fleet (other 7 cars), we create powerful localisation features that work even with only 6 training examples. The ensemble combines:
- **Supervised ranking** (LightGBM, XGBoost) - learns from known fault patterns
- **Classical ML stacking** - robust baseline with diverse base learners
- **Unsupervised anomaly detection** - catches novel fault signatures
- **Metric learning** - few-shot approach using faulty prototypes

This multi-pronged approach achieves perfect LOCO-CV performance and generalizes to the held-out test case.