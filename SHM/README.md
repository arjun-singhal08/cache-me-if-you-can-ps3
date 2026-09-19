# SHM Subsystem - Structural Health Monitoring

## Overview
This subsystem implements a hybrid physics-ML pipeline for cumulative fatigue damage prediction in rail vehicle structural health monitoring (SHM). The approach combines ASTM E1049-85 Rainflow cycle counting with mean-stress corrections (Goodman, Gerber, SWT) and multi-exponent Palmgren-Miner damage accumulation, feeding into a Bayesian Ridge regression in log-space, blended with an ensemble of XGBoost, LightGBM, and CatBoost models.

## Performance
- **Official SHM Score (max(0, 1 - MAPE))**: **0.9027**
- **OOF MAPE**: **9.73%**
- **Training samples**: 64
- **Test samples**: 16

## Physics-Based Feature Engineering
- **ASTM E1049-85 Rainflow** cycle counting (4-point algorithm)
- **Mean-stress corrections**: Goodman, Gerber, Smith-Watson-Topper (SWT)
- **Multi-exponent Miner damage**: m ∈ {3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0, 10.0}
- **Spectral fatigue estimators**: Rayleigh, Dirlik, Tovo-Benasciutto
- **67 physics features** per file (581,120 time steps per file)

## Hybrid Model Architecture
```
Physics Ridge (Bayesian)          Tree Ensemble (XGB + LGB + CatBoost)
       |                                    |
       |-- log(physics_features) → log(D)  |-- log(target) → D
       |                                    |
       +-------- Blended (w=0.465) --------+
                          |
                   Isotonic Calibration
                          |
                     Final Prediction
```

## Directory Structure
```
SHM/
├── predict.py              # CLI inference entry point
├── requirements.txt        # SHM-specific dependencies
├── README.md               # This file
├── src/
│   ├── features/
│   │   ├── physics_features.py   # Rainflow + Miner + Spectral features
│   │   ├── rainflow.py           # Numba-accelerated ASTM E1049-85 Rainflow
│   │   └── __init__.py
│   ├── models/
│   │   ├── hybrid_model.py       # Hybrid ensemble implementation
│   │   └── __init__.py
│   └── utils/
│       └── io.py                 # Data loading utilities
├── models/
│   ├── physics_model.pkl     # Bayesian Ridge on physics features
│   ├── tree_models.pkl       # XGB + LGB + CatBoost CV models
│   ├── blend_weight.pkl      # Optimal blend weight (0.465)
│   ├── physics_indices.pkl   # Physics feature column indices
│   ├── feature_cols.pkl      # Full feature column names
│   ├── global_calibrator.pkl # Isotonic regression
│   └── tree_calibrators.pkl  # Per-model Isotonic calibrators
└── models_hybrid/            # Duplicate for backward compatibility
```

## Usage

### Training
```bash
python train_hybrid.py --data-dir data --output-dir models_hybrid --n-jobs 4 --n-folds 5
```

### Inference
```bash
python predict.py --input data/Test --output shm_predictions.csv --model-dir models_hybrid --n-tta 20
```

### Output Format
```csv
file_id,prediction
test01.csv,0.0386176630838894
test02.csv,0.7899146147521647
...
```

## Competition Format
The official submission format requires `shm_predictions.csv` with columns:
- `file_id`: Test file name (e.g., `test01.csv`)
- `prediction`: Predicted cumulative fatigue damage (float)

## Scoring Metric
The official metric is `max(0, 1 - MAPE)` where:
- MAPE = Mean Absolute Percentage Error = mean(|y_true - y_pred| / |y_true|)
- Score ranges from 0 to 1 (1 = perfect prediction)
- Our model achieves **0.9027** (9.73% MAPE)

## Dependencies
See `requirements.txt` for full list. Key packages:
- numpy, pandas, scikit-learn
- xgboost, lightgbm, catboost
- numba (for Rainflow acceleration)
- scipy, joblib, tqdm