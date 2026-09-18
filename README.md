# NebulaX 2026 Hackathon - Problem Statement 3: Train Condition Monitoring

Unified solution for multiple subsystems of rail vehicle condition monitoring.

## Subsystems Implemented

### 🚂 ACV (Air Conditioning & Ventilation) - Refrigerant Leak Localisation
**Task**: Rank 8 train cars by likelihood of refrigerant leak
- **Approach**: Ensemble of 5 models (LightGBM Ranker, XGBoost Ranker, Classical Stacking, Unsupervised Anomaly Detection, Siamese/Metric Learning)
- **Features**: 1133 per car (statistical, physics-informed, cross-car differential, operating phase)
- **Validation**: LOCO-CV (Leave-One-Case-Out) score: **1.000** (perfect on all 6 training cases)
- **Input**: `.xlsx` files with multivariate telemetry (30-sec sampling)
- **Output**: `acv_predictions.csv` with ranked cars

### 🚪 Door - Temporal Segment Detection & Classification
**Task**: Find each door open/close cycle in continuous stream, classify as Normal vs Abnormal Resistance
- **Approach**: 2-stage pipeline - Segmentation (PELT change-point detection) → Classification (Ensemble)
- **Features**: Segment-level statistical, shape, and physics features
- **Input**: Continuous `Test.csv` stream (motor current, voltage, back-EMF, door position)
- **Output**: `door_predictions.csv` with segments (start_time, end_time, prediction)

## Project Structure
```
nebula-ps3-solution/
├── predict_all.py              # Master prediction script
├── requirements.txt            # Combined dependencies
├── README.md                   # This file
├── ACV/                        # ACV Subsystem
│   ├── data_loader.py          # Fast parquet-based data loading
│   ├── features.py             # 1133 features (statistical + physics + cross-car)
│   ├── models.py               # 5 models + ensemble optimization
│   ├── train_final.py          # Train on all data, save models
│   ├── predict.py              # CLI: python predict.py --input --output
│   ├── app.py                  # Streamlit demo app
│   ├── config.yaml             # ACV config
│   ├── requirements.txt        # ACV-specific deps
│   └── README.md               # ACV documentation
├── Door/                       # Door Subsystem
│   ├── predict.py              # CLI: python predict.py --input Test.csv --output
│   ├── train.py                # Training script
│   ├── config.yaml             # Door config
│   ├── requirements.txt        # Door-specific deps
│   ├── classification/         # Ensemble classifier
│   ├── features/               # Segment feature extraction
│   ├── segmentation/           # PELT-based door cycle detection
│   ├── postprocess/            # IoU optimization
│   ├── utils/                  # IO, metrics
│   └── model/                  # Trained models (.pkl)
└── shared/                     # Shared utilities (future)
```

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Run Predictions

#### All Subsystems (Competition Format)
```bash
# Data structure expected:
# data/
#   ACV/Test/acv_test_case.xlsx
#   Door/Test.csv

python predict_all.py --all --input-dir data --output-dir predictions
```

#### Individual Subsystems

**ACV:**
```bash
cd ACV
python train_final.py                    # Train models (needs data/Train/)
python predict.py --input data/Test --output ../predictions/acv_predictions.csv
# Or demo app:
streamlit run app.py
```

**Door:**
```bash
cd Door
python train.py                          # Train models (needs data/)
python predict.py --input ../data/Door/Test.csv --output ../predictions/door_predictions.csv
```

## Competition Submission Format

### ACV (`acv_predictions.csv`)
```csv
file_id,ranked_cars
acv_test_case.xlsx,01|04|03|05|07|06|08|02
```
**Scoring**: Linear rank-decay: `(8 - (rank - 1)) / 8` → Rank 1=1.0, Rank 2=0.875, ...

### Door (`door_predictions.csv`)
```csv
start_time,end_time,prediction
2024-01-15 08:00:00,2024-01-15 08:00:15,Normal
2024-01-15 08:15:30,2024-01-15 08:15:45,Abnormal resistance
```
**Scoring**: IoU-weighted F1 (timing + label accuracy)

## Data Requirements

Place data in:
```
data/
├── ACV/
│   ├── Train/              # 6 .xlsx files + Train_Labels.csv
│   └── Test/               # acv_test_case.xlsx
└── Door/
    ├── Train/              # Door training data
    └── Test.csv            # Continuous test stream
```

## Key Innovations

### ACV
- **Cross-car differential features**: Z-scores and fleet ranks exploit "only 1 faulty car per train"
- **Physics-informed features**: Pressure ratios, delta-T, control deviations when available
- **Multi-model ensemble**: Supervised ranking + unsupervised anomaly + metric learning
- **Schema-agnostic**: Handles both 8-param and 59-param file formats

### Door
- **PELT change-point detection**: Accurate door cycle segmentation
- **Segment-level features**: Shape, energy, physics-informed
- **IoU-aware post-processing**: Optimizes segment boundaries for scoring metric

## Performance

| Subsystem | Validation Score | Method |
|-----------|------------------|--------|
| ACV | 1.000 (LOCO-CV) | 5-model ensemble |
| Door | TBD (CV) | Segmentation + Ensemble |

## License
For NebulaX 2026 Hackathon submission only.