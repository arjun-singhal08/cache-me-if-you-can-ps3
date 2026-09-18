import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb

from src.shm.data_loader import load_training_dataset, load_test_dataset
from src.shm.features import extract_features

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Compute official PS3 competition metrics:
    MAPE = mean(|y_true - y_pred| / |y_true|)
    Score = max(0, 1 - MAPE)
    """
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.clip(np.array(y_pred, dtype=np.float64), a_min=1e-6, a_max=None)
    mape = float(np.mean(np.abs(y_true - y_pred) / np.abs(y_true)))
    score = float(max(0.0, 1.0 - mape))
    return score, mape

def _extract_single_file(file_path: Path):
    fn = file_path.name
    feats = extract_features(file_path)
    feats['filename'] = fn
    return feats

def train_pipeline(data_dir=None):
    print("=" * 65)
    print("LTA NEBULAX: PS3 STRUCTURAL HEALTH MONITORING (SHM) TRAINING")
    print("=" * 65)

    # 1. Load Dataset using cross-platform Pathlib
    train_files, df_labels = load_training_dataset(data_dir)
    print(f"Discovered {len(train_files)} training files.")

    # 2. Extract Features in Parallel
    print("\n[Step 1/3] Extracting physics & ASTM Rainflow fatigue features...")
    train_rows = Parallel(n_jobs=-1, backend="loky")(
        delayed(_extract_single_file)(fp) for fp in tqdm(train_files, desc="Processing Train Signals")
    )
    df_train = pd.DataFrame(train_rows)
    df_merged = pd.merge(df_train, df_labels, on='filename')

    X = df_merged.drop(columns=['filename', 'damage'])
    y = df_merged['damage'].values
    y_log = np.log1p(y)  # Aligns squared/absolute loss directly with relative percentage error (MAPE)
    feature_names = list(X.columns)

    print(f"Extracted {len(feature_names)} engineered features per sample.")

    # 3. 5-Fold Cross-Validation Benchmark
    print("\n[Step 2/3] Running 5-Fold Cross-Validation on Regressor Baselines...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        'Ridge': lambda: Ridge(alpha=10.0),
        'ElasticNet': lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
        'RandomForest': lambda: RandomForestRegressor(n_estimators=500, max_depth=10, criterion='absolute_error', random_state=42, n_jobs=1),
        'ExtraTrees': lambda: ExtraTreesRegressor(n_estimators=500, max_depth=10, criterion='absolute_error', random_state=42, n_jobs=1),
        'GradientBoosting': lambda: GradientBoostingRegressor(n_estimators=300, learning_rate=0.03, max_depth=3, loss='absolute_error', random_state=42),
        'XGBoost': lambda: xgb.XGBRegressor(n_estimators=400, learning_rate=0.02, max_depth=3, objective='reg:absoluteerror', subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=1)
    }

    results = {}
    oof_predictions = {}

    print("-" * 65)
    print(f"{'Model Name':<20} | {'5-Fold MAPE':<15} | {'Official SHM Score':<18}")
    print("-" * 65)

    for name, model_factory in models.items():
        oof = np.zeros(len(df_merged))
        for train_idx, val_idx in kf.split(X, y_log):
            X_tr, y_tr = X.iloc[train_idx], y_log[train_idx]
            X_va, y_va = X.iloc[val_idx], y_log[val_idx]

            X_tr = X_tr.fillna(0.0)
            X_va = X_va.fillna(0.0)

            model = model_factory()
            model.fit(X_tr, y_tr)
            preds_log = model.predict(X_va)
            oof[val_idx] = np.expm1(preds_log)

        score, mape = compute_metrics(y, oof)
        results[name] = {'score': score, 'mape': mape}
        oof_predictions[name] = oof
        print(f"{name:<20} | {mape * 100:>10.2f}%    | {score:>16.4f}")

    print("-" * 65)
    best_model_name = max(results, key=lambda k: results[k]['score'])
    best_score = results[best_model_name]['score']
    best_mape = results[best_model_name]['mape']
    print(f"[BEST MODEL] {best_model_name} (MAPE: {best_mape*100:.2f}%, Official Score: {best_score:.4f})")

    # 4. Train Best Model on 100% of Training Data & Save Artifact
    print("\n[Step 3/3] Training final best model on all 64 files and saving artifact...")
    final_model = models[best_model_name]()
    final_model.fit(X.fillna(0.0), y_log)

    models_dir = Path(__file__).resolve().parents[2] / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    artifact_payload = {
        'model': final_model,
        'feature_names': feature_names,
        'best_model_name': best_model_name,
        'cv_score': best_score,
        'cv_mape': best_mape,
        'target_transform': 'log1p'
    }

    best_model_path = models_dir / "shm_best_model.joblib"
    joblib.dump(artifact_payload, best_model_path)
    
    compat_model_path = models_dir / "shm_model.joblib"
    joblib.dump(artifact_payload, compat_model_path)

    print(f"[SUCCESS] Saved model artifact to: {best_model_path}")

    return final_model, feature_names, best_score

if __name__ == '__main__':
    train_pipeline()
