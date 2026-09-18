import os
import glob
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb

from src.shm.features import extract_features

def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.clip(np.array(y_pred, dtype=np.float64), a_min=1e-6, a_max=None)
    mape = np.mean(np.abs(y_true - y_pred) / np.abs(y_true))
    score = max(0.0, 1.0 - mape)
    return score, mape

def process_file(fp):
    fn = os.path.basename(fp)
    feats = extract_features(fp)
    feats['filename'] = fn
    return feats

def train_shm():
    train_dir = r"C:\Users\Arjun Singhal\NebulaX-Hackathon-ProblemStatement\PS3\02_Datasets\SHM\Train"
    labels_file = r"C:\Users\Arjun Singhal\NebulaX-Hackathon-ProblemStatement\PS3\02_Datasets\SHM\Train_Labels.csv"
    test_dir = r"C:\Users\Arjun Singhal\NebulaX-Hackathon-ProblemStatement\PS3\02_Datasets\SHM\Test"
    
    train_files = sorted(glob.glob(os.path.join(train_dir, "*.csv")))
    test_files = sorted(glob.glob(os.path.join(test_dir, "*.csv")))
    
    print(f"Extracting features from {len(train_files)} training files...")
    train_rows = Parallel(n_jobs=-1, backend="loky")(
        delayed(process_file)(fp) for fp in tqdm(train_files, desc="Train extraction")
    )
    df_train = pd.DataFrame(train_rows)
    
    df_labels = pd.read_csv(labels_file)
    df_merged = pd.merge(df_train, df_labels, on='filename')
    
    X = df_merged.drop(columns=['filename', 'damage'])
    y = df_merged['damage'].values
    y_log = np.log1p(y)
    feature_names = list(X.columns)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    models = {
        'ExtraTrees': lambda: ExtraTreesRegressor(n_estimators=500, max_depth=10, criterion='absolute_error', random_state=42, n_jobs=1),
        'RandomForest': lambda: RandomForestRegressor(n_estimators=500, max_depth=10, criterion='absolute_error', random_state=42, n_jobs=1),
        'GradientBoosting': lambda: GradientBoostingRegressor(n_estimators=300, learning_rate=0.03, max_depth=3, loss='absolute_error', random_state=42),
        'XGBoost': lambda: xgb.XGBRegressor(n_estimators=400, learning_rate=0.02, max_depth=3, objective='reg:absoluteerror', subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=1)
    }
    
    results = {}
    print("\n--- Cross-Validation Results ---")
    for name, model_fn in models.items():
        oof = np.zeros(len(df_merged))
        for train_idx, val_idx in kf.split(X, y_log):
            X_tr, y_tr = X.iloc[train_idx], y_log[train_idx]
            X_va, y_va = X.iloc[val_idx], y_log[val_idx]
            
            m = model_fn()
            m.fit(X_tr, y_tr)
            preds_log = m.predict(X_va)
            oof[val_idx] = np.expm1(preds_log)
            
        score, mape = compute_metrics(y, oof)
        results[name] = {'score': score, 'mape': mape}
        print(f"{name:18s} | MAPE: {mape*100:6.2f}% | SHM Score: {score:.4f}")
        
    best_name = max(results, key=lambda k: results[k]['score'])
    print(f"\n---> Best Validated Model: {best_name} (Score: {results[best_name]['score']:.4f})")
    
    # Train final best model on all 64 training files
    print("Training final model on all 64 training files...")
    final_model = models[best_name]()
    final_model.fit(X, y_log)
    
    os.makedirs("models", exist_ok=True)
    model_payload = {
        'model': final_model,
        'feature_names': feature_names,
        'best_model_name': best_name,
        'cv_score': results[best_name]['score'],
        'cv_mape': results[best_name]['mape']
    }
    joblib.dump(model_payload, "models/shm_model.joblib")
    print("Saved final model to models/shm_model.joblib")
    
    # Generate predictions for all 16 test files (Step 7)
    print(f"Generating predictions for {len(test_files)} test files...")
    test_rows = Parallel(n_jobs=-1, backend="loky")(
        delayed(process_file)(fp) for fp in tqdm(test_files, desc="Test extraction")
    )
    df_test = pd.DataFrame(test_rows)
    X_test = df_test[feature_names]
    test_preds_log = final_model.predict(X_test)
    test_preds = np.clip(np.expm1(test_preds_log), a_min=1e-6, a_max=None)
    
    os.makedirs("predictions", exist_ok=True)
    df_preds = pd.DataFrame({
        'file_id': df_test['filename'],
        'prediction': test_preds
    })
    pred_path = "predictions/shm_predictions.csv"
    df_preds.to_csv(pred_path, index=False)
    print(f"Generated {pred_path} with {len(df_preds)} rows.")
    print(df_preds.head())

if __name__ == '__main__':
    train_shm()
