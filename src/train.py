import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import lightgbm as lgb
import xgboost as xgb

def compute_metrics(y_true, y_pred):
    """
    Computes MAPE and official hackathon SHM score max(0, 1 - MAPE).
    """
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    y_pred = np.clip(y_pred, a_min=1e-6, a_max=None)
    
    mape = np.mean(np.abs(y_true - y_pred) / np.abs(y_true))
    score = max(0.0, 1.0 - mape)
    return score, mape

def train_and_evaluate():
    features_path = "data_processed/train_features.csv"
    labels_path = r"C:\Users\Arjun Singhal\NebulaX-Hackathon-ProblemStatement\PS3\02_Datasets\SHM\Train_Labels.csv"
    
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Feature file {features_path} not found. Run extract_features.py first.")
        
    df_feats = pd.read_csv(features_path)
    df_labels = pd.read_csv(labels_path)
    
    df_data = pd.merge(df_feats, df_labels, on='filename')
    print(f"Loaded merged data with shape: {df_data.shape}")
    
    X = df_data.drop(columns=['filename', 'damage'])
    y = df_data['damage'].values
    feature_names = list(X.columns)
    
    # Log transformation on target to directly optimize MAPE (relative error)
    y_log = np.log1p(y)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    models_to_test = {
        'LGBM_LogTarget': lambda: lgb.LGBMRegressor(
            objective='mape',
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=15,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=-1,
            n_jobs=1
        ),
        'XGB_LogTarget': lambda: xgb.XGBRegressor(
            objective='reg:absoluteerror',
            n_estimators=300,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=1
        ),
        'ExtraTrees_LogTarget': lambda: ExtraTreesRegressor(
            n_estimators=300,
            max_depth=8,
            criterion='absolute_error',
            random_state=42,
            n_jobs=1
        )
    }
    
    best_overall_score = -1.0
    best_model_name = None
    
    for mname, model_fn in models_to_test.items():
        oof_preds = np.zeros(len(df_data))
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_log)):
            X_train, y_train_log = X.iloc[train_idx], y_log[train_idx]
            X_val, y_val_log = X.iloc[val_idx], y_log[val_idx]
            
            model = model_fn()
            model.fit(X_train, y_train_log)
            
            preds_log = model.predict(X_val)
            preds = np.expm1(preds_log)
            preds = np.clip(preds, a_min=1e-6, a_max=None)
            oof_preds[val_idx] = preds
            
        overall_score, overall_mape = compute_metrics(y, oof_preds)
        print(f"Model: {mname:20s} | OOF MAPE: {overall_mape*100:6.2f}% | SHM Score: {overall_score:.4f}")
        
        if overall_score > best_overall_score:
            best_overall_score = overall_score
            best_model_name = mname
            
    print(f"\n---> Best Model: {best_model_name} with SHM Score: {best_overall_score:.4f}\n")
    
    # Retrain best model on 100% of data
    final_model = models_to_test[best_model_name]()
    final_model.fit(X, y_log) # Train on full log-target
    
    os.makedirs("models", exist_ok=True)
    model_payload = {
        'model': final_model,
        'feature_names': feature_names,
        'use_log_target': True
    }
    joblib.dump(model_payload, "models/shm_model.pkl")
    print(f"Saved final trained model ({best_model_name}) to models/shm_model.pkl")

if __name__ == '__main__':
    train_and_evaluate()

