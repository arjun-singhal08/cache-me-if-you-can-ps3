import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_percentage_error
import joblib
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

from ..features.extract import prepare_features
from ..utils.io import save_model, load_model


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute MAPE with protection against division by zero."""
    mask = np.abs(y_true) > 1e-10
    if not np.any(mask):
        return np.mean(np.abs(y_true - y_pred))
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / np.abs(y_true[mask]))


def infer_load_condition_groups(df: pd.DataFrame, feature_cols: List[str], n_groups: int = 5) -> np.ndarray:
    """
    Infer load condition groups from stress statistics for GroupKFold.
    Uses RMS to create n_groups quantile bins.
    """
    # Use RMS as proxy for load level
    if 'rms' in feature_cols:
        rms_idx = feature_cols.index('rms')
        rms_vals = df.iloc[:, rms_idx].values
    elif 'std' in feature_cols:
        std_idx = feature_cols.index('std')
        rms_vals = df.iloc[:, std_idx].values
    else:
        # Fallback: use file index
        return np.arange(len(df)) % n_groups
    
    # Quantile binning into n_groups
    rms_series = pd.Series(rms_vals)
    try:
        groups = pd.qcut(rms_series, q=n_groups, labels=False, duplicates='drop').values
    except:
        # If duplicates, use rank-based
        groups = pd.qcut(rms_series.rank(method='first'), q=n_groups, labels=False).values
    
    return groups.astype(int)


def train_xgboost_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_cols: List[str],
    n_folds: int = 5,
    params: Optional[Dict] = None,
    random_state: int = 42
) -> Tuple[List, np.ndarray, np.ndarray, IsotonicRegression]:
    """
    Train XGBoost with GroupKFold CV.
    Returns: (models, oof_predictions, oof_targets, calibrator)
    """
    if params is None:
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'mae',
            'tree_method': 'hist',  # CPU; use 'gpu_hist' if GPU available
            'max_depth': 5,
            'learning_rate': 0.03,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'reg_lambda': 1.0,
            'reg_alpha': 0.1,
            'random_state': random_state,
            'n_estimators': 3000,
            'early_stopping_rounds': 100,
            'verbosity': 0,
        }
    
    # Target transform: log(damage + eps)
    eps = 1e-6
    y_log = np.log(y + eps)
    
    # CV split
    gkf = GroupKFold(n_splits=n_folds)
    
    models = []
    oof_preds_log = np.zeros(len(y))
    oof_targets_log = np.zeros(len(y))
    
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y_log, groups)):
        print(f"\n=== Fold {fold + 1}/{n_folds} ===")
        print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_log[train_idx], y_log[val_idx]
        
        # Train
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Predict
        val_pred_log = model.predict(X_val)
        val_pred = np.exp(val_pred_log) - eps
        y_val_true = np.exp(y_val) - eps
        
        mape = compute_mape(y_val_true, val_pred)
        fold_scores.append(mape)
        print(f"  Fold MAPE: {mape:.4f} (Score: {max(0, 1-mape):.4f})")
        
        oof_preds_log[val_idx] = val_pred_log
        oof_targets_log[val_idx] = y_val
        models.append(model)
    
    # Overall OOF MAPE
    oof_preds = np.exp(oof_preds_log) - eps
    oof_targets = np.exp(oof_targets_log) - eps
    oof_mape = compute_mape(oof_targets, oof_preds)
    print(f"\n=== Overall OOF MAPE: {oof_mape:.4f} (Score: {max(0, 1-oof_mape):.4f}) ===")
    print(f"Fold MAPEs: {[f'{s:.4f}' for s in fold_scores]}")
    
    # Isotonic calibration on OOF predictions
    print("Fitting isotonic calibration...")
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(oof_preds, oof_targets)
    
    # Calibrated OOF
    oof_calibrated = calibrator.predict(oof_preds)
    cal_mape = compute_mape(oof_targets, oof_calibrated)
    print(f"Calibrated OOF MAPE: {cal_mape:.4f} (Score: {max(0, 1-cal_mape):.4f})")
    
    return models, oof_preds, oof_targets, calibrator


def train_final_model(
    X: np.ndarray,
    y: np.ndarray,
    params: Optional[Dict] = None,
    random_state: int = 42
) -> xgb.XGBRegressor:
    """Train final model on all data."""
    if params is None:
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'mae',
            'tree_method': 'hist',
            'max_depth': 5,
            'learning_rate': 0.03,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'reg_lambda': 1.0,
            'reg_alpha': 0.1,
            'random_state': random_state,
            'n_estimators': 3000,
            'verbosity': 0,
        }
    
    eps = 1e-6
    y_log = np.log(y + eps)
    
    model = xgb.XGBRegressor(**params)
    model.fit(X, y_log)
    return model


def predict_with_tta(
    models: List[xgb.XGBRegressor],
    X_test: np.ndarray,
    calibrator: IsotonicRegression,
    n_tta: int = 10,
    noise_std: float = 0.005
) -> np.ndarray:
    """
    Predict with Test-Time Augmentation (TTA).
    Adds Gaussian noise to features, averages predictions.
    """
    eps = 1e-6
    n_models = len(models)
    n_samples = X_test.shape[0]
    
    all_preds = np.zeros((n_models, n_tta, n_samples))
    
    for i, model in enumerate(models):
        for t in range(n_tta):
            if t == 0:
                X_aug = X_test
            else:
                # Add small Gaussian noise
                noise = np.random.normal(0, noise_std, X_test.shape).astype(np.float32)
                X_aug = X_test + noise
            
            pred_log = model.predict(X_aug)
            pred = np.exp(pred_log) - eps
            pred = calibrator.predict(pred)
            all_preds[i, t, :] = pred
    
    # Median across models and TTA iterations
    final_preds = np.median(all_preds, axis=(0, 1))
    return final_preds


def save_artifacts(
    models: List,
    calibrator: IsotonicRegression,
    feature_cols: List[str],
    output_dir: str
):
    """Save all model artifacts."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for i, model in enumerate(models):
        save_model(model, f"{output_dir}/model_fold_{i}.pkl")
    
    save_model(calibrator, f"{output_dir}/calibrator.pkl")
    joblib.dump(feature_cols, f"{output_dir}/feature_cols.pkl")
    print(f"Saved artifacts to {output_dir}")


def load_artifacts(model_dir: str, n_folds: int = 5) -> Tuple[List, IsotonicRegression, List[str]]:
    """Load all model artifacts."""
    models = []
    for i in range(n_folds):
        models.append(load_model(f"{model_dir}/model_fold_{i}.pkl"))
    calibrator = load_model(f"{model_dir}/calibrator.pkl")
    feature_cols = joblib.load(f"{model_dir}/feature_cols.pkl")
    return models, calibrator, feature_cols