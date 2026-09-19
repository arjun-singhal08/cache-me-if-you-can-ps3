import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.isotonic import IsotonicRegression
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
import joblib
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Any
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


def infer_groups_by_damage_quantiles(y: np.ndarray, n_groups: int = 5) -> np.ndarray:
    """Create groups based on damage quantiles for stratified GroupKFold."""
    try:
        groups = pd.qcut(y, q=n_groups, labels=False, duplicates='drop').values
    except:
        groups = pd.qcut(pd.Series(y).rank(method='first'), q=n_groups, labels=False).values
    return groups.astype(int)


def infer_groups_by_rms_quantiles(df: pd.DataFrame, feature_cols: List[str], n_groups: int = 5) -> np.ndarray:
    """Create groups based on RMS quantiles."""
    if 'rms' in feature_cols:
        rms_idx = feature_cols.index('rms')
        rms_vals = df.iloc[:, rms_idx].values
    elif 'std' in feature_cols:
        std_idx = feature_cols.index('std')
        rms_vals = df.iloc[:, std_idx].values
    else:
        return np.arange(len(df)) % n_groups
    
    rms_series = pd.Series(rms_vals)
    try:
        groups = pd.qcut(rms_series, q=n_groups, labels=False, duplicates='drop').values
    except:
        groups = pd.qcut(rms_series.rank(method='first'), q=n_groups, labels=False).values
    return groups.astype(int)


def get_xgb_params(random_state: int = 42, use_gpu: bool = False) -> Dict:
    """Get XGBoost parameters."""
    return {
        'objective': 'reg:squarederror',
        'eval_metric': 'mae',
        'tree_method': 'gpu_hist' if use_gpu else 'hist',
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


def get_lgb_params(random_state: int = 42, use_gpu: bool = False) -> Dict:
    """Get LightGBM parameters for quantile regression."""
    return {
        'objective': 'quantile',
        'alpha': 0.5,  # median
        'metric': 'mape',
        'boosting_type': 'gbdt',
        'device': 'gpu' if use_gpu else 'cpu',
        'max_depth': 5,
        'learning_rate': 0.03,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_samples': 10,
        'reg_lambda': 1.0,
        'reg_alpha': 0.1,
        'random_state': random_state,
        'n_estimators': 3000,
        'verbose': -1,
    }


def get_lgb_quantile_params(alpha: float, random_state: int = 42, use_gpu: bool = False) -> Dict:
    """Get LightGBM parameters for specific quantile."""
    params = get_lgb_params(random_state, use_gpu)
    params['objective'] = 'quantile'
    params['alpha'] = alpha
    return params


def get_catboost_params(random_state: int = 42, use_gpu: bool = False) -> Dict:
    """Get CatBoost parameters."""
    return {
        'loss_function': 'MAE',
        'eval_metric': 'MAPE',
        'task_type': 'GPU' if use_gpu else 'CPU',
        'depth': 5,
        'learning_rate': 0.03,
        'subsample': 0.8,
        'colsample_bylevel': 0.8,
        'min_data_in_leaf': 10,
        'l2_leaf_reg': 1.0,
        'random_seed': random_state,
        'iterations': 3000,
        'early_stopping_rounds': 100,
        'verbose': False,
    }


def train_model_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_type: str = 'xgb',
    n_folds: int = 5,
    params: Optional[Dict] = None,
    random_state: int = 42,
    use_gpu: bool = False,
    quantile_alpha: float = 0.5
) -> Tuple[List, np.ndarray, np.ndarray]:
    """
    Train a model with GroupKFold CV.
    Returns: (models, oof_predictions, oof_targets)
    """
    eps = 1e-6
    y_log = np.log(y + eps)
    
    gkf = GroupKFold(n_splits=n_folds)
    
    models = []
    oof_preds_log = np.zeros(len(y))
    oof_targets_log = np.zeros(len(y))
    
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y_log, groups)):
        print(f"  Fold {fold + 1}/{n_folds}: Train={len(train_idx)}, Val={len(val_idx)}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_log[train_idx], y_log[val_idx]
        
        if model_type == 'xgb':
            if params is None:
                params = get_xgb_params(random_state + fold, use_gpu)
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            val_pred_log = model.predict(X_val)
            
        elif model_type == 'lgb':
            if params is None:
                params = get_lgb_quantile_params(quantile_alpha, random_state + fold, use_gpu)
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
            )
            val_pred_log = model.predict(X_val, num_iteration=model.best_iteration)
            
        elif model_type == 'catboost':
            if params is None:
                params = get_catboost_params(random_state + fold, use_gpu)
            model = CatBoostRegressor(**params)
            model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
            val_pred_log = model.predict(X_val)
            
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        val_pred = np.exp(val_pred_log) - eps
        y_val_true = np.exp(y_val) - eps
        
        mape = compute_mape(y_val_true, val_pred)
        fold_scores.append(mape)
        print(f"    Fold MAPE: {mape:.4f} (Score: {max(0, 1-mape):.4f})")
        
        oof_preds_log[val_idx] = val_pred_log
        oof_targets_log[val_idx] = y_val
        models.append(model)
    
    oof_preds = np.exp(oof_preds_log) - eps
    oof_targets = np.exp(oof_targets_log) - eps
    oof_mape = compute_mape(oof_targets, oof_preds)
    print(f"  Overall OOF MAPE: {oof_mape:.4f} (Score: {max(0, 1-oof_mape):.4f})")
    
    return models, oof_preds, oof_targets


def train_quantile_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_folds: int = 5,
    quantiles: List[float] = [0.1, 0.5, 0.9],
    use_gpu: bool = False
) -> Tuple[Dict[str, List], np.ndarray]:
    """
    Train LightGBM quantile regression ensemble.
    Returns: (models_dict, median_oof_predictions)
    """
    eps = 1e-6
    y_log = np.log(y + eps)
    
    gkf = GroupKFold(n_splits=n_folds)
    models_dict = {q: [] for q in quantiles}
    oof_preds_dict = {q: np.zeros(len(y)) for q in quantiles}
    oof_targets_log = np.zeros(len(y))
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y_log, groups)):
        print(f"  Quantile Fold {fold + 1}/{n_folds}")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_log[train_idx], y_log[val_idx]
        
        for q in quantiles:
            params = get_lgb_quantile_params(q, fold, use_gpu)
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
            )
            val_pred_log = model.predict(X_val, num_iteration=model.best_iteration)
            oof_preds_dict[q][val_idx] = val_pred_log
            models_dict[q].append(model)
        
        oof_targets_log[val_idx] = y_val
    
    # Median of quantile predictions (in original space)
    oof_targets = np.exp(oof_targets_log) - eps
    median_oof = np.median([np.exp(oof_preds_dict[q]) - eps for q in quantiles], axis=0)
    
    mape = compute_mape(oof_targets, median_oof)
    print(f"  Quantile Ensemble OOF MAPE: {mape:.4f} (Score: {max(0, 1-mape):.4f})")
    
    return models_dict, median_oof


def train_stacking_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_folds: int = 5,
    use_gpu: bool = False
) -> Tuple[Dict, np.ndarray, IsotonicRegression]:
    """
    Train stacking ensemble: XGB + LGB (median) + CatBoost -> Ridge meta-learner.
    """
    print("\n=== Training Base Models for Stacking ===")
    
    # Train base models
    xgb_models, xgb_oof, _ = train_model_cv(X, y, groups, 'xgb', n_folds, use_gpu=use_gpu)
    lgb_models_dict, lgb_oof = train_quantile_ensemble(X, y, groups, n_folds, [0.1, 0.5, 0.9], use_gpu)
    cat_models, cat_oof, _ = train_model_cv(X, y, groups, 'catboost', n_folds, use_gpu=use_gpu)
    
    # Stack OOF predictions
    eps = 1e-6
    y_true = y
    X_meta = np.column_stack([xgb_oof, lgb_oof, cat_oof])
    
    # Train meta-learner with CV
    gkf = GroupKFold(n_splits=n_folds)
    meta_oof = np.zeros(len(y))
    meta_models = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_meta, y_true, groups)):
        X_meta_train, X_meta_val = X_meta[train_idx], X_meta[val_idx]
        y_meta_train, y_meta_val = y_true[train_idx], y_true[val_idx]
        
        meta_model = Ridge(alpha=1.0, random_state=42)
        meta_model.fit(X_meta_train, y_meta_train)
        meta_oof[val_idx] = meta_model.predict(X_meta_val)
        meta_models.append(meta_model)
    
    meta_mape = compute_mape(y_true, meta_oof)
    print(f"\n  Stacking OOF MAPE: {meta_mape:.4f} (Score: {max(0, 1-meta_mape):.4f})")
    
    # Fit final meta-learner on all data
    final_meta = Ridge(alpha=1.0, random_state=42)
    final_meta.fit(X_meta, y_true)
    
    # Isotonic calibration on stacked predictions
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(meta_oof, y_true)
    calibrated_oof = calibrator.predict(meta_oof)
    cal_mape = compute_mape(y_true, calibrated_oof)
    print(f"  Calibrated Stacking MAPE: {cal_mape:.4f} (Score: {max(0, 1-cal_mape):.4f})")
    
    models_dict = {
        'xgb': xgb_models,
        'lgb': lgb_models_dict,
        'catboost': cat_models,
        'meta': meta_models,
        'final_meta': final_meta,
    }
    
    return models_dict, calibrated_oof, calibrator


def select_features_permutation(
    X: np.ndarray,
    y: np.ndarray,
    feature_cols: List[str],
    model,
    n_repeats: int = 10,
    threshold: float = 0.001
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Select features using permutation importance."""
    print("\n=== Feature Selection (Permutation Importance) ===")
    result = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=42, n_jobs=-1)
    
    importances = result.importances_mean
    selected_idx = importances > threshold
    
    print(f"Selected {selected_idx.sum()}/{len(feature_cols)} features (threshold={threshold})")
    for idx in np.argsort(importances)[::-1][:15]:
        print(f"  {feature_cols[idx]}: {importances[idx]:.6f}")
    
    return X[:, selected_idx], [feature_cols[i] for i in np.where(selected_idx)[0]], importances


def calibrate_predictions(oof_preds: np.ndarray, oof_targets: np.ndarray) -> Tuple[IsotonicRegression, np.ndarray]:
    """Fit isotonic calibration and return calibrated predictions."""
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(oof_preds, oof_targets)
    calibrated = calibrator.predict(oof_preds)
    return calibrator, calibrated


def predict_ensemble_tta(
    models_dict: Dict,
    X_test: np.ndarray,
    calibrator: IsotonicRegression,
    n_tta: int = 10,
    noise_std: float = 0.005,
    use_stacking: bool = True
) -> np.ndarray:
    """
    Predict with ensemble + TTA.
    """
    eps = 1e-6
    n_samples = X_test.shape[0]
    
    if use_stacking and 'xgb' in models_dict:
        # Stacking prediction
        all_base_preds = []
        
        # XGB predictions
        xgb_preds = np.zeros((len(models_dict['xgb']), n_tta, n_samples))
        for i, model in enumerate(models_dict['xgb']):
            for t in range(n_tta):
                X_aug = X_test if t == 0 else X_test + np.random.normal(0, noise_std, X_test.shape).astype(np.float32)
                pred_log = model.predict(X_aug)
                xgb_preds[i, t] = np.exp(pred_log) - eps
        xgb_median = np.median(xgb_preds, axis=(0, 1))
        all_base_preds.append(xgb_median)
        
        # LGB quantile predictions (use median quantile models)
        lgb_preds = np.zeros((len(models_dict['lgb'][0.5]), n_tta, n_samples))
        for i, model in enumerate(models_dict['lgb'][0.5]):
            for t in range(n_tta):
                X_aug = X_test if t == 0 else X_test + np.random.normal(0, noise_std, X_test.shape).astype(np.float32)
                pred_log = model.predict(X_aug, num_iteration=model.best_iteration)
                lgb_preds[i, t] = np.exp(pred_log) - eps
        lgb_median = np.median(lgb_preds, axis=(0, 1))
        all_base_preds.append(lgb_median)
        
        # CatBoost predictions
        cat_preds = np.zeros((len(models_dict['catboost']), n_tta, n_samples))
        for i, model in enumerate(models_dict['catboost']):
            for t in range(n_tta):
                X_aug = X_test if t == 0 else X_test + np.random.normal(0, noise_std, X_test.shape).astype(np.float32)
                pred_log = model.predict(X_aug)
                cat_preds[i, t] = np.exp(pred_log) - eps
        cat_median = np.median(cat_preds, axis=(0, 1))
        all_base_preds.append(cat_median)
        
        # Stack
        X_meta = np.column_stack(all_base_preds)
        final_preds = models_dict['final_meta'].predict(X_meta)
        final_preds = calibrator.predict(final_preds)
        
    else:
        # Simple median ensemble
        all_preds = []
        for model_list in models_dict.values():
            if isinstance(model_list, list):
                for model in model_list:
                    preds = np.zeros((n_tta, n_samples))
                    for t in range(n_tta):
                        X_aug = X_test if t == 0 else X_test + np.random.normal(0, noise_std, X_test.shape).astype(np.float32)
                        if hasattr(model, 'best_iteration'):
                            pred_log = model.predict(X_aug, num_iteration=model.best_iteration)
                        else:
                            pred_log = model.predict(X_aug)
                        preds[t] = np.exp(pred_log) - eps
                    all_preds.append(np.median(preds, axis=0))
        final_preds = np.median(all_preds, axis=0)
        final_preds = calibrator.predict(final_preds)
    
    return final_preds


def save_artifacts_v2(models_dict: Dict, calibrator: IsotonicRegression, feature_cols: List[str], output_dir: str):
    """Save all model artifacts."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for key, val in models_dict.items():
        if key == 'lgb':
            for q, model_list in val.items():
                for i, model in enumerate(model_list):
                    joblib.dump(model, f"{output_dir}/lgb_q{q}_fold_{i}.pkl")
        elif isinstance(val, list):
            for i, model in enumerate(val):
                joblib.dump(model, f"{output_dir}/{key}_fold_{i}.pkl")
        else:
            joblib.dump(val, f"{output_dir}/{key}.pkl")
    
    save_model(calibrator, f"{output_dir}/calibrator.pkl")
    joblib.dump(feature_cols, f"{output_dir}/feature_cols.pkl")
    print(f"Saved artifacts to {output_dir}")


def load_artifacts_v2(model_dir: str, n_folds: int = 5) -> Tuple[Dict, IsotonicRegression, List[str]]:
    """Load all model artifacts."""
    models_dict = {}
    
    # XGB
    models_dict['xgb'] = [load_model(f"{model_dir}/xgb_fold_{i}.pkl") for i in range(n_folds)]
    
    # LGB quantiles
    models_dict['lgb'] = {}
    for q in [0.1, 0.5, 0.9]:
        models_dict['lgb'][q] = [load_model(f"{model_dir}/lgb_q{q}_fold_{i}.pkl") for i in range(n_folds)]
    
    # CatBoost
    models_dict['catboost'] = [load_model(f"{model_dir}/catboost_fold_{i}.pkl") for i in range(n_folds)]
    
    # Meta
    models_dict['meta'] = [load_model(f"{model_dir}/meta_fold_{i}.pkl") for i in range(n_folds)]
    models_dict['final_meta'] = load_model(f"{model_dir}/final_meta.pkl")
    
    calibrator = load_model(f"{model_dir}/calibrator.pkl")
    feature_cols = joblib.load(f"{model_dir}/feature_cols.pkl")
    
    return models_dict, calibrator, feature_cols