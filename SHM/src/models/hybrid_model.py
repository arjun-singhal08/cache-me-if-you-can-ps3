"""
Hybrid Physics + ML Models for SHM Fatigue Damage Prediction
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.optimize import minimize
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import joblib
import warnings
warnings.filterwarnings("ignore")


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.abs(y_true) > 1e-10
    if not np.any(mask):
        return np.mean(np.abs(y_true - y_pred))
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / np.abs(y_true[mask]))


def identify_physics_features(feature_cols: List[str]) -> List[int]:
    physics_prefixes = [
        "miner_", "rayleigh_", "dirlik_", "tovo_", "swt_",
        "goodman", "gerber", "morrow",
        "rf_range", "rf_mean", "n_cycles",
        "m0", "m1", "m2", "m4",
        "sigma_uts"
    ]
    physics_indices = []
    for i, col in enumerate(feature_cols):
        if any(col.startswith(p) for p in physics_prefixes):
            physics_indices.append(i)
    return physics_indices


class PhysicsRidgeRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, alpha: float = 1.0, use_bayesian: bool = True):
        self.alpha = alpha
        self.use_bayesian = use_bayesian
        self.physics_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.physics_indices = None
        self.model = None
        self.feature_cols = None
        self.feature_shifts = None
        self._global_min = None

    def fit(self, X: np.ndarray, y: np.ndarray, feature_cols: List[str], physics_indices: List[int],
            global_shifts: np.ndarray = None, global_min: np.ndarray = None):
        self.physics_indices = physics_indices
        self.feature_cols = feature_cols

        X_phys = X[:, physics_indices]
        eps = 1e-10

        if global_shifts is not None and global_min is not None:
            self.feature_shifts = global_shifts
            self._global_min = global_min
        else:
            self._global_min = X_phys.min(axis=0)
            self.feature_shifts = np.zeros(X_phys.shape[1])
            for i in range(X_phys.shape[1]):
                if self._global_min[i] <= 0:
                    self.feature_shifts[i] = abs(self._global_min[i]) + eps

        X_phys = X[:, physics_indices]
        X_phys_shifted = X_phys.copy()
        for i in range(X_phys.shape[1]):
            if self.feature_shifts[i] > 0:
                X_phys_shifted[:, i] = X_phys[:, i] + self.feature_shifts[i]
            min_after_shift = X_phys_shifted[:, i].min()
            if min_after_shift <= 0:
                additional_shift = abs(min_after_shift) + 1e-10
                self.feature_shifts[i] += additional_shift
                X_phys_shifted[:, i] += additional_shift

        X_phys_log = np.log(X_phys_shifted + 1e-10)
        y_log = np.log(y + 1e-10)

        X_phys_scaled = self.physics_scaler.fit_transform(X_phys_log)
        y_scaled = self.target_scaler.fit_transform(y_log.reshape(-1, 1)).flatten()

        if self.use_bayesian:
            self.model = BayesianRidge(alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6)
        else:
            self.model = Ridge(alpha=self.alpha)

        self.model.fit(X_phys_scaled, y_scaled)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        eps = 1e-10
        X_phys = X[:, self.physics_indices]
        X_phys_shifted = X_phys.copy()
        for i in range(X_phys.shape[1]):
            if self.feature_shifts[i] > 0:
                X_phys_shifted[:, i] = X_phys[:, i] + self.feature_shifts[i]
            min_after_shift = X_phys_shifted[:, i].min()
            if min_after_shift <= 0:
                additional_shift = abs(min_after_shift) + 1e-10
                X_phys_shifted[:, i] += additional_shift

        X_phys_log = np.log(X_phys_shifted + 1e-10)
        X_phys_scaled = self.physics_scaler.transform(X_phys_log)
        y_pred_scaled = self.model.predict(X_phys_scaled)
        y_pred_log = self.target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        return np.exp(y_pred_log) - 1e-10


class HybridEnsembleRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, physics_alpha: float = 1.0, use_bayesian: bool = True,
                 xgb_params: dict = None, lgb_params: dict = None, cat_params: dict = None,
                 n_tta: int = 20, noise_std: float = 0.005):
        self.physics_alpha = physics_alpha
        self.use_bayesian = use_bayesian
        self.xgb_params = xgb_params or {}
        self.lgb_params = lgb_params or {}
        self.cat_params = cat_params or {}
        self.n_tta = n_tta
        self.noise_std = noise_std
        self.physics_model = None
        self.tree_models = {}
        self.tree_calibrators = {}
        self.blend_weight = 0.5
        self.physics_indices = None
        self.feature_cols = None
        self.global_calibrator = None

    def fit(self, X: np.ndarray, y: np.ndarray, feature_cols: List[str], groups: np.ndarray,
            n_folds: int = 5, optimize_blend: bool = True):
        self.feature_cols = feature_cols
        self.physics_indices = identify_physics_features(feature_cols)

        if len(self.physics_indices) == 0:
            raise ValueError("No physics features identified!")

        print(f"Physics features: {len(self.physics_indices)} / {len(feature_cols)}")

        full_physics_model = PhysicsRidgeRegressor(
            alpha=self.physics_alpha, use_bayesian=self.use_bayesian
        )
        full_physics_model.fit(X, y, feature_cols, self.physics_indices)
        global_shifts = full_physics_model.feature_shifts
        global_min = full_physics_model._global_min

        self.physics_model = PhysicsRidgeRegressor(
            alpha=self.physics_alpha, use_bayesian=self.use_bayesian
        )
        self.physics_model.fit(X, y, feature_cols, self.physics_indices)

        print("\nTraining Tree Ensembles (XGB, LGB, CatBoost)...")

        eps = 1e-6
        y_log = np.log(y + eps)
        gkf = GroupKFold(n_splits=5)

        default_xgb = {
            "objective": "reg:squarederror", "eval_metric": "mae",
            "tree_method": "hist", "max_depth": 5, "learning_rate": 0.03,
            "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3,
            "reg_lambda": 1.0, "reg_alpha": 0.1, "random_state": 42,
            "n_estimators": 3000, "early_stopping_rounds": 100, "verbosity": 0,
        }

        default_lgb = {
            "objective": "quantile", "alpha": 0.5, "metric": "mape",
            "boosting_type": "gbdt", "device": "cpu", "max_depth": 5,
            "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8,
            "min_child_samples": 10, "reg_lambda": 1.0, "reg_alpha": 0.1,
            "random_state": 42, "n_estimators": 3000, "verbose": -1,
        }

        default_cat = {
            "loss_function": "MAE", "eval_metric": "MAPE", "task_type": "CPU",
            "depth": 5, "learning_rate": 0.03, "subsample": 0.8,
            "colsample_bylevel": 0.8, "min_data_in_leaf": 10, "l2_leaf_reg": 1.0,
            "random_seed": 42, "iterations": 3000, "early_stopping_rounds": 100, "verbose": False,
        }

        oof_physics = np.zeros(len(y))
        oof_xgb = np.zeros(len(y))
        oof_lgb = np.zeros(len(y))
        oof_cat = np.zeros(len(y))
        oof_targets = np.zeros(len(y))

        xgb_models = []
        lgb_models = []
        cat_models = []

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y_log, groups)):
            print(f"  Fold {fold + 1}/5: Train={len(train_idx)}, Val={len(val_idx)}")

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y_log[train_idx], y_log[val_idx]
            y_train_true, y_val_true = np.exp(y_train) - eps, np.exp(y_val) - eps

            phys_fold = PhysicsRidgeRegressor(alpha=1.0, use_bayesian=True)
            phys_fold.fit(X_train, np.exp(y_train) - 1e-6, feature_cols, self.physics_indices, global_shifts=global_shifts, global_min=global_min)
            oof_physics[val_idx] = phys_fold.predict(X_val)

            xgb_model = xgb.XGBRegressor(**default_xgb)
            xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            xgb_pred_log = xgb_model.predict(X_val)
            oof_xgb[val_idx] = np.exp(xgb_pred_log) - 1e-6
            xgb_models.append(xgb_model)

            lgb_train = lgb.Dataset(X_train, label=y_train)
            lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
            lgb_model = lgb.train(
                default_lgb, lgb_train, valid_sets=[lgb_val],
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
            )
            lgb_pred_log = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
            oof_lgb[val_idx] = np.exp(lgb_pred_log) - 1e-6
            lgb_models.append(lgb_model)

            cat_model = CatBoostRegressor(**default_cat)
            cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
            cat_pred_log = cat_model.predict(X_val)
            oof_cat[val_idx] = np.exp(cat_pred_log) - 1e-6
            cat_models.append(cat_model)

            oof_targets[val_idx] = y_val_true

        self.tree_models = {"xgb": xgb_models, "lgb": lgb_models, "cat": cat_models}

        print("Calibrating individual models...")
        self.tree_calibrators = {}
        for name, oof in [("physics", oof_physics), ("xgb", oof_xgb), ("lgb", oof_lgb), ("cat", oof_cat)]:
            cal = IsotonicRegression(out_of_bounds="clip")
            cal.fit(oof, oof_targets)
            self.tree_calibrators[name] = cal
            cal_pred = cal.predict(oof)
            mape = compute_mape(oof_targets, cal_pred)
            print(f"  {name}: Calibrated MAPE = {mape:.4f} (Score = {max(0, 1-mape):.4f})")

        physics_cal = self.tree_calibrators["physics"].predict(oof_physics)
        tree_cal = np.median([
            self.tree_calibrators["xgb"].predict(oof_xgb),
            self.tree_calibrators["lgb"].predict(oof_lgb),
            self.tree_calibrators["cat"].predict(oof_cat)
        ], axis=0)

        def blend_objective(w):
            w = np.clip(w, 0, 1)
            blended = w * physics_cal + (1 - w) * tree_cal
            return compute_mape(oof_targets, blended)

        result = minimize(blend_objective, x0=[0.5], bounds=[(0, 1)], method="L-BFGS-B")
        self.blend_weight = float(np.clip(result.x[0], 0, 1))
        print(f"  Optimal blend weight (physics): {self.blend_weight:.4f}")
        print(f"  Blended MAPE: {result.fun:.4f} (Score: {max(0, 1-result.fun):.4f})")

        physics_oof_cal = self.tree_calibrators["physics"].predict(oof_physics)
        tree_oof_cal = np.median([
            self.tree_calibrators["xgb"].predict(oof_xgb),
            self.tree_calibrators["lgb"].predict(oof_lgb),
            self.tree_calibrators["cat"].predict(oof_cat)
        ], axis=0)
        final_oof = self.blend_weight * physics_cal + (1 - self.blend_weight) * tree_oof_cal

        self.global_calibrator = IsotonicRegression(out_of_bounds="clip")
        self.global_calibrator.fit(final_oof, oof_targets)
        final_calibrated = self.global_calibrator.predict(final_oof)
        final_mape = compute_mape(oof_targets, final_calibrated)
        print(f"\nFinal OOF MAPE: {final_mape:.4f} (Score: {max(0, 1-final_mape):.4f})")

        self.tree_models = {"xgb": xgb_models, "lgb": lgb_models, "cat": cat_models}
        return self

    def _predict_trees_tta(self, X_test: np.ndarray) -> dict:
        eps = 1e-6
        n_samples = X_test.shape[0]
        preds = {}
        for name, models in self.tree_models.items():
            all_preds = []
            for model in models:
                model_preds = np.zeros((self.n_tta, n_samples))
                for t in range(self.n_tta):
                    X_aug = X_test if t == 0 else X_test + np.random.normal(0, self.noise_std, X_test.shape).astype(np.float32)
                    if name == "lgb" and hasattr(model, "best_iteration") and model.best_iteration:
                        pred_log = model.predict(X_aug, num_iteration=model.best_iteration)
                    else:
                        pred_log = model.predict(X_aug)
                    if pred_log.ndim > 1:
                        pred_log = pred_log.flatten()
                    model_preds[t] = np.exp(pred_log) - 1e-6
                all_preds.append(np.median(model_preds, axis=0))
            preds[name] = np.median(all_preds, axis=0)
        return preds

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        physics_pred = self.physics_model.predict(X_test)
        physics_pred_cal = self.tree_calibrators["physics"].predict(physics_pred)

        tree_preds = self._predict_trees_tta(X_test)
        tree_cal = {}
        for name, pred in tree_preds.items():
            tree_cal[name] = self.tree_calibrators[name].predict(pred)

        tree_ensemble = np.median([tree_cal["xgb"], tree_cal["lgb"], tree_cal["cat"]], axis=0)

        blended = self.blend_weight * physics_pred_cal + (1 - self.blend_weight) * tree_ensemble
        final_pred = self.global_calibrator.predict(blended)
        return final_pred

    def save(self, path: str):
        import os
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.physics_model, f"{path}/physics_model.pkl")
        joblib.dump(self.tree_models, f"{path}/tree_models.pkl")
        joblib.dump(self.tree_calibrators, f"{path}/tree_calibrators.pkl")
        joblib.dump(self.global_calibrator, f"{path}/global_calibrator.pkl")
        joblib.dump(self.physics_indices, f"{path}/physics_indices.pkl")
        joblib.dump(self.feature_cols, f"{path}/feature_cols.pkl")
        joblib.dump(self.blend_weight, f"{path}/blend_weight.pkl")

    @classmethod
    def load(cls, path: str):
        instance = cls()
        instance.physics_model = joblib.load(f"{path}/physics_model.pkl")
        instance.tree_models = joblib.load(f"{path}/tree_models.pkl")
        instance.tree_calibrators = joblib.load(f"{path}/tree_calibrators.pkl")
        instance.global_calibrator = joblib.load(f"{path}/global_calibrator.pkl")
        instance.physics_indices = joblib.load(f"{path}/physics_indices.pkl")
        instance.feature_cols = joblib.load(f"{path}/feature_cols.pkl")
        instance.blend_weight = joblib.load(f"{path}/blend_weight.pkl")
        return instance


def train_hybrid_pipeline(X: np.ndarray, y: np.ndarray, feature_cols: List[str], groups: np.ndarray,
                          n_folds: int = 5, output_dir: str = "models_hybrid") -> HybridEnsembleRegressor:
    model = HybridEnsembleRegressor(physics_alpha=1.0, use_bayesian=True, n_tta=20, noise_std=0.005)
    model.fit(X, y, feature_cols, groups, n_folds=n_folds, optimize_blend=True)
    model.save(output_dir)
    return model