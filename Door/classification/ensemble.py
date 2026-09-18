# classification/ensemble.py
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from scipy.stats import rankdata
import joblib

class DoorFaultEnsemble:
    """Stacked ensemble for door fault classification with door-adaptive calibration."""
    
    def __init__(self, config: dict):
        self.config = config
        self.scaler = RobustScaler()
        self.base_models = {}
        self.meta_model = None
        self.calibrators = {}  # per-door calibrators
        self.feature_names = None
        self.is_fitted = False
        
    def build_base_models(self):
        """Initialize base learners."""
        xgb_cfg = self.config.get('classification', {}).get('xgb', {})
        lgbm_cfg = self.config.get('classification', {}).get('lgbm', {})
        
        self.base_models = {
            'xgb': XGBClassifier(
                n_estimators=xgb_cfg.get('n_estimators', 500),
                max_depth=xgb_cfg.get('max_depth', 6),
                learning_rate=xgb_cfg.get('learning_rate', 0.05),
                scale_pos_weight=xgb_cfg.get('scale_pos_weight', 2.67),
                subsample=xgb_cfg.get('subsample', 0.8),
                colsample_bytree=xgb_cfg.get('colsample_bytree', 0.8),
                eval_metric=xgb_cfg.get('eval_metric', 'auc'),
                random_state=xgb_cfg.get('random_state', 42),
                n_jobs=-1,
                verbosity=0
            ),
            'lgbm': LGBMClassifier(
                n_estimators=lgbm_cfg.get('n_estimators', 500),
                max_depth=lgbm_cfg.get('max_depth', 6),
                learning_rate=lgbm_cfg.get('learning_rate', 0.05),
                class_weight=lgbm_cfg.get('class_weight', 'balanced'),
                subsample=lgbm_cfg.get('subsample', 0.8),
                colsample_bytree=lgbm_cfg.get('colsample_bytree', 0.8),
                random_state=lgbm_cfg.get('random_state', 42),
                n_jobs=-1,
                verbosity=-1
            )
        }
        
        # Meta-learner
        meta_cfg = self.config.get('classification', {}).get('meta', {})
        self.meta_model = LogisticRegression(
            C=meta_cfg.get('C', 1.0),
            random_state=meta_cfg.get('random_state', 42),
            max_iter=1000,
            class_weight='balanced'
        )
    
    def _get_door_id(self, seg_df: pd.DataFrame) -> str:
        """Extract door identifier from sensor pattern."""
        # DCSR, DCSL, DLSR, DLSL pattern
        dcsr = seg_df['DCSR'].mode()[0] if len(seg_df) > 0 else 0
        dcsl = seg_df['DCSL'].mode()[0] if len(seg_df) > 0 else 0
        dlsr = seg_df['DLSR'].mode()[0] if len(seg_df) > 0 else 0
        dlsl = seg_df['DLSL'].mode()[0] if len(seg_df) > 0 else 0
        return f"{dcsr}{dcsl}{dlsr}{dlsl}"
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            seg_dfs: Optional[List[pd.DataFrame]] = None,
            groups: Optional[np.ndarray] = None):
        """
        Fit the stacked ensemble.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (0=Normal, 1=Abnormal)
            seg_dfs: Optional list of segment DataFrames for door adaptation
            groups: Optional group labels for GroupKFold
        """
        from sklearn.model_selection import GroupKFold, StratifiedKFold
        
        self.feature_names = [f'feat_{i}' for i in range(X.shape[1])]
        self.build_base_models()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Determine CV strategy
        n_splits = 5
        if groups is not None:
            n_groups = len(np.unique(groups))
            if n_groups >= n_splits:
                cv = GroupKFold(n_splits=n_splits)
                cv_iter = cv.split(X_scaled, y, groups)
            else:
                # Not enough groups for GroupKFold, use StratifiedKFold
                cv = StratifiedKFold(n_splits=min(n_splits, n_groups), shuffle=True, random_state=42)
                cv_iter = cv.split(X_scaled, y)
        else:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            cv_iter = cv.split(X_scaled, y)
        
        # Generate meta-features via cross-validation
        n_samples = len(y)
        meta_features = np.zeros((n_samples, len(self.base_models)))
        
        for fold, (train_idx, val_idx) in enumerate(cv_iter):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            for i, (name, model) in enumerate(self.base_models.items()):
                model.fit(X_train, y_train)
                meta_features[val_idx, i] = model.predict_proba(X_val)[:, 1]
        
        # Train base models on full data
        for name, model in self.base_models.items():
            model.fit(X_scaled, y)
        
        # Train meta-learner
        self.meta_model.fit(meta_features, y)
        
        # Fit per-door calibrators if seg_dfs provided
        if seg_dfs is not None:
            self._fit_door_calibrators(meta_features, y, seg_dfs)
        else:
            # Global isotonic calibration
            self.calibrators['global'] = IsotonicRegression(out_of_bounds='clip')
            self.calibrators['global'].fit(meta_features[:, 0], y)  # Use first base model as reference
        
        self.is_fitted = True
    
    def _fit_door_calibrators(self, meta_features: np.ndarray, y: np.ndarray, seg_dfs: List[pd.DataFrame]):
        """Fit isotonic calibration per door."""
        door_ids = [self._get_door_id(df) for df in seg_dfs]
        unique_doors = np.unique(door_ids)
        
        # Meta predictions (use mean of base models)
        meta_preds = meta_features.mean(axis=1)
        
        for door in unique_doors:
            mask = np.array(door_ids) == door
            if mask.sum() >= 5:  # Minimum samples for calibration
                cal = IsotonicRegression(out_of_bounds='clip')
                cal.fit(meta_preds[mask], y[mask])
                self.calibrators[door] = cal
            else:
                # Too few samples, use global
                pass
        
        # Always have a global fallback
        if 'global' not in self.calibrators:
            self.calibrators['global'] = IsotonicRegression(out_of_bounds='clip')
            self.calibrators['global'].fit(meta_preds, y)
    
    def predict_proba(self, X: np.ndarray, 
                      seg_dfs: Optional[List[pd.DataFrame]] = None) -> np.ndarray:
        """Predict probability of Abnormal resistance."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        X_scaled = self.scaler.transform(X)
        
        # Get base model predictions
        base_probas = np.column_stack([
            model.predict_proba(X_scaled)[:, 1] 
            for model in self.base_models.values()
        ])
        
        # Meta prediction
        meta_pred = self.meta_model.predict_proba(base_probas)[:, 1]
        
        # Apply calibration
        if seg_dfs is not None and self.calibrators:
            calibrated = np.zeros_like(meta_pred)
            door_ids = [self._get_door_id(df) for df in seg_dfs]
            unique_doors = np.unique(door_ids)
            
            for door in unique_doors:
                mask = np.array(door_ids) == door
                if door in self.calibrators:
                    calibrated[mask] = self.calibrators[door].predict(meta_pred[mask])
                elif 'global' in self.calibrators:
                    calibrated[mask] = self.calibrators['global'].predict(meta_pred[mask])
                else:
                    calibrated[mask] = meta_pred[mask]
            return calibrated
        elif 'global' in self.calibrators:
            return self.calibrators['global'].predict(meta_pred)
        else:
            return meta_pred
    
    def predict(self, X: np.ndarray, 
                seg_dfs: Optional[List[pd.DataFrame]] = None,
                threshold: float = 0.5) -> np.ndarray:
        """Predict class labels."""
        probas = self.predict_proba(X, seg_dfs)
        return (probas >= threshold).astype(int)
    
    def save(self, path: str):
        """Save model to disk."""
        joblib.dump({
            'scaler': self.scaler,
            'base_models': self.base_models,
            'meta_model': self.meta_model,
            'calibrators': self.calibrators,
            'feature_names': self.feature_names,
            'config': self.config,
            'is_fitted': self.is_fitted
        }, path)
    
    @classmethod
    def load(cls, path: str):
        """Load model from disk."""
        data = joblib.load(path)
        model = cls(data['config'])
        model.scaler = data['scaler']
        model.base_models = data['base_models']
        model.meta_model = data['meta_model']
        model.calibrators = data['calibrators']
        model.feature_names = data['feature_names']
        model.is_fitted = data['is_fitted']
        return model