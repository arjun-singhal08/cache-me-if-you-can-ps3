import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import ndcg_score
import lightgbm as lgb
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')


def rank_decay_score(y_true_rank: int, n: int = 8) -> float:
    """Compute linear rank-decay score: (n - (r - 1)) / n"""
    return (n - (y_true_rank - 1)) / n


def evaluate_ranking(y_true_binary: np.ndarray, scores: np.ndarray, groups: np.ndarray, n_cars: int = 8) -> Dict:
    """Evaluate ranking performance using rank-decay score per case."""
    from collections import defaultdict
    
    case_scores = defaultdict(list)
    case_labels = defaultdict(list)
    
    for idx, g in enumerate(groups):
        case_scores[g].append(scores[idx])
        case_labels[g].append(y_true_binary[idx])
    
    total_score = 0.0
    n_cases = len(case_scores)
    rank_details = []
    
    for case_idx in sorted(case_scores.keys()):
        case_s = np.array(case_scores[case_idx])
        case_y = np.array(case_labels[case_idx])
        
        # Rank cars by score (descending = higher score = more likely faulty)
        ranked_indices = np.argsort(-case_s)
        ranked_labels = case_y[ranked_indices]
        
        # Find rank of true faulty car (where label == 1)
        faulty_rank = np.where(ranked_labels == 1)[0]
        if len(faulty_rank) > 0:
            rank = faulty_rank[0] + 1  # 1-indexed
            score = rank_decay_score(rank, n_cars)
        else:
            rank = n_cars
            score = rank_decay_score(n_cars, n_cars)
        
        total_score += score
        rank_details.append({'case': case_idx, 'rank': rank, 'score': score})
    
    return {
        'mean_score': total_score / n_cases if n_cases > 0 else 0.0,
        'details': rank_details
    }


class LightGBMRanker:
    """LightGBM with LambdaRank objective for learning to rank."""
    
    def __init__(self, n_estimators=200, learning_rate=0.05, num_leaves=31, 
                 max_depth=-1, random_state=42, **kwargs):
        self.params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'num_leaves': num_leaves,
            'max_depth': max_depth,
            'random_state': random_state,
            'verbosity': -1,
            'force_col_wise': True,
            **kwargs
        }
        self.model = None
        self.scaler = StandardScaler()
        
    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray):
        """Fit with group information for ranking."""
        X_scaled = self.scaler.fit_transform(X)
        
        # Create group array for LightGBM (number of samples per case)
        unique_groups, group_counts = np.unique(groups, return_counts=True)
        group_array = group_counts.tolist()
        
        # For LambdaRank, we need relevance labels (1 for faulty, 0 for others)
        # LightGBM expects higher label = more relevant
        self.model = lgb.LGBMRanker(**self.params)
        self.model.fit(X_scaled, y, group=group_array)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict ranking scores."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def save(self, path: str):
        joblib.dump({'model': self.model, 'scaler': self.scaler, 'params': self.params}, path)
    
    @classmethod
    def load(cls, path: str):
        data = joblib.load(path)
        obj = cls.__new__(cls)
        obj.model = data['model']
        obj.scaler = data['scaler']
        obj.params = data['params']
        return obj


class XGBoostRanker:
    """XGBoost with ranking objective."""
    
    def __init__(self, n_estimators=200, learning_rate=0.05, max_depth=6, 
                 random_state=42, **kwargs):
        self.params = {
            'objective': 'rank:pairwise',
            'eval_metric': 'ndcg',
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'random_state': random_state,
            'verbosity': 0,
            'tree_method': 'hist',
            **kwargs
        }
        self.model = None
        self.scaler = StandardScaler()
        
    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)
        
        unique_groups, group_counts = np.unique(groups, return_counts=True)
        group_array = group_counts.tolist()
        
        self.model = xgb.XGBRanker(**self.params)
        self.model.fit(X_scaled, y, group=group_array)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def save(self, path: str):
        joblib.dump({'model': self.model, 'scaler': self.scaler, 'params': self.params}, path)
    
    @classmethod
    def load(cls, path: str):
        data = joblib.load(path)
        obj = cls.__new__(cls)
        obj.model = data['model']
        obj.scaler = data['scaler']
        obj.params = data['params']
        return obj


class ClassicalEnsemble:
    """Ensemble of classical ML models with stacking."""
    
    def __init__(self, random_state=42):
        self.base_models = {
            'rf': RandomForestClassifier(n_estimators=300, max_depth=10, 
                                        min_samples_split=5, random_state=random_state, n_jobs=-1),
            'et': ExtraTreesClassifier(n_estimators=300, max_depth=10, 
                                      min_samples_split=5, random_state=random_state, n_jobs=-1),
            'gb': GradientBoostingClassifier(n_estimators=200, max_depth=5, 
                                            learning_rate=0.05, random_state=random_state),
            'hgb': HistGradientBoostingClassifier(max_iter=200, max_depth=5, 
                                                 learning_rate=0.05, random_state=random_state),
        }
        self.meta_learner = LogisticRegression(random_state=random_state, max_iter=1000)
        self.scaler = StandardScaler()
        self.fitted_base = {}
        
    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)
        
        # Use LeaveOneGroupOut CV for out-of-fold predictions
        logo = LeaveOneGroupOut()
        n_samples = X.shape[0]
        n_models = len(self.base_models)
        oof_preds = np.zeros((n_samples, n_models))
        
        for model_idx, (name, model) in enumerate(self.base_models.items()):
            print(f"  Training base model: {name}")
            oof_pred = np.zeros(n_samples)
            
            for train_idx, val_idx in logo.split(X_scaled, y, groups):
                model.fit(X_scaled[train_idx], y[train_idx])
                oof_pred[val_idx] = model.predict_proba(X_scaled[val_idx])[:, 1]
            
            oof_preds[:, model_idx] = oof_pred
            # Refit on full data
            model.fit(X_scaled, y)
            self.fitted_base[name] = model
        
        # Train meta-learner on OOF predictions
        self.meta_learner.fit(oof_preds, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        
        base_preds = np.zeros((X.shape[0], len(self.fitted_base)))
        for idx, (name, model) in enumerate(self.fitted_base.items()):
            base_preds[:, idx] = model.predict_proba(X_scaled)[:, 1]
        
        # Meta-learner prediction
        return self.meta_learner.predict_proba(base_preds)[:, 1]
    
    def save(self, path: str):
        joblib.dump({
            'base_models': self.fitted_base,
            'meta_learner': self.meta_learner,
            'scaler': self.scaler
        }, path)
    
    @classmethod
    def load(cls, path: str):
        data = joblib.load(path)
        obj = cls.__new__(cls)
        obj.fitted_base = data['base_models']
        obj.meta_learner = data['meta_learner']
        obj.scaler = data['scaler']
        obj.base_models = {}  # Not needed after loading
        return obj


class UnsupervisedAnomalyDetector:
    """Unsupervised anomaly detection for complementary scoring."""
    
    def __init__(self, contamination=0.125):  # 1/8 cars expected to be faulty
        self.models = {
            'isolation_forest': None,  # Will use sklearn's IsolationForest
            'lof': LocalOutlierFactor(n_neighbors=3, contamination=contamination, novelty=True),
        }
        self.scaler = StandardScaler()
        self.fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray = None, groups: np.ndarray = None):
        """Fit on normal data only (y=0) if labels available, else all data."""
        from sklearn.ensemble import IsolationForest
        
        X_scaled = self.scaler.fit_transform(X)
        
        # If labels provided, fit only on normal samples
        if y is not None:
            normal_mask = (y == 0)
            X_normal = X_scaled[normal_mask]
        else:
            X_normal = X_scaled
        
        self.models['isolation_forest'] = IsolationForest(
            n_estimators=200, contamination=0.125, random_state=42, n_jobs=-1
        )
        self.models['isolation_forest'].fit(X_normal)
        self.models['lof'].fit(X_normal)
        self.fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores (higher = more anomalous)."""
        if not self.fitted:
            raise ValueError("Model not fitted")
        
        X_scaled = self.scaler.transform(X)
        
        # Isolation Forest: decision_function (higher = more normal), negate for anomaly
        iso_scores = -self.models['isolation_forest'].decision_function(X_scaled)
        
        # LOF: decision_function (higher = more normal), negate for anomaly
        lof_scores = -self.models['lof'].decision_function(X_scaled)
        
        # Average and normalize
        combined = (iso_scores + lof_scores) / 2
        # Min-max normalize to [0, 1]
        combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-8)
        
        return combined
    
    def save(self, path: str):
        joblib.dump({
            'models': self.models,
            'scaler': self.scaler,
            'fitted': self.fitted
        }, path)
    
    @classmethod
    def load(cls, path: str):
        data = joblib.load(path)
        obj = cls.__new__(cls)
        obj.models = data['models']
        obj.scaler = data['scaler']
        obj.fitted = data['fitted']
        return obj


class SiameseRanker:
    """Simple metric learning approach using distance to faulty prototypes."""
    
    def __init__(self, embedding_dim=64, random_state=42):
        self.embedding_dim = embedding_dim
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.prototype = None  # Mean embedding of faulty cars
        self.fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray):
        """Learn simple linear projection and compute faulty prototype."""
        from sklearn.decomposition import PCA
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        
        X_scaled = self.scaler.fit_transform(X)
        
        # Use LDA for supervised dimensionality reduction (max 1 component for binary)
        n_components = min(self.embedding_dim, len(np.unique(y)) - 1)
        if n_components > 0:
            try:
                self.lda = LDA(n_components=n_components)
                X_emb = self.lda.fit_transform(X_scaled, y)
            except:
                # Fallback to PCA
                self.lda = PCA(n_components=min(self.embedding_dim, X_scaled.shape[1]))
                X_emb = self.lda.fit_transform(X_scaled)
        else:
            self.lda = PCA(n_components=min(self.embedding_dim, X_scaled.shape[1]))
            X_emb = self.lda.fit_transform(X_scaled)
        
        # Compute prototype (mean embedding) of faulty cars
        faulty_mask = (y == 1)
        if np.any(faulty_mask):
            self.prototype = np.mean(X_emb[faulty_mask], axis=0)
        else:
            self.prototype = np.mean(X_emb, axis=0)
        
        self.fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly score as distance to faulty prototype (closer = more likely faulty)."""
        if not self.fitted:
            raise ValueError("Model not fitted")
        
        X_scaled = self.scaler.transform(X)
        X_emb = self.lda.transform(X_scaled)
        
        # Distance to prototype (negative because closer = more likely faulty)
        distances = np.linalg.norm(X_emb - self.prototype, axis=1)
        
        # Convert to score (higher = more likely faulty) by negating and normalizing
        scores = -distances
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        
        return scores
    
    def save(self, path: str):
        joblib.dump({
            'lda': self.lda,
            'scaler': self.scaler,
            'prototype': self.prototype,
            'fitted': self.fitted
        }, path)
    
    @classmethod
    def load(cls, path: str):
        data = joblib.load(path)
        obj = cls.__new__(cls)
        obj.lda = data['lda']
        obj.scaler = data['scaler']
        obj.prototype = data['prototype']
        obj.fitted = data['fitted']
        return obj


def train_all_models(X: np.ndarray, y: np.ndarray, groups: np.ndarray, 
                     feature_names: List[str]) -> Dict:
    """Train all models and return dictionary of fitted models."""
    models = {}
    
    print("Training LightGBM Ranker...")
    models['lgbm'] = LightGBMRanker(n_estimators=300, learning_rate=0.03, num_leaves=63)
    models['lgbm'].fit(X, y, groups)
    
    print("Training XGBoost Ranker...")
    models['xgb'] = XGBoostRanker(n_estimators=300, learning_rate=0.03, max_depth=5)
    models['xgb'].fit(X, y, groups)
    
    print("Training Classical Ensemble...")
    models['classical'] = ClassicalEnsemble()
    models['classical'].fit(X, y, groups)
    
    print("Training Unsupervised Anomaly Detector...")
    models['unsup'] = UnsupervisedAnomalyDetector()
    models['unsup'].fit(X, y, groups)
    
    print("Training Siamese/Metric Learning...")
    models['siamese'] = SiameseRanker()
    models['siamese'].fit(X, y, groups)
    
    return models


def evaluate_all_models(models: Dict, X: np.ndarray, y: np.ndarray, 
                       groups: np.ndarray) -> Dict:
    """Evaluate all models on validation data."""
    results = {}
    
    for name, model in models.items():
        print(f"Evaluating {name}...")
        scores = model.predict(X)
        eval_result = evaluate_ranking(y, scores, groups)
        results[name] = {
            'scores': scores,
            'mean_score': eval_result['mean_score'],
            'details': eval_result['details']
        }
        print(f"  {name}: mean rank-decay = {eval_result['mean_score']:.4f}")
    
    return results


def optimize_ensemble_weights(model_scores: Dict, y: np.ndarray, groups: np.ndarray,
                              n_cars: int = 8) -> np.ndarray:
    """Optimize ensemble weights using grid search on validation."""
    from itertools import product
    
    model_names = list(model_scores.keys())
    n_models = len(model_names)
    
    # Get scores for each model
    score_matrix = np.column_stack([model_scores[name]['scores'] for name in model_names])
    
    # Simple grid search over weight combinations
    best_weights = None
    best_score = -1
    
    # Try uniform weights
    weights = np.ones(n_models) / n_models
    ensemble_scores = score_matrix @ weights
    eval_result = evaluate_ranking(y, ensemble_scores, groups, n_cars)
    if eval_result['mean_score'] > best_score:
        best_score = eval_result['mean_score']
        best_weights = weights.copy()
    
    # Try emphasizing each model
    for i in range(n_models):
        weights = np.ones(n_models) * 0.1
        weights[i] = 0.1 + 0.9 * (n_models - 1) / n_models  # Give more weight to model i
        weights = weights / weights.sum()
        ensemble_scores = score_matrix @ weights
        eval_result = evaluate_ranking(y, ensemble_scores, groups, n_cars)
        if eval_result['mean_score'] > best_score:
            best_score = eval_result['mean_score']
            best_weights = weights.copy()
    
    # Try all pairs
    for i in range(n_models):
        for j in range(i+1, n_models):
            weights = np.ones(n_models) * 0.05
            weights[i] = 0.4
            weights[j] = 0.4
            weights = weights / weights.sum()
            ensemble_scores = score_matrix @ weights
            eval_result = evaluate_ranking(y, ensemble_scores, groups, n_cars)
            if eval_result['mean_score'] > best_score:
                best_score = eval_result['mean_score']
                best_weights = weights.copy()
    
    print(f"Best ensemble weights: {dict(zip(model_names, best_weights))}")
    print(f"Best ensemble score: {best_score:.4f}")
    
    return best_weights


def run_loco_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray, 
                feature_names: List[str]) -> Dict:
    """Run Leave-One-Case-Out CV and return detailed results."""
    logo = LeaveOneGroupOut()
    unique_groups = np.unique(groups)
    n_cases = len(unique_groups)
    
    all_predictions = np.zeros(len(y))
    all_true = np.zeros(len(y))
    fold_results = {}
    
    for fold, (train_idx, val_idx) in enumerate(logo.split(X, y, groups)):
        case_id = groups[val_idx[0]]
        print(f"\n--- Fold {fold+1}/{n_cases} (Case {case_id}) ---")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        g_train, g_val = groups[train_idx], groups[val_idx]
        
        # Train models on this fold
        models = train_all_models(X_train, y_train, g_train, feature_names)
        
        # Evaluate each model
        fold_scores = {}
        for name, model in models.items():
            scores = model.predict(X_val)
            eval_result = evaluate_ranking(y_val, scores, g_val)
            fold_scores[name] = {
                'scores': scores,
                'mean_score': eval_result['mean_score']
            }
            print(f"  {name}: {eval_result['mean_score']:.4f}")
        
        # Optimize ensemble weights on this fold
        weights = optimize_ensemble_weights(fold_scores, y_val, g_val)
        
        # Compute ensemble prediction
        score_matrix = np.column_stack([fold_scores[name]['scores'] for name in fold_scores.keys()])
        ensemble_scores = score_matrix @ weights
        ensemble_result = evaluate_ranking(y_val, ensemble_scores, g_val)
        print(f"  Ensemble: {ensemble_result['mean_score']:.4f}")
        
        all_predictions[val_idx] = ensemble_scores
        all_true[val_idx] = y_val
        
        fold_results[case_id] = {
            'weights': weights,
            'model_scores': {k: v['mean_score'] for k, v in fold_scores.items()},
            'ensemble_score': ensemble_result['mean_score'],
            'details': ensemble_result['details']
        }
    
    # Overall evaluation
    overall_result = evaluate_ranking(all_true, all_predictions, groups)
    print(f"\n=== OVERALL LOCO-CV Score: {overall_result['mean_score']:.4f} ===")
    
    return {
        'overall_score': overall_result['mean_score'],
        'predictions': all_predictions,
        'true_labels': all_true,
        'fold_results': fold_results,
        'details': overall_result['details']
    }


if __name__ == "__main__":
    from data_loader import load_data
    from features import build_feature_matrix
    
    print("Loading data...")
    data = load_data()
    
    print("Building features...")
    X, y, groups, car_ids, feature_names, X_test, test_car_ids = build_feature_matrix(
        data['train_cases'], data['test_df']
    )
    
    print(f"Data: X={X.shape}, y={y.sum()}/{len(y)} faulty")
    
    # Run LOCO-CV
    results = run_loco_cv(X, y, groups, feature_names)
    
    # Save results
    joblib.dump(results, 'results/loco_cv_results.pkl')
    print("Results saved to results/loco_cv_results.pkl")