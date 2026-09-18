import os
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score, balanced_accuracy_score
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
import lightgbm as lgb

from src.rail.data_loader import load_training_dataset, find_rail_data_dir
from src.rail.features import extract_features

def _extract_single_train_file(file_path: Path):
    fn = file_path.name
    feats = extract_features(file_path)
    feats['filename'] = fn
    return feats

def train_rail_pipeline(data_dir=None):
    print("=" * 70)
    print("LTA NEBULAX: PS3 RAIL CORRUGATION CLASSIFICATION TRAINING")
    print("=" * 70)

    # 1. Load Dataset
    train_files, df_labels = load_training_dataset(data_dir)
    print(f"Discovered {len(train_files)} training files in Train/ folder.")
    print("Label distribution in dataset:")
    print(df_labels['label'].value_counts())

    # 2. Extract Features in Parallel
    print("\n[Step 1/3] Extracting bilateral vibration, shock & spectral features...")
    train_rows = Parallel(n_jobs=-1, backend="loky")(
        delayed(_extract_single_train_file)(fp) for fp in tqdm(train_files, desc="Processing Train Signals")
    )
    df_train = pd.DataFrame(train_rows)
    df_merged = pd.merge(df_train, df_labels, on='filename')

    X = df_merged.drop(columns=['filename', 'label']).fillna(0.0)
    y = df_merged['label'].values
    feature_names = list(X.columns)

    print(f"Extracted {len(feature_names)} engineered features per recording.")

    # Encode target labels
    classes = sorted(list(np.unique(y)))
    label_to_id = {c: i for i, c in enumerate(classes)}
    id_to_label = {i: c for i, c in enumerate(classes)}
    y_encoded = np.array([label_to_id[val] for val in y])

    # 3. 5-Fold Stratified Cross-Validation Benchmark
    print("\n[Step 2/3] Benchmarking Classifiers with 5-Fold Stratified CV...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Classifiers configured for imbalanced multi-class detection
    models = {
        'ExtraTrees': lambda: ExtraTreesClassifier(
            n_estimators=300,
            max_depth=12,
            class_weight='balanced',
            random_state=42,
            n_jobs=1
        ),
        'RandomForest': lambda: RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=1
        ),
        'LightGBM': lambda: lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            class_weight='balanced',
            random_state=42,
            n_jobs=1,
            verbosity=-1
        ),
        'GradientBoosting': lambda: GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=4,
            random_state=42
        )
    }

    results = {}
    oof_predictions = {}

    print("-" * 70)
    print(f"{'Model Name':<20} | {'Macro F1':<15} | {'Balanced Accuracy':<20}")
    print("-" * 70)

    for name, model_factory in models.items():
        oof = np.zeros(len(y_encoded), dtype=int)
        for train_idx, val_idx in skf.split(X, y_encoded):
            X_tr, y_tr = X.iloc[train_idx], y_encoded[train_idx]
            X_va, y_va = X.iloc[val_idx], y_encoded[val_idx]

            clf = model_factory()
            clf.fit(X_tr, y_tr)
            oof[val_idx] = clf.predict(X_va)

        macro_f1 = float(f1_score(y_encoded, oof, average='macro'))
        bal_acc = float(balanced_accuracy_score(y_encoded, oof))
        results[name] = {'macro_f1': macro_f1, 'balanced_accuracy': bal_acc, 'oof': oof}
        oof_predictions[name] = oof

        print(f"{name:<20} | {macro_f1 * 100:>12.2f}% | {bal_acc * 100:>17.2f}%")

    print("-" * 70)
    best_model_name = max(results, key=lambda k: results[k]['macro_f1'])
    best_f1 = results[best_model_name]['macro_f1']
    best_bal_acc = results[best_model_name]['balanced_accuracy']
    best_oof = results[best_model_name]['oof']

    print(f"\n[WINNING MODEL] {best_model_name}")
    print(f"  Macro F1 Score:    {best_f1 * 100:.2f}%")
    print(f"  Balanced Accuracy: {best_bal_acc * 100:.2f}%")

    # Log Detailed Classification Report & Confusion Matrix
    y_true_str = [id_to_label[i] for i in y_encoded]
    y_pred_str = [id_to_label[i] for i in best_oof]

    print("\nDetailed Out-of-Fold Classification Report:")
    print(classification_report(y_true_str, y_pred_str, target_names=classes, digits=4))

    cm = confusion_matrix(y_true_str, y_pred_str, labels=classes)
    df_cm = pd.DataFrame(cm, index=[f'True {c}' for c in classes], columns=[f'Pred {c}' for c in classes])
    print("Out-of-Fold Confusion Matrix:")
    print(df_cm)

    # 4. Train Best Model on 100% of Training Data & Save Artifacts
    print(f"\n[Step 3/3] Training final {best_model_name} on all 272 training recordings...")
    final_clf = models[best_model_name]()
    final_clf.fit(X, y_encoded)

    models_dir = Path(__file__).resolve().parents[2] / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    artifact_payload = {
        'model': final_clf,
        'feature_names': feature_names,
        'classes': classes,
        'label_to_id': label_to_id,
        'id_to_label': id_to_label,
        'best_model_name': best_model_name,
        'macro_f1': best_f1,
        'balanced_accuracy': best_bal_acc
    }

    best_model_path = models_dir / "rail_best_model.joblib"
    joblib.dump(artifact_payload, best_model_path)

    compat_model_path = models_dir / "rail_model.joblib"
    joblib.dump(artifact_payload, compat_model_path)

    print(f"[SUCCESS] Model artifact saved to: {best_model_path}")
    print(f"[SUCCESS] Model artifact mirror saved to: {compat_model_path}")

    return final_clf, feature_names, best_f1

if __name__ == '__main__':
    train_rail_pipeline()
