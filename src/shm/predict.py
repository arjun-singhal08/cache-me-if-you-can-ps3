import os
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from typing import Union, List

from src.shm.data_loader import load_test_dataset, find_shm_data_dir
from src.shm.features import extract_features

def _load_shm_artifact(model_path: Union[str, Path] = None):
    base_dir = Path(__file__).resolve().parents[2]
    if model_path is None:
        candidate_models = [
            base_dir / "models" / "shm_best_model.joblib",
            base_dir / "models" / "shm_model.joblib"
        ]
        for cm in candidate_models:
            if cm.exists():
                model_path = cm
                break
        if model_path is None:
            raise FileNotFoundError("Trained SHM model artifact not found at models/shm_best_model.joblib. Please run src/shm/train.py first.")
    else:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Specified model artifact not found at: {model_path}")
            
    payload = joblib.load(model_path)
    return payload['model'], payload['feature_names']


def predict_single_signal(file_or_signal: Union[str, Path, np.ndarray, pd.DataFrame, pd.Series], model=None, feature_names=None) -> float:
    """
    Inference on a single signal or CSV file using the exact ExtraTrees pipeline.
    """
    if model is None or feature_names is None:
        model, feature_names = _load_shm_artifact()
        
    feats = extract_features(file_or_signal)
    df_feat = pd.DataFrame([feats])[feature_names].fillna(0.0)
    pred_log = model.predict(df_feat)[0]
    pred_damage = float(np.expm1(pred_log))
    
    # Enforce physical damage bounds [0.0, 1.0]
    return float(np.clip(pred_damage, 0.0, 1.0))


def generate_shm_predictions(input_target: Union[str, Path] = None, output_path: Union[str, Path] = None, model_path: Union[str, Path] = None) -> pd.DataFrame:
    """
    Unified Inference & Submission Generator for PS3 Structural Health Monitoring.
    
    Parameters:
        input_target: str/Path (directory containing test CSVs, single CSV file, or None for default test set)
        output_path: str/Path to write output predictions CSV
        model_path: str/Path to trained model artifact
    """
    base_dir = Path(__file__).resolve().parents[2]
    model, feature_names = _load_shm_artifact(model_path)

    # 1. Resolve input files
    target_files: List[Path] = []
    
    if input_target is None:
        target_files = load_test_dataset()
    else:
        inp = Path(input_target)
        if inp.is_file():
            target_files = [inp]
        elif inp.is_dir():
            # If pointing to root SHM folder containing 'Test' subfolder, resolve to Test/
            if (inp / "Test").exists() and (inp / "Test").is_dir():
                inp = inp / "Test"
            target_files = sorted(list(inp.glob("*.csv")))
            if not target_files:
                raise FileNotFoundError(f"No CSV files found in input directory: {inp}")
        else:
            raise FileNotFoundError(f"Input path does not exist: {inp}")

    print(f"Running inference on {len(target_files)} file(s)...")

    # 2. Extract features and predict
    results = []
    for fp in target_files:
        feats = extract_features(fp)
        df_feat = pd.DataFrame([feats])[feature_names].fillna(0.0)
        pred_log = model.predict(df_feat)[0]
        pred_val = float(np.clip(np.expm1(pred_log), 0.0, 1.0))
        results.append({
            'file_id': fp.name,
            'prediction': np.round(pred_val, 6)
        })

    df_submission = pd.DataFrame(results)
    df_submission = df_submission.sort_values(by='file_id').reset_index(drop=True)

    # 3. Export CSV if output path is requested or default
    if output_path is None:
        output_path = base_dir / "predictions" / "shm_predictions.csv"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_submission.to_csv(output_path, index=False)
    print(f"[SUCCESS] Output written to: {output_path}")
    print(df_submission.head(16))

    return df_submission


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PS3 SHM Inference & Predictions Generator")
    parser.add_argument("--input", "-i", type=str, default=None, help="Input directory or single CSV file")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output destination CSV path")
    parser.add_argument("--model", "-m", type=str, default=None, help="Path to trained model artifact")
    args = parser.parse_args()

    generate_shm_predictions(input_target=args.input, output_path=args.output, model_path=args.model)
