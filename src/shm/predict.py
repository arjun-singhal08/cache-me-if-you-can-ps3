import os
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from src.shm.data_loader import load_test_dataset, find_shm_data_dir
from src.shm.features import extract_features

def _process_test_file(fp: Path):
    fn = fp.name
    feats = extract_features(fp)
    feats['filename'] = fn
    return feats

def generate_shm_predictions(data_dir=None, output_path=None, model_path=None) -> pd.DataFrame:
    """
    Inference & Official Submission Generator for PS3 Structural Health Monitoring.
    """
    # 1. Resolve paths
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
            raise FileNotFoundError("Trained SHM model not found at models/shm_best_model.joblib. Please run src/shm/train.py first.")
            
    if output_path is None:
        output_path = base_dir / "predictions" / "shm_predictions.csv"
    else:
        output_path = Path(output_path)
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 2. Load model artifact
    print(f"Loading trained model artifact from: {model_path}")
    payload = joblib.load(model_path)
    model = payload['model']
    feature_names = payload['feature_names']
    
    # 3. Load Test files
    test_files = load_test_dataset(data_dir)
    print(f"Discovered {len(test_files)} test files to evaluate.")
    
    # 4. Extract features in parallel
    print("Extracting features for held-out test files...")
    test_rows = Parallel(n_jobs=-1, backend="loky")(
        delayed(_process_test_file)(fp) for fp in tqdm(test_files, desc="Inferencing Test Signals")
    )
    df_test = pd.DataFrame(test_rows)
    X_test = df_test[feature_names].fillna(0.0)
    
    # 5. Predict & invert log-transformation
    preds_log = model.predict(X_test)
    preds = np.expm1(preds_log)
    
    # 6. Clip to physically valid non-negative damage bounds [0, 1]
    preds = np.clip(preds, a_min=1e-5, a_max=1.0)
    
    # 7. Format output DataFrame
    df_submission = pd.DataFrame({
        'file_id': df_test['filename'],
        'prediction': np.round(preds, 6)
    })
    
    # Sort naturally (test01.csv, test02.csv, ...)
    df_submission = df_submission.sort_values(by='file_id').reset_index(drop=True)
    
    # 8. Export CSV
    df_submission.to_csv(output_path, index=False)
    print(f"\n[SUCCESS] Generated official submission file: {output_path}")
    print("\nSubmission Preview:")
    print(df_submission.head(10))
    print(f"\nTotal rows generated: {len(df_submission)}")
    
    return df_submission

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate SHM Predictions for PS3")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to SHM dataset directory")
    parser.add_argument("--output", type=str, default=None, help="Output path for shm_predictions.csv")
    parser.add_argument("--model", type=str, default=None, help="Path to trained model artifact")
    args = parser.parse_args()
    
    generate_shm_predictions(data_dir=args.data_dir, output_path=args.output, model_path=args.model)

