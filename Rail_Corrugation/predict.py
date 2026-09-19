"""
Inference script for rail corrugation detection using joblib model.
Required CLI: predict.py --input <test_folder> --output <predictions.csv>
"""

import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.features import load_and_extract


def main():
    parser = argparse.ArgumentParser(description="Predict rail corrugation faults")
    parser.add_argument("--input", type=str, required=True, help="Test data directory")
    parser.add_argument("--output", type=str, required=True, help="Output CSV file")
    parser.add_argument("--model", type=str, default=None, help="Model path (default: models/rail_best_model_final.joblib)")
    
    args = parser.parse_args()
    
    # Resolve model path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = args.model or os.path.join(base_dir, "models", "rail_best_model_final.joblib")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    
    # Load optimal thresholds if available
    thresholds_path = os.path.join(base_dir, "models", "rail_optimal_thresholds_final.joblib")
    if os.path.exists(thresholds_path):
        thresholds = joblib.load(thresholds_path)
        print(f"Loaded optimal thresholds: {thresholds}")
    else:
        thresholds = None
    
    # Get test files
    test_files = sorted([f for f in os.listdir(args.input) if f.endswith(".csv")])
    print(f"Found {len(test_files)} test files")
    
    label_map = {0: "Normal", 1: "Side I", 2: "Side II"}
    
    results = []
    for fname in tqdm(test_files, desc="Predicting"):
        fpath = os.path.join(args.input, fname)
        
        # Extract features
        try:
            feats, names = load_and_extract(fpath)
        except Exception as e:
            print(f"Error extracting features from {fname}: {e}")
            results.append({"file_id": fname, "prediction": "Normal"})
            continue
        
        # Predict
        pred_idx = model.predict(feats.reshape(1, -1))[0]
        
        # Apply thresholds if available (for probabilistic models)
        if hasattr(model, "predict_proba") and thresholds is not None:
            proba = model.predict_proba(feats.reshape(1, -1))[0]
            # Apply threshold logic if needed
            pred_idx = np.argmax(proba)
        
        pred_label = label_map.get(int(pred_idx), "Normal")
        results.append({"file_id": fname, "prediction": pred_label})
    
    # Save predictions
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"\nSaved predictions to {args.output}")
    print(f"Prediction distribution: {df['prediction'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()