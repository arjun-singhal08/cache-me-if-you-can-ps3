import os
import glob
import argparse
import joblib
import numpy as np
import pandas as pd
from src.extract_features import extract_features_from_signal

def run_inference(input_dir: str, output_dir: str, model_path: str = "models/shm_model.pkl"):
    """
    Inference script conforming to NebulaX PS3 submission contract.
    Reads test files from input_dir, extracts features, runs model, and writes shm_predictions.csv.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found. Please run src/train.py first.")
        
    model_payload = joblib.load(model_path)
    model = model_payload['model']
    feature_names = model_payload['feature_names']
    use_log_target = model_payload.get('use_log_target', False)
    
    test_files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    if not test_files:
        raise ValueError(f"No CSV files found in input directory: {input_dir}")
        
    print(f"Running SHM inference on {len(test_files)} files from {input_dir}...")
    
    records = []
    for filepath in test_files:
        filename = os.path.basename(filepath)
        df_sig = pd.read_csv(filepath, header=None)
        signal = df_sig.iloc[:, 0].values
        
        feats = extract_features_from_signal(signal)
        records.append({'file_id': filename, **feats})
        
    df_test_feats = pd.DataFrame(records)
    
    # Ensure feature alignment
    X_test = df_test_feats[feature_names]
    
    # Predict
    raw_preds = model.predict(X_test)
    if use_log_target:
        preds = np.expm1(raw_preds)
    else:
        preds = raw_preds
        
    preds = np.clip(preds, a_min=1e-6, a_max=None)
    
    # Format output DataFrame
    df_out = pd.DataFrame({
        'file_id': df_test_feats['file_id'],
        'prediction': preds
    })
    
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "shm_predictions.csv")
    df_out.to_csv(out_file, index=False)
    print(f"Successfully generated predictions at: {out_file}")
    print("\nPreview of shm_predictions.csv:")
    print(df_out.to_string(index=False))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SHM Subsystem Prediction Script")
    parser.add_argument("--input", required=True, help="Directory containing test CSV files")
    parser.add_argument("--output", required=True, help="Directory to save shm_predictions.csv")
    
    args = parser.parse_args()
    run_inference(args.input, args.output)

