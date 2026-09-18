import argparse
import sys
import os
import re
from pathlib import Path
from typing import List, Union, Tuple, Optional
import joblib
import pandas as pd
import numpy as np

from src.rail.data_loader import natural_sort_key, validate_vibration_data
from src.rail.features import extract_features

def load_rail_model(model_path: Optional[Union[str, Path]] = None):
    """
    Load the trained Rail Corrugation model artifact.
    """
    if model_path is None:
        base_dir = Path(__file__).resolve().parents[2]
        candidates = [
            base_dir / "models" / "rail_best_model.joblib",
            base_dir / "models" / "rail_model.joblib",
            Path("models/rail_best_model.joblib"),
            Path("models/rail_model.joblib")
        ]
        for c in candidates:
            if c.exists():
                model_path = c
                break
        if model_path is None:
            raise FileNotFoundError("Trained rail model artifact not found in models/")

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    artifact = joblib.load(model_path)
    return artifact


def predict_single_file(file_path: Union[str, Path], artifact: Optional[dict] = None) -> Tuple[str, dict]:
    """
    Predicts corrugation state ('Normal', 'Side I', 'Side II') for a single recording.
    Returns:
        predicted_class: str
        details: dict containing feature highlights and class probabilities
    """
    file_path = Path(file_path)
    if artifact is None:
        artifact = load_rail_model()

    model = artifact['model']
    feature_names = artifact['feature_names']
    id_to_label = artifact['id_to_label']

    feats = extract_features(file_path)
    feat_vector = pd.DataFrame([[feats.get(fn, 0.0) for fn in feature_names]], columns=feature_names)

    pred_id = int(model.predict(feat_vector)[0])
    predicted_label = id_to_label[pred_id]

    probs = {}
    if hasattr(model, 'predict_proba'):
        prob_vals = model.predict_proba(feat_vector)[0]
        for idx, p in enumerate(prob_vals):
            probs[id_to_label[idx]] = float(p)

    details = {
        'file_id': file_path.name,
        'prediction': predicted_label,
        'probabilities': probs,
        's1_vib_rms': feats.get('s1_vib_rms', 0.0),
        's2_vib_rms': feats.get('s2_vib_rms', 0.0),
        'asym_vib_rms_ratio': feats.get('asym_vib_rms_ratio', 1.0),
        'norm_asym_vib_index': feats.get('norm_asym_vib_index', 0.0),
        's1_shk_rms': feats.get('s1_shk_rms', 0.0),
        's2_shk_rms': feats.get('s2_shk_rms', 0.0),
        'asym_shk_rms_ratio': feats.get('asym_shk_rms_ratio', 1.0),
        's1_dom_freq': feats.get('s1_dom_freq', 0.0),
        's2_dom_freq': feats.get('s2_dom_freq', 0.0),
    }

    return predicted_label, details


def generate_rail_predictions(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    model_path: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Batch inference for rail corrugation classification.
    Accepts a single CSV file or a directory containing multiple CSV test files.
    """
    input_p = Path(input_path)
    output_p = Path(output_path)
    output_p.parent.mkdir(parents=True, exist_ok=True)

    artifact = load_rail_model(model_path)
    model = artifact['model']
    feature_names = artifact['feature_names']
    id_to_label = artifact['id_to_label']

    if input_p.is_dir():
        files = sorted(list(input_p.glob("*.csv")), key=natural_sort_key)
        if not files:
            raise FileNotFoundError(f"No CSV files found in input directory: {input_p}")
    else:
        if not input_p.exists():
            raise FileNotFoundError(f"Input file not found: {input_p}")
        files = [input_p]

    print(f"Running rail corrugation inference on {len(files)} file(s)...")

    results = []
    for fp in files:
        feats = extract_features(fp)
        feat_vector = pd.DataFrame([[feats.get(fn, 0.0) for fn in feature_names]], columns=feature_names)
        pred_id = int(model.predict(feat_vector)[0])
        pred_label = id_to_label[pred_id]
        results.append({'file_id': fp.name, 'prediction': pred_label})

    df_out = pd.DataFrame(results, columns=['file_id', 'prediction'])
    df_out.to_csv(output_p, index=False)
    print(f"[SUCCESS] Output written to: {output_p}")
    print(df_out.head(10))
    print(f"\nPrediction value counts:\n{df_out['prediction'].value_counts()}")

    return df_out


def main():
    parser = argparse.ArgumentParser(description="Rail Corrugation Fault Classifier CLI")
    parser.add_argument("--input", required=True, help="Path to a single vibration CSV file or directory of CSV test files")
    parser.add_argument("--output", default="predictions/rail_predictions.csv", help="Path to output CSV destination")
    parser.add_argument("--model", default=None, help="Optional path to model artifact .joblib")

    args = parser.parse_args()
    generate_rail_predictions(args.input, args.output, args.model)


if __name__ == "__main__":
    main()
