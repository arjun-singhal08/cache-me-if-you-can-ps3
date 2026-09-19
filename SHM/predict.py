#!/usr/bin/env python3
"""
SHM Fatigue Damage Prediction - Competition Entry Point (Hybrid Model)
Usage: python predict.py --input <test_dir> --output <predictions.csv>
"""
import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.features.physics_features import extract_features_batch
from src.models.hybrid_model import HybridEnsembleRegressor
from src.utils.io import get_filepaths, prepare_submission


def main():
    parser = argparse.ArgumentParser(description='SHM Fatigue Damage Prediction (Hybrid)')
    parser.add_argument('--input', type=str, required=True, help='Input test directory')
    parser.add_argument('--output', type=str, required=True, help='Output predictions CSV')
    parser.add_argument('--model-dir', type=str, default='models', help='Model artifacts directory')
    parser.add_argument('--n-tta', type=int, default=20, help='Test-time augmentation iterations')
    parser.add_argument('--sigma-uts-min', type=float, default=600.0)
    parser.add_argument('--sigma-uts-max', type=float, default=900.0)
    args = parser.parse_args()

    test_dir = Path(args.input)
    model_dir = Path(args.model_dir)

    # Get test files
    test_files = get_filepaths(test_dir)
    if not test_files:
        print(f"No CSV files found in {test_dir}")
        sys.exit(1)

    print(f"Found {len(test_files)} test files")
    print(f"Loading model from {model_dir}...")

    # Load model
    model = HybridEnsembleRegressor.load(model_dir)
    print(f"Loaded hybrid model (blend weight: {model.blend_weight:.4f})")

    # Extract features
    print("Extracting physics features...")
    m_values = [3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0, 10.0]
    test_features = extract_features_batch(
        test_files,
        m_values=m_values,
        sigma_uts_range=(args.sigma_uts_min, args.sigma_uts_max),
        n_jobs=-1
    )

    # Prepare feature matrix
    feature_cols = [c for c in test_features.columns if c != 'file_id']
    X_test = test_features[feature_cols].values.astype(np.float32)

    # Ensure feature alignment
    if X_test.shape[1] > len(model.feature_cols):
        X_test = X_test[:, :len(model.feature_cols)]
    elif X_test.shape[1] < len(model.feature_cols):
        pad = np.zeros((X_test.shape[0], len(model.feature_cols) - X_test.shape[1]), dtype=np.float32)
        X_test = np.hstack([X_test, pad])

    # Predict
    print(f"Predicting with hybrid model (TTA={args.n_tta})...")
    predictions = model.predict(X_test)

    # Save submission
    prepare_submission(predictions, test_files, args.output)
    print(f"Predictions saved to {args.output}")

    # Print summary
    print(f"\nPrediction summary:")
    print(f"  Mean: {predictions.mean():.6f}")
    print(f"  Std:  {predictions.std():.6f}")
    print(f"  Min:  {predictions.min():.6f}")
    print(f"  Max:  {predictions.max():.6f}")


if __name__ == '__main__':
    main()