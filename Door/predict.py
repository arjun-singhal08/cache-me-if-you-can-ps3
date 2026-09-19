#!/usr/bin/env python3
# predict.py - Inference script for Door fault diagnosis
# Usage: python predict.py --input Test.csv --output door_predictions.csv

import os
import sys
import argparse
import yaml
import joblib
import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.io import load_test_data, format_datetime, save_predictions
from segmentation.detector import DoorSegmenter
from features.extract import extract_segment_features
from classification.ensemble import DoorFaultEnsemble
from postprocess.optimize import optimize_predictions

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='Door fault diagnosis prediction')
    parser.add_argument('--input', required=True, help='Input test CSV file')
    parser.add_argument('--output', required=True, help='Output predictions CSV file')
    parser.add_argument('--model-dir', default=None, help='Model directory (default: app/model)')
    args = parser.parse_args()
    
    # Resolve paths - predict.py is in app/, so model is in app/model/
    base_dir = os.path.dirname(os.path.abspath(__file__))  # This is app/
    app_dir = base_dir
    model_dir = args.model_dir or os.path.join(app_dir, 'model')
    config_path = os.path.join(app_dir, 'config.yaml')
    
    # Load config
    config = load_config(config_path)
    
    # Load models
    print(f"Loading models from {model_dir}...")
    segmenter = joblib.load(os.path.join(model_dir, 'segmenter.pkl'))
    classifier = DoorFaultEnsemble.load(os.path.join(model_dir, 'classifier.pkl'))
    feature_names = joblib.load(os.path.join(model_dir, 'feature_names.pkl'))
    
    # Load test data
    print(f"Loading test data from {args.input}...")
    test_df = load_test_data(args.input)
    print(f"Test data: {len(test_df)} rows")
    
    # Stage 1: Segmentation
    print("Running segmentation...")
    pred_segments = segmenter.predict(test_df)
    print(f"Detected {len(pred_segments)} segments")
    
    if not pred_segments:
        print("No segments detected!")
        save_predictions([], args.output)
        return
    
    # Stage 2: Feature extraction
    print("Extracting features...")
    pred_features = []
    pred_seg_dfs = []
    for start, end, op in pred_segments:
        mask = (test_df['dt'] >= start) & (test_df['dt'] <= end)
        seg_df = test_df[mask].copy()
        if len(seg_df) >= 20:
            feat = extract_segment_features(seg_df, op)
            # Ensure feature order matches training
            feat_vec = [feat.get(name, 0.0) for name in feature_names]
            pred_features.append(feat_vec)
            pred_seg_dfs.append(seg_df)
        else:
            # Pad with zeros for too-short segments
            pred_features.append([0.0] * len(feature_names))
            pred_seg_dfs.append(seg_df)
    
    X_pred = np.array(pred_features)
    
    # Stage 3: Classification
    print("Running classification...")
    probas = classifier.predict_proba(X_pred, pred_seg_dfs)
    
    # Stage 4: Prepare raw predictions with confidence
    raw_segments = []
    for (start, end, op), p_abn in zip(pred_segments, probas):
        if p_abn > 0.5:
            label = 'Abnormal resistance'
            conf = float(p_abn)
        else:
            label = 'Normal'
            conf = float(1 - p_abn)
        raw_segments.append((start, end, label, conf))
    
    # Stage 5: Post-processing for IoU optimization
    print("Post-processing...")
    timestamps = test_df['dt'].values
    current = test_df['Motor current(mA)'].values
    position = test_df['Door leaf position'].values
    
    final_segments = optimize_predictions(
        raw_segments, timestamps, current, position, config.get('postprocess', {})
    )
    
    print(f"Final predictions: {len(final_segments)} segments")
    for s, e, l in final_segments:
        print(f"  {format_datetime(s)} - {format_datetime(e)}: {l}")
    
    # Save
    save_predictions(final_segments, args.output)
    print(f"\nPredictions saved to {args.output}")

if __name__ == '__main__':
    main()