#!/usr/bin/env python
"""
ACV Refrigerant Leak Localisation - Prediction Script
Usage: python predict.py --input <input_dir> --output <output_csv>
"""
import argparse
import os
import sys
import joblib
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import ACVDataLoader
from features import build_feature_matrix, extract_car_features
from models import (
    LightGBMRanker, XGBoostRanker, ClassicalEnsemble, 
    UnsupervisedAnomalyDetector, SiameseRanker
)


def load_models(model_dir: str = "models"):
    """Load all trained models and metadata."""
    metadata = joblib.load(os.path.join(model_dir, 'ensemble_metadata.pkl'))
    weights = metadata['weights']
    feature_names = metadata['feature_names']
    model_names = metadata['model_names']
    
    models = {}
    for name in model_names:
        model_path = os.path.join(model_dir, f'{name}_model.pkl')
        if name == 'lgbm':
            models[name] = LightGBMRanker.load(model_path)
        elif name == 'xgb':
            models[name] = XGBoostRanker.load(model_path)
        elif name == 'classical':
            models[name] = ClassicalEnsemble.load(model_path)
        elif name == 'unsup':
            models[name] = UnsupervisedAnomalyDetector.load(model_path)
        elif name == 'siamese':
            models[name] = SiameseRanker.load(model_path)
    
    return models, weights, feature_names


def parse_car_dataframes(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Parse a dataframe into per-car dataframes."""
    car_params = {}
    for col in df.columns:
        if col.startswith('Car ') and ' - ' in col:
            cid, param = col.split(' - ', 1)
            cid = cid.replace('Car ', '')
            if cid not in car_params:
                car_params[cid] = []
            car_params[cid].append(param)
    
    id_cols = ['Car model', 'Train number', 'Time']
    car_dfs = {}
    for car_id, params in car_params.items():
        car_cols = id_cols + [f'Car {car_id} - {p}' for p in params]
        car_df = df[car_cols].copy()
        car_df.columns = id_cols + params
        car_dfs[car_id] = car_df
    
    return car_dfs


def build_test_features(test_df: pd.DataFrame, feature_names: list) -> Tuple[np.ndarray, list]:
    """Build feature matrix for test data."""
    car_dfs = parse_car_dataframes(test_df)
    
    test_features = []
    test_car_ids = []
    
    for car_id in sorted(car_dfs.keys()):
        car_df = car_dfs[car_id]
        feats = extract_car_features(car_df, car_id, car_dfs)
        test_features.append(feats)
        test_car_ids.append(car_id)
    
    test_feat_df = pd.DataFrame(test_features)
    
    # Align columns with training
    for col in feature_names:
        if col not in test_feat_df.columns:
            test_feat_df[col] = 0
    test_feat_df = test_feat_df[feature_names].fillna(0).replace([np.inf, -np.inf], 0)
    
    return test_feat_df.values, test_car_ids


def predict_case(models: dict, weights: np.ndarray, feature_names: list, 
                 test_df: pd.DataFrame) -> str:
    """Predict ranked cars for a single test case."""
    X_test, test_car_ids = build_test_features(test_df, feature_names)
    
    # Get predictions from each model
    test_scores_dict = {}
    for name, model in models.items():
        test_scores_dict[name] = model.predict(X_test)
    
    # Ensemble
    score_matrix = np.column_stack([test_scores_dict[name] for name in models.keys()])
    ensemble_scores = score_matrix @ weights
    
    # Rank cars
    ranked_indices = np.argsort(-ensemble_scores)
    ranked_cars = [test_car_ids[i] for i in ranked_indices]
    
    return ' | '.join(ranked_cars)


def main():
    parser = argparse.ArgumentParser(description='ACV Refrigerant Leak Localisation Prediction')
    parser.add_argument('--input', '-i', required=True, help='Input directory containing .xlsx files')
    parser.add_argument('--output', '-o', required=True, help='Output CSV file path')
    parser.add_argument('--model-dir', '-m', default='models', help='Model directory')
    
    args = parser.parse_args()
    
    # Load models
    print(f"Loading models from {args.model_dir}...")
    models, weights, feature_names = load_models(args.model_dir)
    print(f"Loaded {len(models)} models")
    
    # Process input files
    input_dir = args.input
    xlsx_files = [f for f in os.listdir(input_dir) if f.endswith('.xlsx')]
    
    if not xlsx_files:
        print(f"No .xlsx files found in {input_dir}")
        sys.exit(1)
    
    results = []
    
    loader = ACVDataLoader()
    
    for filename in sorted(xlsx_files):
        filepath = os.path.join(input_dir, filename)
        print(f"Processing {filename}...")
        
        # Load case
        df = loader.load_case(filepath)
        
        # Predict
        ranked_cars = predict_case(models, weights, feature_names, df)
        
        results.append({
            'file_id': filename,
            'ranked_cars': ranked_cars
        })
        print(f"  Ranked: {ranked_cars}")
    
    # Save predictions
    pred_df = pd.DataFrame(results)
    pred_df.to_csv(args.output, index=False)
    print(f"\nPredictions saved to {args.output}")


if __name__ == "__main__":
    main()