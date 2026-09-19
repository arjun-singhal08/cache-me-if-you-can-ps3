#!/usr/bin/env python
"""
ACV Refrigerant Leak Localisation - Prediction Pipeline & API
Usage (CLI): python ACV/predict.py --input <input_file_or_dir> --output <output_csv> [--model-dir <models_path>]
Python API:
    from ACV.predict import predict_acv, predict_acv_with_diagnostics
    pred_df = predict_acv("path/to/acv_test_case.xlsx")
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

import joblib
import numpy as np
import pandas as pd

# Ensure package directory is in sys.path for direct script and submodule execution
ACV_DIR = Path(__file__).resolve().parent
if str(ACV_DIR) not in sys.path:
    sys.path.insert(0, str(ACV_DIR))

try:
    from .data_loader import ACVDataLoader
    from .features import extract_car_features
    from .models import (
        LightGBMRanker, XGBoostRanker, ClassicalEnsemble, 
        UnsupervisedAnomalyDetector, SiameseRanker
    )
except ImportError:
    from data_loader import ACVDataLoader
    from features import extract_car_features
    from models import (
        LightGBMRanker, XGBoostRanker, ClassicalEnsemble, 
        UnsupervisedAnomalyDetector, SiameseRanker
    )

DEFAULT_MODEL_DIR = ACV_DIR / "models"


def load_models(
    model_dir: Optional[Union[str, Path]] = None
) -> Tuple[Dict[str, Any], np.ndarray, List[str]]:
    """Load all trained ACV models and ensemble metadata.
    
    Args:
        model_dir: Directory containing trained model .pkl files.
                   Defaults to ACV/models relative to this file.
    
    Returns:
        models: Dictionary of loaded model instances.
        weights: Ensemble weights array.
        feature_names: List of expected feature names in order.
    """
    if model_dir is None:
        model_dir = DEFAULT_MODEL_DIR
    else:
        model_dir = Path(model_dir).resolve()
        
    if not model_dir.exists():
        raise FileNotFoundError(f"Required ACV model directory not found: {model_dir}")

    meta_path = model_dir / 'ensemble_metadata.pkl'
    if not meta_path.exists():
        raise FileNotFoundError(f"Required ACV model artifact not found: {meta_path}")

    try:
        metadata = joblib.load(meta_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load ACV ensemble metadata from {meta_path}: {e}")

    weights = metadata.get('weights')
    feature_names = metadata.get('feature_names')
    model_names = metadata.get('model_names', ['lgbm', 'xgb', 'classical', 'unsup', 'siamese'])

    if weights is None or feature_names is None:
        raise ValueError(f"Corrupted ACV metadata artifact at {meta_path}: missing weights or feature_names.")

    models = {}
    for name in model_names:
        model_path = model_dir / f'{name}_model.pkl'
        if not model_path.exists():
            raise FileNotFoundError(f"Required ACV model artifact not found: {model_path}")
            
        try:
            if name == 'lgbm':
                models[name] = LightGBMRanker.load(str(model_path))
            elif name == 'xgb':
                models[name] = XGBoostRanker.load(str(model_path))
            elif name == 'classical':
                models[name] = ClassicalEnsemble.load(str(model_path))
            elif name == 'unsup':
                models[name] = UnsupervisedAnomalyDetector.load(str(model_path))
            elif name == 'siamese':
                models[name] = SiameseRanker.load(str(model_path))
            else:
                models[name] = joblib.load(str(model_path))
        except Exception as e:
            raise RuntimeError(f"Error loading ACV model artifact '{name}' from {model_path}: {e}")

    return models, weights, feature_names


def parse_car_dataframes(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Parse a dataframe into per-car telemetry dataframes with schema validation.
    
    Args:
        df: Raw DataFrame containing train telemetry.
        
    Returns:
        Dictionary mapping normalized two-digit car IDs (e.g. '01') to car-specific DataFrames.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df)}")
        
    if df.empty:
        raise ValueError("Provided telemetry DataFrame is empty.")

    car_params: Dict[str, List[str]] = {}
    for col in df.columns:
        if str(col).startswith('Car ') and ' - ' in str(col):
            cid, param = str(col).split(' - ', 1)
            cid = cid.replace('Car ', '').strip()
            # Normalize single digits (e.g. '1' -> '01')
            if cid.isdigit() and len(cid) == 1:
                cid = f"{int(cid):02d}"
            if cid not in car_params:
                car_params[cid] = []
            car_params[cid].append(param)
    
    if len(car_params) < 2:
        found_cars = list(car_params.keys())
        raise ValueError(
            f"Expected multi-car telemetry (typically 8 cars, minimum 2 required), "
            f"found {len(found_cars)} cars ({found_cars}). Columns: {list(df.columns[:10])}..."
        )

    # Validate Time column
    if 'Time' not in df.columns:
        time_cols = [c for c in df.columns if str(c).strip().lower() == 'time']
        if time_cols:
            df = df.rename(columns={time_cols[0]: 'Time'})
        else:
            raise ValueError("Required telemetry column 'Time' is missing from workbook.")

    id_cols = [c for c in ['Car model', 'Train number', 'Time'] if c in df.columns]
    if 'Time' not in id_cols:
        id_cols.append('Time')

    car_dfs = {}
    for car_id, params in car_params.items():
        car_cols = id_cols + [
            f'Car {car_id} - {p}' if f'Car {car_id} - {p}' in df.columns else f'Car {int(car_id)} - {p}'
            for p in params
        ]
        valid_cols = [c for c in car_cols if c in df.columns]
        car_df = df[valid_cols].copy()
        
        rename_map = {}
        for c in valid_cols:
            if c not in id_cols and ' - ' in c:
                rename_map[c] = c.split(' - ', 1)[1]
        car_df = car_df.rename(columns=rename_map)
        car_dfs[car_id] = car_df
    
    return car_dfs


def build_test_features(
    test_df: pd.DataFrame, 
    feature_names: List[str]
) -> Tuple[np.ndarray, List[str], Dict[str, pd.DataFrame]]:
    """Build feature matrix for test data aligned with model expectations.
    
    Args:
        test_df: DataFrame containing ACV test case telemetry.
        feature_names: Ordered list of feature names expected by the model ensemble.
        
    Returns:
        X_test: 2D numpy array of shape (n_cars, n_features).
        test_car_ids: Ordered list of car IDs evaluated.
        car_dfs: Dictionary of per-car DataFrames.
    """
    car_dfs = parse_car_dataframes(test_df)
    
    test_features = []
    test_car_ids = []
    
    for car_id in sorted(car_dfs.keys()):
        car_df = car_dfs[car_id]
        feats = extract_car_features(car_df, car_id, car_dfs)
        test_features.append(feats)
        test_car_ids.append(car_id)
    
    test_feat_df = pd.DataFrame(test_features)
    
    # Align columns with training feature set (reindex is clean, fast, and defragmented)
    test_feat_df = test_feat_df.reindex(columns=feature_names, fill_value=0.0)
    test_feat_df = test_feat_df.fillna(0.0).replace([np.inf, -np.inf], 0.0)
    
    return test_feat_df.values, test_car_ids, car_dfs


def predict_case(
    models: Dict[str, Any], 
    weights: np.ndarray, 
    feature_names: List[str], 
    test_df: pd.DataFrame
) -> Dict[str, Any]:
    """Predict ranked cars and extract full diagnostic information for a single case.
    
    Args:
        models: Dict of loaded model instances.
        weights: Model weights array.
        feature_names: Expected feature names.
        test_df: DataFrame of test telemetry.
        
    Returns:
        Dict containing ranked string, ranked list, scores, and diagnostics.
    """
    X_test, test_car_ids, car_dfs = build_test_features(test_df, feature_names)
    
    # Get predictions from each model
    test_scores_dict = {}
    for name, model in models.items():
        test_scores_dict[name] = model.predict(X_test)
    
    # Ensemble
    score_matrix = np.column_stack([test_scores_dict[name] for name in models.keys()])
    ensemble_scores = score_matrix @ weights
    
    # Rank cars descending (highest score = most likely faulty)
    ranked_indices = np.argsort(-ensemble_scores)
    ranked_cars = [test_car_ids[i] for i in ranked_indices]
    ranked_scores = [float(ensemble_scores[i]) for i in ranked_indices]
    
    # Format string: 01|04|03|05|07|06|08|02 (strictly no spaces around |)
    ranked_string = '|'.join(ranked_cars)

    # Compute genuine per-car diagnostic summaries
    car_diagnostics = {}
    for idx, cid in enumerate(test_car_ids):
        car_df = car_dfs[cid]
        diag: Dict[str, Any] = {
            'car_id': cid,
            'ensemble_score': float(ensemble_scores[idx]),
            'rank': int(ranked_cars.index(cid) + 1),
            'model_scores': {m: float(test_scores_dict[m][idx]) for m in models.keys()}
        }
        # Compute indoor temperature stats if available
        temp_cols = [c for c in car_df.columns if 'indoor' in str(c).lower() or ('temp' in str(c).lower() and 'set' not in str(c).lower())]
        if temp_cols:
            col_series = pd.to_numeric(car_df[temp_cols[0]], errors='coerce')
            diag['indoor_temp_mean'] = float(col_series.mean()) if not col_series.isna().all() else 0.0
            diag['indoor_temp_min'] = float(col_series.min()) if not col_series.isna().all() else 0.0
            diag['indoor_temp_max'] = float(col_series.max()) if not col_series.isna().all() else 0.0
        
        # Compute Delta-T (cooling capacity proxy) if indoor and outdoor temps exist
        outdoor_cols = [c for c in car_df.columns if 'outdoor' in str(c).lower()]
        if temp_cols and outdoor_cols:
            tin = pd.to_numeric(car_df[temp_cols[0]], errors='coerce')
            tout = pd.to_numeric(car_df[outdoor_cols[0]], errors='coerce')
            dt = tin - tout
            diag['delta_t_mean'] = float(dt.mean()) if not dt.isna().all() else 0.0

        car_diagnostics[cid] = diag

    return {
        'ranked_string': ranked_string,
        'ranked_cars': ranked_cars,
        'ranked_scores': ranked_scores,
        'ensemble_scores': {cid: float(score) for cid, score in zip(test_car_ids, ensemble_scores)},
        'per_model_scores': test_scores_dict,
        'car_diagnostics': car_diagnostics,
        'test_car_ids': test_car_ids,
        'car_dfs': car_dfs,
        'weights': weights
    }


def predict_acv(
    input_path: Union[str, Path, pd.DataFrame, Any],
    model_dir: Optional[Union[str, Path]] = None,
    filename: Optional[str] = None
) -> pd.DataFrame:
    """Predict ACV refrigerant leak localisation returning the required submission DataFrame.
    
    Args:
        input_path: Path to .xlsx file, directory of .xlsx files, pd.DataFrame, or file-like object.
        model_dir: Path to ACV models directory. Defaults to ACV/models.
        filename: Optional filename override for single file or DataFrame inputs.
        
    Returns:
        pd.DataFrame with exactly ['file_id', 'ranked_cars'].
    """
    res_df, _ = predict_acv_with_diagnostics(input_path, model_dir=model_dir, filename=filename)
    return res_df


def predict_acv_with_diagnostics(
    input_path: Union[str, Path, pd.DataFrame, Any],
    model_dir: Optional[Union[str, Path]] = None,
    filename: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Predict ACV refrigerant leak localisation and return rich diagnostic evidence.
    
    Args:
        input_path: Path to .xlsx file, directory of .xlsx files, pd.DataFrame, or file-like object.
        model_dir: Path to ACV models directory. Defaults to ACV/models.
        filename: Optional filename override for single file or DataFrame inputs.
        
    Returns:
        Tuple of:
          - pd.DataFrame with exactly ['file_id', 'ranked_cars']
          - Dict containing diagnostic data for UI/reporting
    """
    models, weights, feature_names = load_models(model_dir)
    loader = ACVDataLoader()

    results = []
    diagnostics = {}

    if isinstance(input_path, pd.DataFrame):
        file_id = filename or "uploaded_case.xlsx"
        case_res = predict_case(models, weights, feature_names, input_path)
        results.append({
            'file_id': file_id,
            'ranked_cars': case_res['ranked_string']
        })
        diagnostics[file_id] = case_res
    elif hasattr(input_path, 'read'):
        # Uploaded file / BytesIO object from Streamlit
        file_id = filename or getattr(input_path, 'name', 'uploaded_case.xlsx')
        df = loader.load_case(input_path)
        case_res = predict_case(models, weights, feature_names, df)
        results.append({
            'file_id': file_id,
            'ranked_cars': case_res['ranked_string']
        })
        diagnostics[file_id] = case_res
    else:
        inp = Path(input_path).resolve()
        if not inp.exists():
            raise FileNotFoundError(f"Input path does not exist: {inp}")

        if inp.is_file():
            file_id = filename or inp.name
            df = loader.load_case(inp)
            case_res = predict_case(models, weights, feature_names, df)
            results.append({
                'file_id': file_id,
                'ranked_cars': case_res['ranked_string']
            })
            diagnostics[file_id] = case_res
        elif inp.is_dir():
            xlsx_files = sorted(list(inp.glob("*.xlsx")) + list(inp.glob("*.parquet")))
            if not xlsx_files:
                raise FileNotFoundError(f"No .xlsx or .parquet files found in directory: {inp}")
            for fp in xlsx_files:
                df = loader.load_case(fp)
                case_res = predict_case(models, weights, feature_names, df)
                results.append({
                    'file_id': fp.name,
                    'ranked_cars': case_res['ranked_string']
                })
                diagnostics[fp.name] = case_res

    pred_df = pd.DataFrame(results)[['file_id', 'ranked_cars']]
    return pred_df, diagnostics


def main():
    parser = argparse.ArgumentParser(description='ACV Refrigerant Leak Localisation Prediction')
    parser.add_argument('--input', '-i', required=True, help='Input .xlsx file or directory containing .xlsx files')
    parser.add_argument('--output', '-o', required=True, help='Output CSV file path')
    parser.add_argument('--model-dir', '-m', default=None, help='Model directory (defaults to ACV/models)')
    
    args = parser.parse_args()
    
    print(f"Running ACV prediction on: {args.input}")
    print(f"Model directory: {args.model_dir or DEFAULT_MODEL_DIR}")
    
    pred_df, diagnostics = predict_acv_with_diagnostics(args.input, model_dir=args.model_dir)
    
    for _, row in pred_df.iterrows():
        print(f"Processed: {row['file_id']} -> Ranked: {row['ranked_cars']}")
    
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_path, index=False)
    print(f"\n[SUCCESS] ACV predictions saved to {out_path}")


if __name__ == "__main__":
    main()