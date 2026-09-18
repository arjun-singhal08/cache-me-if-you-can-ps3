# train.py
import os
import sys
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.io import load_train_data, parse_datetime
from utils.metrics import evaluate_segmentation, compute_soft_f1
from segmentation.detector import DoorSegmenter
from features.extract import extract_segment_features
from classification.ensemble import DoorFaultEnsemble
from postprocess.optimize import optimize_predictions

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def extract_training_segments(train_df: pd.DataFrame, segments_df: pd.DataFrame):
    """Extract feature vectors and labels for each training segment."""
    features_list = []
    labels = []
    seg_dfs = []
    segment_ids = []
    
    for _, seg in segments_df.iterrows():
        mask = (train_df['dt'] >= seg['start_dt']) & (train_df['dt'] <= seg['end_dt'])
        seg_df = train_df[mask].copy()
        
        if len(seg_df) < 20:  # Skip too short
            continue
        
        feat = extract_segment_features(seg_df, seg['operation'])
        features_list.append(feat)
        labels.append(1 if seg['status'] == 'Abnormal resistance' else 0)
        seg_dfs.append(seg_df)
        segment_ids.append(seg['segment_id'])
    
    # Convert to matrix
    feature_names = sorted(features_list[0].keys())
    X = np.array([[f[name] for name in feature_names] for f in features_list])
    y = np.array(labels)
    
    return X, y, seg_dfs, segment_ids, feature_names

def evaluate_segmenter(segmenter: DoorSegmenter, train_df: pd.DataFrame, segments_df: pd.DataFrame):
    """Evaluate segmentation quality on training data (boundary detection only)."""
    pred_segments = segmenter.predict(train_df)
    
    true_segments = [
        (row['start_dt'], row['end_dt'])
        for _, row in segments_df.iterrows()
    ]
    pred_bounds = [(s, e) for s, e, _ in pred_segments]
    
    # Convert to nanoseconds for comparison
    def to_ns(dt):
        if hasattr(dt, 'value'):
            return dt.value
        return pd.Timestamp(dt).value
    
    # Compute boundary IoU (ignoring labels)
    ious = []
    for ps, pe in pred_bounds:
        best_iou = 0
        ps_ns, pe_ns = to_ns(ps), to_ns(pe)
        for ts, te in true_segments:
            ts_ns, te_ns = to_ns(ts), to_ns(te)
            inter = max(0, min(pe_ns, te_ns) - max(ps_ns, ts_ns))
            union = (pe_ns - ps_ns) + (te_ns - ts_ns) - inter
            if union > 0:
                iou = inter / union
                best_iou = max(best_iou, iou)
        ious.append(best_iou)
    
    # Also check how many true segments are covered
    matched_true = 0
    for ts, te in true_segments:
        ts_ns, te_ns = to_ns(ts), to_ns(te)
        for ps, pe in pred_bounds:
            ps_ns, pe_ns = to_ns(ps), to_ns(pe)
            inter = max(0, min(pe_ns, te_ns) - max(ps_ns, ts_ns))
            union = (pe_ns - ps_ns) + (te_ns - ts_ns) - inter
            if union > 0 and inter / union > 0.3:
                matched_true += 1
                break
    
    results = {
        'mean_iou': np.mean(ious) if ious else 0,
        'max_iou': np.max(ious) if ious else 0,
        'n_pred': len(pred_segments),
        'n_true': len(true_segments),
        'matched_true': matched_true,
        'recall_boundary': matched_true / len(true_segments) if true_segments else 0
    }
    print(f"Segmentation: {results}")
    return results

def main():
    # Paths - train.py is in app/, data is in ../
    base_dir = os.path.dirname(os.path.dirname(__file__))  # Go up to Door/
    data_dir = os.path.join(base_dir)
    app_dir = os.path.join(base_dir, 'app')
    model_dir = os.path.join(app_dir, 'model')
    
    train_path = os.path.join(data_dir, 'Train.csv')
    segments_path = os.path.join(data_dir, 'Train_Segments_Answer.csv')
    config_path = os.path.join(app_dir, 'config.yaml')
    
    # Load config
    config = load_config(config_path)
    
    # Load data
    print("Loading training data...")
    train_df, segments_df = load_train_data(train_path, segments_path)
    print(f"Train: {len(train_df)} rows, {len(segments_df)} segments")
    print(f"Class distribution: {segments_df['status'].value_counts().to_dict()}")
    
    # Train segmenter
    print("\nTraining segmenter...")
    segmenter = DoorSegmenter(config.get('segmentation', {}))
    segmenter.fit(train_df, segments_df)
    
    # Evaluate segmentation
    print("\nEvaluating segmentation...")
    seg_results = evaluate_segmenter(segmenter, train_df, segments_df)
    
    # Extract features from ground truth segments (for classifier training)
    print("\nExtracting features from ground truth segments...")
    X, y, seg_dfs, seg_ids, feature_names = extract_training_segments(train_df, segments_df)
    print(f"Feature matrix: {X.shape}, Labels: {y.sum()}/{len(y)} abnormal")
    print(f"Features: {feature_names}")
    
    # Train classifier
    print("\nTraining classifier ensemble...")
    classifier = DoorFaultEnsemble(config)
    
    # Use segment_ids as groups for GroupKFold (prevent leakage)
    groups = np.array([s.split('_')[1] if '_' in s else s for s in seg_ids])
    # Actually use a simpler grouping - by operation+status pattern
    groups = np.array([f"{segments_df.iloc[i]['operation']}_{segments_df.iloc[i]['status']}" 
                       for i in range(len(seg_ids))])
    
    classifier.fit(X, y, seg_dfs=seg_dfs, groups=groups)
    
    # Evaluate on training segments (using ground truth segments)
    print("\nEvaluating classifier on ground truth segments...")
    train_probas = classifier.predict_proba(X, seg_dfs)
    train_preds = (train_probas >= 0.5).astype(int)
    
    from sklearn.metrics import classification_report, f1_score
    print(classification_report(y, train_preds, target_names=['Normal', 'Abnormal']))
    print(f"Macro F1: {f1_score(y, train_preds, average='macro'):.4f}")
    
    # Full pipeline evaluation on training data
    print("\nFull pipeline evaluation on training data...")
    pred_segments = segmenter.predict(train_df)
    
    # Extract features for predicted segments
    pred_features = []
    pred_seg_dfs = []
    for start, end, op in pred_segments:
        mask = (train_df['dt'] >= start) & (train_df['dt'] <= end)
        seg_df = train_df[mask].copy()
        if len(seg_df) >= 20:
            feat = extract_segment_features(seg_df, op)
            pred_features.append(feat)
            pred_seg_dfs.append(seg_df)
    
    if pred_features:
        feature_names = sorted(pred_features[0].keys())
        X_pred = np.array([[f[name] for name in feature_names] for f in pred_features])
        
        # Predict
        probas = classifier.predict_proba(X_pred, pred_seg_dfs)
        
        # Create raw segments with confidence
        raw_segments = []
        for (start, end, op), p_abn in zip(pred_segments, probas):
            if p_abn > 0.5:
                label = 'Abnormal resistance'
                conf = p_abn
            else:
                label = 'Normal'
                conf = 1 - p_abn
            raw_segments.append((start, end, label, conf))
        
        # Post-process
        timestamps = train_df['dt'].values
        current = train_df['Motor current(mA)'].values
        position = train_df['Door leaf position'].values
        
        final_segments = optimize_predictions(
            raw_segments, timestamps, current, position, config.get('postprocess', {})
        )
        
        # Evaluate full pipeline
        true_segments = [
            (row['start_dt'], row['end_dt'], row['status'])
            for _, row in segments_df.iterrows()
        ]
        
        results = evaluate_segmentation(final_segments, true_segments)
        print(f"\nFull Pipeline Results:")
        for k, v in results.items():
            print(f"  {k}: {v}")
    
    # Save models
    print("\nSaving models...")
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(segmenter, os.path.join(model_dir, 'segmenter.pkl'))
    classifier.save(os.path.join(model_dir, 'classifier.pkl'))
    joblib.dump(feature_names, os.path.join(model_dir, 'feature_names.pkl'))
    
    print("Training complete!")

if __name__ == '__main__':
    main()