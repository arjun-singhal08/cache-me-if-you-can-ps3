import sys
import numpy as np
import pandas as pd
sys.path.insert(0, r'C:\Users\Arjun Singhal\NebulaX-Hackathon-ProblemStatement\PS3\02_Datasets\Door\app')
from utils.io import load_train_data
from utils.metrics import evaluate_segmentation, compute_soft_f1
from segmentation.detector import DoorSegmenter
from features.extract import extract_segment_features
from classification.ensemble import DoorFaultEnsemble
from postprocess.optimize import optimize_predictions
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold

train, segments = load_train_data(
    r'C:\Users\Arjun Singhal\NebulaX-Hackathon-ProblemStatement\PS3\02_Datasets\Door\Train.csv',
    r'C:\Users\Arjun Singhal\NebulaX-Hackathon-ProblemStatement\PS3\02_Datasets\Door\Train_Segments_Answer.csv'
)

config = {
    'segmentation': {
        'min_segment_len': 80, 'max_segment_len': 350,
        'refinement_window_ms': 500, 'min_gap_between_segments': 100,
        'position_threshold': 100
    },
    'postprocess': {
        'conf_threshold': 0.35, 'merge_gap_ms': 500,
        'nms_iou_threshold': 0.3, 'min_segment_duration_ms': 500
    }
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []

for fold, (seg_train_idx, seg_val_idx) in enumerate(skf.split(segments, segments['status'])):
    train_seg_df = segments.iloc[seg_train_idx].reset_index(drop=True)
    val_seg_df = segments.iloc[seg_val_idx].reset_index(drop=True)
    
    segmenter = DoorSegmenter(config['segmentation'])
    segmenter.fit(train, train_seg_df)
    
    X_train_list = []
    y_train = []
    train_seg_dfs = []
    for _, seg in train_seg_df.iterrows():
        mask = (train['dt'] >= seg['start_dt']) & (train['dt'] <= seg['end_dt'])
        seg_df = train[mask]
        if len(seg_df) >= 20:
            feat = extract_segment_features(seg_df, seg['operation'])
            X_train_list.append(feat)
            y_train.append(1 if seg['status'] == 'Abnormal resistance' else 0)
            train_seg_dfs.append(seg_df)
    
    feature_names = sorted(X_train_list[0].keys())
    X_train = np.array([[f[name] for name in feature_names] for f in X_train_list])
    y_train = np.array(y_train)
    
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, 
                        scale_pos_weight=2.67, subsample=0.8, colsample_bytree=0.8,
                        eval_metric='auc', random_state=42, n_jobs=-1, verbosity=0)
    xgb.fit(X_train_scaled, y_train)
    
    X_val_list = []
    y_val = []
    val_seg_dfs = []
    for _, seg in val_seg_df.iterrows():
        mask = (train['dt'] >= seg['start_dt']) & (train['dt'] <= seg['end_dt'])
        seg_df = train[mask]
        if len(seg_df) >= 20:
            feat = extract_segment_features(seg_df, seg['operation'])
            X_val_list.append(feat)
            y_val.append(1 if seg['status'] == 'Abnormal resistance' else 0)
            val_seg_dfs.append(seg_df)
    
    X_val = np.array([[f[name] for name in feature_names] for f in X_val_list])
    y_val = np.array(y_val)
    X_val_scaled = scaler.transform(X_val)
    val_probas = xgb.predict_proba(X_val_scaled)[:, 1]
    
    raw_segments = []
    for (_, seg), p_abn in zip(val_seg_df.iterrows(), val_probas):
        if p_abn > 0.5:
            label = 'Abnormal resistance'
            conf = float(p_abn)
        else:
            label = 'Normal'
            conf = float(1 - p_abn)
        raw_segments.append((seg['start_dt'], seg['end_dt'], label, conf))
    
    timestamps = train['dt'].values
    current = train['Motor current(mA)'].values
    position = train['Door leaf position'].values
    
    final_segments = optimize_predictions(
        raw_segments, timestamps, current, position, config['postprocess']
    )
    
    true_segments = [(row['start_dt'], row['end_dt'], row['status']) for _, row in val_seg_df.iterrows()]
    
    results = evaluate_segmentation(final_segments, true_segments)
    fold_scores.append(results['soft_f1'])
    print('Fold {}: soft_f1 = {:.4f}, n_pred={}, n_true={}'.format(fold, results['soft_f1'], results['n_pred'], results['n_true']))

print('CV soft_f1: {:.4f} (+/- {:.4f})'.format(np.mean(fold_scores), np.std(fold_scores)))