# utils/metrics.py
import numpy as np
import pandas as pd
from typing import List, Tuple

def _to_ns(dt):
    """Convert timestamp to nanoseconds since epoch."""
    if hasattr(dt, 'value'):
        return dt.value
    if isinstance(dt, np.datetime64):
        return dt.astype('datetime64[ns]').astype(np.int64)
    return pd.Timestamp(dt).value

def compute_iou(pred_start, pred_end, true_start, true_end) -> float:
    """Compute Intersection over Union of two time intervals."""
    pred_s = _to_ns(pred_start)
    pred_e = _to_ns(pred_end)
    true_s = _to_ns(true_start)
    true_e = _to_ns(true_end)
    
    intersection = max(0, min(pred_e, true_e) - max(pred_s, true_s))
    union = (pred_e - pred_s) + (true_e - true_s) - intersection
    
    if union <= 0:
        return 0.0
    return intersection / union

def match_predictions_to_truth(pred_segments: List[Tuple[pd.Timestamp, pd.Timestamp, str]],
                                true_segments: List[Tuple[pd.Timestamp, pd.Timestamp, str]]) -> Tuple[List[float], List[bool], List[bool]]:
    """
    Match predicted segments to true segments using greedy IoU matching.
    Returns: (ious, matched_pred, matched_true)
    """
    n_pred = len(pred_segments)
    n_true = len(true_segments)
    
    # Compute all pairwise IoUs for same-label pairs
    iou_matrix = np.zeros((n_pred, n_true))
    for i, (ps, pe, pl) in enumerate(pred_segments):
        for j, (ts, te, tl) in enumerate(true_segments):
            if pl == tl:
                iou_matrix[i, j] = compute_iou(ps, pe, ts, te)
    
    # Greedy matching by highest IoU
    matched_pred = [False] * n_pred
    matched_true = [False] * n_true
    ious = []
    
    # Get all valid pairs sorted by IoU descending
    pairs = []
    for i in range(n_pred):
        for j in range(n_true):
            if iou_matrix[i, j] > 0:
                pairs.append((iou_matrix[i, j], i, j))
    pairs.sort(reverse=True)
    
    for iou, i, j in pairs:
        if not matched_pred[i] and not matched_true[j]:
            matched_pred[i] = True
            matched_true[j] = True
            ious.append(iou)
    
    return ious, matched_pred, matched_true

def compute_soft_f1(pred_segments: List[Tuple[pd.Timestamp, pd.Timestamp, str]],
                     true_segments: List[Tuple[pd.Timestamp, pd.Timestamp, str]]) -> float:
    """Compute IoU-weighted F1 score."""
    ious, matched_pred, matched_true = match_predictions_to_truth(pred_segments, true_segments)
    
    if not true_segments or not pred_segments:
        return 0.0
    
    soft_recall = sum(ious) / len(true_segments)
    soft_precision = sum(ious) / len(pred_segments)
    
    if soft_recall + soft_precision == 0:
        return 0.0
    
    return 2 * soft_recall * soft_precision / (soft_recall + soft_precision)

def evaluate_segmentation(pred_segments: List[Tuple[pd.Timestamp, pd.Timestamp, str]],
                          true_segments: List[Tuple[pd.Timestamp, pd.Timestamp, str]]) -> dict:
    """Evaluate segmentation quality."""
    ious, matched_pred, matched_true = match_predictions_to_truth(pred_segments, true_segments)
    
    return {
        'soft_f1': compute_soft_f1(pred_segments, true_segments),
        'mean_iou': np.mean(ious) if ious else 0.0,
        'matched_ratio': sum(matched_true) / len(true_segments) if true_segments else 0.0,
        'false_positive_rate': (len(pred_segments) - sum(matched_pred)) / len(pred_segments) if pred_segments else 0.0,
        'n_true': len(true_segments),
        'n_pred': len(pred_segments),
        'n_matched': sum(matched_true)
    }