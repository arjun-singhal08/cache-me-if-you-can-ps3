# postprocess/optimize.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from utils.metrics import compute_iou

def merge_adjacent_segments(segments: List[Dict], max_gap_ms: float = 500) -> List[Dict]:
    """Merge adjacent segments with same label and small gap."""
    if not segments:
        return []
    
    merged = [segments[0].copy()]
    for seg in segments[1:]:
        last = merged[-1]
        gap = (seg['start'] - last['end']) / np.timedelta64(1, 'ms')
        
        if gap < max_gap_ms and seg['label'] == last['label']:
            # Merge
            last['end'] = seg['end']
            last['conf'] = max(last['conf'], seg['conf'])
        else:
            merged.append(seg.copy())
    
    return merged

def refine_segment_boundaries(segments: List[Dict], 
                               timestamps: np.ndarray,
                               current: np.ndarray,
                               position: np.ndarray,
                               sample_rate: float = 50.0) -> List[Dict]:
    """
    Refine segment boundaries using current derivative peaks (start) 
    and position settling (end).
    
    NOTE: Disabled by default as segmenter already refines boundaries.
    Re-enable only if it improves validation IoU.
    """
    # Return segments unchanged - segmenter already refines boundaries
    return segments

def nms_overlap(segments: List[Dict], iou_threshold: float = 0.3) -> List[Dict]:
    """
    Non-maximum suppression for overlapping segments.
    Keeps highest confidence prediction for overlapping regions.
    """
    if not segments:
        return []
    
    # Sort by confidence descending
    sorted_segs = sorted(segments, key=lambda x: x['conf'], reverse=True)
    keep = []
    
    for seg in sorted_segs:
        overlap = False
        for kept in keep:
            iou = compute_iou(seg['start'], seg['end'], kept['start'], kept['end'])
            if iou > iou_threshold:
                overlap = True
                break
        if not overlap:
            keep.append(seg)
    
    # Sort back by time
    keep.sort(key=lambda x: x['start'])
    return keep

def filter_by_confidence(segments: List[Dict], threshold: float = 0.35) -> List[Dict]:
    """Remove segments with confidence below threshold."""
    return [s for s in segments if s['conf'] >= threshold]

def filter_by_duration(segments: List[Dict], min_duration_ms: float = 500) -> List[Dict]:
    """Remove segments shorter than minimum duration."""
    filtered = []
    for s in segments:
        dur = (s['end'] - s['start']) / np.timedelta64(1, 'ms')
        if dur >= min_duration_ms:
            filtered.append(s)
    return filtered

def optimize_predictions(raw_segments: List[Tuple[pd.Timestamp, pd.Timestamp, str, float]],
                         timestamps: np.ndarray,
                         current: np.ndarray,
                         position: np.ndarray,
                         config: dict) -> List[Tuple[pd.Timestamp, pd.Timestamp, str]]:
    """
    Full post-processing pipeline for IoU optimization.
    
    Args:
        raw_segments: List of (start, end, label, confidence)
        timestamps: Full timestamp array
        current: Full current signal
        position: Full position signal
        config: Post-processing config dict
    
    Returns:
        Optimized list of (start, end, label)
    """
    # Convert to dict format
    segments = [
        {'start': s, 'end': e, 'label': l, 'conf': c}
        for s, e, l, c in raw_segments
    ]
    
    # 1. Filter by confidence
    segments = filter_by_confidence(segments, config.get('conf_threshold', 0.35))
    
    # 2. Filter by duration
    segments = filter_by_duration(segments, config.get('min_segment_duration_ms', 500))
    
    # 3. Merge adjacent same-label
    segments = merge_adjacent_segments(segments, config.get('merge_gap_ms', 500))
    
    # 4. Refine boundaries
    segments = refine_segment_boundaries(
        segments, timestamps, current, position
    )
    
    # 5. NMS for overlaps
    segments = nms_overlap(segments, config.get('nms_iou_threshold', 0.3))
    
    # Convert back to tuples
    return [(s['start'], s['end'], s['label']) for s in segments]