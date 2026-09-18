import numpy as np
from scipy.signal import find_peaks
import rainflow
from typing import Dict, Any

def extract_fatigue_features(signal: np.ndarray, ptp_val: float) -> Dict[str, Any]:
    """
    Extract ASTM E1049-85 Rainflow cycle counting & S-N fatigue features.
    """
    feats = {}
    try:
        prominence = 0.01 * ptp_val if ptp_val > 1e-6 else 1e-4
        peaks, _ = find_peaks(signal, prominence=prominence)
        valleys, _ = find_peaks(-signal, prominence=prominence)
        
        extrema_idx = np.sort(np.concatenate(([0], peaks, valleys, [len(signal)-1])))
        extrema_sig = signal[extrema_idx]
        
        cycles = list(rainflow.count_cycles(extrema_sig))
        if cycles:
            ranges = np.array([c[0] for c in cycles])
            counts = np.array([c[1] for c in cycles])
            
            feats['rf_count'] = len(cycles)
            feats['rf_total_cycles'] = float(np.sum(counts))
            feats['rf_max_range'] = float(np.max(ranges))
            feats['rf_mean_range'] = float(np.average(ranges, weights=counts))
            feats['rf_std_range'] = float(np.std(ranges))
            
            # S-N curve damage accumulators: sum(n_i * delta_sigma^m)
            for m in [2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
                m_str = str(m).replace('.', '_')
                feats[f'rf_damage_m{m_str}'] = float(np.sum(counts * (ranges ** m)))
                
            hist, _ = np.histogram(ranges, bins=10, weights=counts)
            for i, h in enumerate(hist):
                feats[f'rf_bin_{i}'] = float(h)
        else:
            feats['rf_count'] = 0
            feats['rf_total_cycles'] = 0.0
            feats['rf_max_range'] = 0.0
            feats['rf_mean_range'] = 0.0
            feats['rf_std_range'] = 0.0
            for m in [2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
                m_str = str(m).replace('.', '_')
                feats[f'rf_damage_m{m_str}'] = 0.0
            for i in range(10):
                feats[f'rf_bin_{i}'] = 0.0
    except Exception:
        feats['rf_count'] = 0
        feats['rf_total_cycles'] = 0.0
        feats['rf_max_range'] = 0.0
        feats['rf_mean_range'] = 0.0
        feats['rf_std_range'] = 0.0
        for m in [2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
            m_str = str(m).replace('.', '_')
            feats[f'rf_damage_m{m_str}'] = 0.0
        for i in range(10):
            feats[f'rf_bin_{i}'] = 0.0
            
    return feats

