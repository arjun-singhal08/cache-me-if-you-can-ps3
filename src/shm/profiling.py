import numpy as np
from typing import Dict, Any

def profile_signal(signal: np.ndarray) -> Dict[str, Any]:
    """
    Generates an engineering profile overview of the raw dynamic stress time-series.
    """
    mean_val = float(np.mean(signal))
    std_val = float(np.std(signal))
    min_val = float(np.min(signal))
    max_val = float(np.max(signal))
    ptp_val = max_val - min_val
    rms_val = float(np.sqrt(np.mean(signal ** 2)))
    crest_factor = (max_val / rms_val) if rms_val > 1e-8 else 0.0
    
    q1, q25, q50, q75, q99 = np.percentile(signal, [1, 25, 50, 75, 99])
    
    return {
        'n_samples': len(signal),
        'mean_stress': mean_val,
        'std_stress': std_val,
        'min_stress': min_val,
        'max_stress': max_val,
        'peak_to_peak': ptp_val,
        'rms_stress': rms_val,
        'crest_factor': crest_factor,
        'median_stress': float(q50),
        'iqr': float(q75 - q25),
        'q01': float(q1),
        'q99': float(q99)
    }

