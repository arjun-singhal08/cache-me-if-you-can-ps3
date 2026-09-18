import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks
import rainflow

def extract_features(file_or_signal) -> dict:
    """
    Extract physics-informed, statistical, spectral, and Rainflow fatigue features
    from an SHM dynamic stress time series file or 1D array.
    
    Parameters:
        file_or_signal: str (filepath) or np.ndarray / pd.Series (1D signal)
        
    Returns:
        dict of engineered features
    """
    if isinstance(file_or_signal, str):
        df_sig = pd.read_csv(file_or_signal, header=None)
        signal = df_sig.iloc[:, 0].values
    elif isinstance(file_or_signal, pd.DataFrame):
        signal = file_or_signal.iloc[:, 0].values
    elif isinstance(file_or_signal, pd.Series):
        signal = file_or_signal.values
    else:
        signal = np.array(file_or_signal, dtype=np.float64)
        
    signal = signal.astype(np.float64)
    feats = {}
    
    n_samples = len(signal)
    feats['n_samples'] = n_samples
    
    # 1. Time-Domain / Statistical Moments
    mean_val = float(np.mean(signal))
    std_val = float(np.std(signal))
    min_val = float(np.min(signal))
    max_val = float(np.max(signal))
    ptp_val = max_val - min_val
    rms_val = float(np.sqrt(np.mean(signal ** 2)))
    
    feats['mean'] = mean_val
    feats['std'] = std_val
    feats['min'] = min_val
    feats['max'] = max_val
    feats['ptp'] = ptp_val
    feats['rms'] = rms_val
    feats['crest_factor'] = (max_val / rms_val) if rms_val > 1e-8 else 0.0
    feats['peak_factor'] = (max(abs(min_val), abs(max_val)) / rms_val) if rms_val > 1e-8 else 0.0
    
    feats['skewness'] = float(stats.skew(signal))
    feats['kurtosis'] = float(stats.kurtosis(signal))
    feats['mad'] = float(np.mean(np.abs(signal - mean_val)))
    
    # Quantiles & Percentiles
    q1, q5, q25, q50, q75, q95, q99 = np.percentile(signal, [1, 5, 25, 50, 75, 95, 99])
    feats['q01'] = float(q1)
    feats['q05'] = float(q5)
    feats['q25'] = float(q25)
    feats['q50'] = float(q50)
    feats['q75'] = float(q75)
    feats['q95'] = float(q95)
    feats['q99'] = float(q99)
    feats['iqr'] = float(q75 - q25)
    feats['energy'] = float(np.sum(signal ** 2))
    
    # 2. Fast Rainflow Cycle Counting (ASTM E1049-85) & S-N Fatigue Accumulators
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
    except Exception as e:
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

    # 3. Frequency-Domain / Spectral Features (FFT)
    fft_vals = np.abs(np.fft.rfft(signal))
    feats['fft_mean'] = float(np.mean(fft_vals))
    feats['fft_std'] = float(np.std(fft_vals))
    feats['fft_max'] = float(np.max(fft_vals))
    feats['fft_skew'] = float(stats.skew(fft_vals))
    feats['fft_kurtosis'] = float(stats.kurtosis(fft_vals))
    
    n_fft = len(fft_vals)
    band_len = max(1, n_fft // 5)
    for b in range(5):
        band_vals = fft_vals[b*band_len : (b+1)*band_len]
        feats[f'fft_band_{b}_energy'] = float(np.sum(band_vals ** 2)) if len(band_vals) > 0 else 0.0

    return feats

