import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.signal import find_peaks
import rainflow
from typing import Dict, Union

def _get_feature_keys() -> list:
    """Returns the standardized, ordered list of all feature names."""
    base_keys = [
        'n_samples', 'mean', 'std', 'variance', 'min', 'max', 'peak_to_peak', 'rms',
        'crest_factor', 'peak_factor', 'skewness', 'kurtosis', 'mean_abs_diff', 'max_abs_diff',
        'q01', 'q05', 'q25', 'q50', 'q75', 'q90', 'q95', 'q99', 'iqr', 'energy',
        'rf_cycle_count', 'rf_total_cycles', 'rf_max_range', 'rf_mean_range', 'rf_std_range',
        'amp_mean', 'amp_max', 'amp_std', 'amp_p90', 'amp_p95', 'amp_p99',
        'cycles_gt_10mpa', 'cycles_gt_20mpa', 'cycles_gt_40mpa', 'cycles_gt_60mpa'
    ]
    sn_keys = []
    for m in [2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
        m_str = str(m).replace('.', '_')
        sn_keys.append(f'rf_damage_m{m_str}')
        sn_keys.append(f'amp_damage_m{m_str}')
    hist_keys = [f'rf_bin_{i}' for i in range(10)]
    fft_keys = ['fft_mean', 'fft_std', 'fft_max', 'fft_skew', 'fft_kurtosis']
    band_keys = [f'fft_band_{b}_energy' for b in range(5)]
    return base_keys + sn_keys + hist_keys + fft_keys + band_keys

def extract_features(file_or_signal: Union[str, Path, np.ndarray, pd.DataFrame, pd.Series]) -> Dict[str, float]:
    """
    Extract physics-informed, statistical, spectral, and ASTM E1049-85 Rainflow fatigue
    features from a dynamic stress time series.
    
    Hardened against constant/flatline signals, zero-variance, NaNs, and non-finite values.
    """
    if isinstance(file_or_signal, (str, Path)):
        df = pd.read_csv(file_or_signal, header=None)
        signal = df.iloc[:, 0].values
    elif isinstance(file_or_signal, pd.DataFrame):
        signal = file_or_signal.iloc[:, 0].values
    elif isinstance(file_or_signal, pd.Series):
        signal = file_or_signal.values
    else:
        signal = np.array(file_or_signal, dtype=np.float64)

    # Sanitize input array
    signal = np.nan_to_num(np.array(signal, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    
    feats: Dict[str, float] = {}
    n_samples = len(signal)
    all_keys = _get_feature_keys()

    # Edge Case: Empty or zero-length input
    if n_samples == 0:
        return {k: 0.0 for k in all_keys}

    feats['n_samples'] = float(n_samples)

    # 1. Statistical Moments & Amplitudes
    mean_val = float(np.mean(signal))
    std_val = float(np.std(signal))
    var_val = float(np.var(signal))
    min_val = float(np.min(signal))
    max_val = float(np.max(signal))
    ptp_val = max_val - min_val
    rms_val = float(np.sqrt(np.mean(signal ** 2)))

    feats['mean'] = mean_val
    feats['std'] = std_val
    feats['variance'] = var_val
    feats['min'] = min_val
    feats['max'] = max_val
    feats['peak_to_peak'] = ptp_val
    feats['rms'] = rms_val

    # Crest & Peak factors with division-by-zero protection
    feats['crest_factor'] = float(np.nan_to_num((max_val / rms_val) if rms_val > 1e-12 else 0.0, nan=0.0, posinf=0.0, neginf=0.0))
    feats['peak_factor'] = float(np.nan_to_num((max(abs(min_val), abs(max_val)) / rms_val) if rms_val > 1e-12 else 0.0, nan=0.0, posinf=0.0, neginf=0.0))

    # Skewness & Kurtosis with zero-variance protection
    if std_val > 1e-12:
        feats['skewness'] = float(np.nan_to_num(stats.skew(signal), nan=0.0, posinf=0.0, neginf=0.0))
        feats['kurtosis'] = float(np.nan_to_num(stats.kurtosis(signal), nan=0.0, posinf=0.0, neginf=0.0))
    else:
        feats['skewness'] = 0.0
        feats['kurtosis'] = 0.0

    if n_samples > 1:
        diff_arr = np.abs(np.diff(signal))
        feats['mean_abs_diff'] = float(np.nan_to_num(np.mean(diff_arr), nan=0.0, posinf=0.0, neginf=0.0))
        feats['max_abs_diff'] = float(np.nan_to_num(np.max(diff_arr), nan=0.0, posinf=0.0, neginf=0.0))
    else:
        feats['mean_abs_diff'] = 0.0
        feats['max_abs_diff'] = 0.0

    # 2. Percentiles & Signal Energy
    q1, q5, q25, q50, q75, q90, q95, q99 = np.percentile(signal, [1, 5, 25, 50, 75, 90, 95, 99])
    feats['q01'] = float(q1)
    feats['q05'] = float(q5)
    feats['q25'] = float(q25)
    feats['q50'] = float(q50)
    feats['q75'] = float(q75)
    feats['q90'] = float(q90)
    feats['q95'] = float(q95)
    feats['q99'] = float(q99)
    feats['iqr'] = float(q75 - q25)
    feats['energy'] = float(np.sum(signal ** 2))

    # 3. Fatigue / Rainflow Cycle Counting Proxies (ASTM E1049-85)
    # Edge Case Check: If constant signal / zero variance, do NOT run rainflow
    if std_val < 1e-12 or ptp_val < 1e-12:
        _populate_empty_rf_features(feats)
    else:
        try:
            prominence = 0.01 * ptp_val if ptp_val > 1e-6 else 1e-4
            peaks, _ = find_peaks(signal, prominence=prominence)
            valleys, _ = find_peaks(-signal, prominence=prominence)

            extrema_idx = np.sort(np.concatenate(([0], peaks, valleys, [len(signal) - 1])))
            extrema_sig = signal[extrema_idx]

            cycles = list(rainflow.count_cycles(extrema_sig))
            if cycles:
                ranges = np.array([c[0] for c in cycles], dtype=np.float64)
                counts = np.array([c[1] for c in cycles], dtype=np.float64)
                amplitudes = ranges / 2.0

                feats['rf_cycle_count'] = float(len(cycles))
                feats['rf_total_cycles'] = float(np.sum(counts))
                feats['rf_max_range'] = float(np.max(ranges))
                feats['rf_mean_range'] = float(np.nan_to_num(np.average(ranges, weights=counts), nan=0.0))
                feats['rf_std_range'] = float(np.nan_to_num(np.std(ranges), nan=0.0))

                feats['amp_mean'] = float(np.nan_to_num(np.average(amplitudes, weights=counts), nan=0.0))
                feats['amp_max'] = float(np.max(amplitudes))
                feats['amp_std'] = float(np.nan_to_num(np.std(amplitudes), nan=0.0))

                amp_p90, amp_p95, amp_p99 = np.percentile(amplitudes, [90, 95, 99])
                feats['amp_p90'] = float(amp_p90)
                feats['amp_p95'] = float(amp_p95)
                feats['amp_p99'] = float(amp_p99)

                feats['cycles_gt_10mpa'] = float(np.sum(counts[ranges > 10.0]))
                feats['cycles_gt_20mpa'] = float(np.sum(counts[ranges > 20.0]))
                feats['cycles_gt_40mpa'] = float(np.sum(counts[ranges > 40.0]))
                feats['cycles_gt_60mpa'] = float(np.sum(counts[ranges > 60.0]))

                for m in [2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
                    m_str = str(m).replace('.', '_')
                    feats[f'rf_damage_m{m_str}'] = float(np.nan_to_num(np.sum(counts * (ranges ** m)), nan=0.0))
                    feats[f'amp_damage_m{m_str}'] = float(np.nan_to_num(np.sum(counts * (amplitudes ** m)), nan=0.0))

                hist, _ = np.histogram(ranges, bins=10, weights=counts)
                for i, h in enumerate(hist):
                    feats[f'rf_bin_{i}'] = float(h)
            else:
                _populate_empty_rf_features(feats)
        except Exception:
            _populate_empty_rf_features(feats)

    # 4. Spectral / Frequency Domain Moments (FFT)
    if std_val < 1e-12:
        feats['fft_mean'] = 0.0
        feats['fft_std'] = 0.0
        feats['fft_max'] = 0.0
        feats['fft_skew'] = 0.0
        feats['fft_kurtosis'] = 0.0
        for b in range(5):
            feats[f'fft_band_{b}_energy'] = 0.0
    else:
        fft_vals = np.abs(np.fft.rfft(signal))
        feats['fft_mean'] = float(np.mean(fft_vals))
        feats['fft_std'] = float(np.std(fft_vals))
        feats['fft_max'] = float(np.max(fft_vals))
        feats['fft_skew'] = float(np.nan_to_num(stats.skew(fft_vals), nan=0.0))
        feats['fft_kurtosis'] = float(np.nan_to_num(stats.kurtosis(fft_vals), nan=0.0))

        n_fft = len(fft_vals)
        band_len = max(1, n_fft // 5)
        for b in range(5):
            band_vals = fft_vals[b * band_len : (b + 1) * band_len]
            feats[f'fft_band_{b}_energy'] = float(np.sum(band_vals ** 2)) if len(band_vals) > 0 else 0.0

    # Ensure all required keys exist and are finite numbers
    for k in all_keys:
        if k not in feats:
            feats[k] = 0.0
        else:
            feats[k] = float(np.nan_to_num(feats[k], nan=0.0, posinf=0.0, neginf=0.0))

    return feats


def _populate_empty_rf_features(feats: Dict[str, float]):
    feats['rf_cycle_count'] = 0.0
    feats['rf_total_cycles'] = 0.0
    feats['rf_max_range'] = 0.0
    feats['rf_mean_range'] = 0.0
    feats['rf_std_range'] = 0.0
    feats['amp_mean'] = 0.0
    feats['amp_max'] = 0.0
    feats['amp_std'] = 0.0
    feats['amp_p90'] = 0.0
    feats['amp_p95'] = 0.0
    feats['amp_p99'] = 0.0
    feats['cycles_gt_10mpa'] = 0.0
    feats['cycles_gt_20mpa'] = 0.0
    feats['cycles_gt_40mpa'] = 0.0
    feats['cycles_gt_60mpa'] = 0.0
    for m in [2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
        m_str = str(m).replace('.', '_')
        feats[f'rf_damage_m{m_str}'] = 0.0
        feats[f'amp_damage_m{m_str}'] = 0.0
    for i in range(10):
        feats[f'rf_bin_{i}'] = 0.0
