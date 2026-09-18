import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.signal import find_peaks
import rainflow
from typing import Dict, Union

def extract_features(file_or_signal: Union[str, Path, np.ndarray, pd.DataFrame, pd.Series]) -> Dict[str, float]:
    """
    Extract physics-informed, statistical, spectral, and ASTM E1049-85 Rainflow fatigue
    features from a ~580k point 1D dynamic stress time series.
    
    Parameters:
        file_or_signal: Path/str to CSV or 1D numeric array/series.
        
    Returns:
        dict of engineered high-signal tabular features.
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

    signal = signal.astype(np.float64)
    feats: Dict[str, float] = {}

    n_samples = len(signal)
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
    feats['crest_factor'] = (max_val / rms_val) if rms_val > 1e-8 else 0.0
    feats['peak_factor'] = (max(abs(min_val), abs(max_val)) / rms_val) if rms_val > 1e-8 else 0.0

    feats['skewness'] = float(stats.skew(signal))
    feats['kurtosis'] = float(stats.kurtosis(signal))
    feats['mean_abs_diff'] = float(np.mean(np.abs(np.diff(signal)))) if len(signal) > 1 else 0.0
    feats['max_abs_diff'] = float(np.max(np.abs(np.diff(signal)))) if len(signal) > 1 else 0.0

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
    try:
        # Turning point extrema filter (prominence threshold) for fast ASTM rainflow extraction
        prominence = 0.01 * ptp_val if ptp_val > 1e-6 else 1e-4
        peaks, _ = find_peaks(signal, prominence=prominence)
        valleys, _ = find_peaks(-signal, prominence=prominence)

        extrema_idx = np.sort(np.concatenate(([0], peaks, valleys, [len(signal) - 1])))
        extrema_sig = signal[extrema_idx]

        cycles = list(rainflow.count_cycles(extrema_sig))
        if cycles:
            ranges = np.array([c[0] for c in cycles], dtype=np.float64)
            counts = np.array([c[1] for c in cycles], dtype=np.float64)
            amplitudes = ranges / 2.0  # stress amplitude sigma_a = Delta_sigma / 2

            feats['rf_cycle_count'] = float(len(cycles))
            feats['rf_total_cycles'] = float(np.sum(counts))
            feats['rf_max_range'] = float(np.max(ranges))
            feats['rf_mean_range'] = float(np.average(ranges, weights=counts))
            feats['rf_std_range'] = float(np.std(ranges))

            # Amplitude statistics & high percentiles
            feats['amp_mean'] = float(np.average(amplitudes, weights=counts))
            feats['amp_max'] = float(np.max(amplitudes))
            feats['amp_std'] = float(np.std(amplitudes))
            
            amp_p90, amp_p95, amp_p99 = np.percentile(amplitudes, [90, 95, 99])
            feats['amp_p90'] = float(amp_p90)
            feats['amp_p95'] = float(amp_p95)
            feats['amp_p99'] = float(amp_p99)

            # High-amplitude cycle counts exceeding stress thresholds (e.g. 20, 40, 60 MPa)
            feats['cycles_gt_10mpa'] = float(np.sum(counts[ranges > 10.0]))
            feats['cycles_gt_20mpa'] = float(np.sum(counts[ranges > 20.0]))
            feats['cycles_gt_40mpa'] = float(np.sum(counts[ranges > 40.0]))
            feats['cycles_gt_60mpa'] = float(np.sum(counts[ranges > 60.0]))

            # S-N Fatigue damage accumulators: sum(counts * range^m) & sum(counts * amplitude^m)
            for m in [2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
                m_str = str(m).replace('.', '_')
                feats[f'rf_damage_m{m_str}'] = float(np.sum(counts * (ranges ** m)))
                feats[f'amp_damage_m{m_str}'] = float(np.sum(counts * (amplitudes ** m)))

            # Histogram bins of stress ranges
            hist, _ = np.histogram(ranges, bins=10, weights=counts)
            for i, h in enumerate(hist):
                feats[f'rf_bin_{i}'] = float(h)
        else:
            _populate_empty_rf_features(feats)
    except Exception:
        _populate_empty_rf_features(feats)

    # 4. Spectral / Frequency Domain Moments (FFT)
    fft_vals = np.abs(np.fft.rfft(signal))
    feats['fft_mean'] = float(np.mean(fft_vals))
    feats['fft_std'] = float(np.std(fft_vals))
    feats['fft_max'] = float(np.max(fft_vals))
    feats['fft_skew'] = float(stats.skew(fft_vals))
    feats['fft_kurtosis'] = float(stats.kurtosis(fft_vals))

    n_fft = len(fft_vals)
    band_len = max(1, n_fft // 5)
    for b in range(5):
        band_vals = fft_vals[b * band_len : (b + 1) * band_len]
        feats[f'fft_band_{b}_energy'] = float(np.sum(band_vals ** 2)) if len(band_vals) > 0 else 0.0

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
