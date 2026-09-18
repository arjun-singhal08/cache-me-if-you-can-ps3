import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats, signal
from typing import Dict, Union, List, Optional

def _compute_channel_moments(data: np.ndarray, prefix: str) -> Dict[str, float]:
    """
    Compute time-domain moments, amplitudes, and shape indicators for a 2D array (time, channels)
    or 1D array.
    """
    feats = {}
    if data.size == 0:
        for metric in ['mean', 'std', 'var', 'rms', 'ptp', 'skew', 'kurt', 'crest_factor', 'shape_factor', 'impulse_factor']:
            feats[f'{prefix}_{metric}'] = 0.0
        return feats

    # Flatten across time & channels for aggregated stats, or column-wise mean
    flat = np.nan_to_num(data.ravel(), nan=0.0, posinf=0.0, neginf=0.0)
    std_val = float(np.std(flat))
    rms_val = float(np.sqrt(np.mean(flat ** 2)))
    mean_abs = float(np.mean(np.abs(flat)))
    max_abs = float(np.max(np.abs(flat))) if len(flat) > 0 else 0.0
    ptp_val = float(np.ptp(flat)) if len(flat) > 0 else 0.0

    feats[f'{prefix}_mean'] = float(np.mean(flat))
    feats[f'{prefix}_std'] = std_val
    feats[f'{prefix}_var'] = float(np.var(flat))
    feats[f'{prefix}_rms'] = rms_val
    feats[f'{prefix}_ptp'] = ptp_val

    # Statistical higher moments
    if std_val > 1e-12:
        feats[f'{prefix}_skew'] = float(np.nan_to_num(stats.skew(flat), nan=0.0))
        feats[f'{prefix}_kurt'] = float(np.nan_to_num(stats.kurtosis(flat), nan=0.0))
    else:
        feats[f'{prefix}_skew'] = 0.0
        feats[f'{prefix}_kurt'] = 0.0

    # Dimensionless shape factors
    feats[f'{prefix}_crest_factor'] = float(np.nan_to_num(max_abs / (rms_val + 1e-12), nan=0.0))
    feats[f'{prefix}_shape_factor'] = float(np.nan_to_num(rms_val / (mean_abs + 1e-12), nan=0.0))
    feats[f'{prefix}_impulse_factor'] = float(np.nan_to_num(max_abs / (mean_abs + 1e-12), nan=0.0))

    return feats


def extract_features(df_or_path: Union[str, Path, pd.DataFrame]) -> Dict[str, float]:
    """
    Extract physics-grounded vibration, shock, bilateral asymmetry, and spectral features
    from 129-channel recording (1s duration, 10,000 Hz sampling rate).
    
    Hardened against NaNs, infinities, and zero-variance edge cases.
    """
    if isinstance(df_or_path, (str, Path)):
        df = pd.read_csv(df_or_path)
    else:
        df = df_or_path.copy()

    # Sanitize dataframe
    if df.isna().any().any() or np.isinf(df.values).any():
        df = df.fillna(0.0).replace([np.inf, -np.inf], 0.0)

    feats: Dict[str, float] = {}
    fs = 10000.0  # 10 kHz sampling rate

    # 1. Operating Condition (Rotating Speed)
    speed_col = df.columns[0]
    speed_vals = np.nan_to_num(df[speed_col].values.astype(np.float64), nan=0.0)
    feats['speed_mean'] = float(np.mean(speed_vals))
    feats['speed_std'] = float(np.std(speed_vals))
    feats['speed_max'] = float(np.max(speed_vals)) if len(speed_vals) > 0 else 0.0
    # Pulse frequency (0->1 transitions per second)
    if len(speed_vals) > 1:
        pulse_transitions = np.sum((speed_vals[:-1] == 0) & (speed_vals[1:] == 1))
        feats['speed_pulse_freq'] = float(pulse_transitions)
    else:
        feats['speed_pulse_freq'] = 0.0

    # 2. Identify Side I (Left) and Side II (Right) Channels
    # Positions 1, 3, 5, 7 -> Side I (Left)
    # Positions 2, 4, 6, 8 -> Side II (Right)
    side1_pos = [1, 3, 5, 7]
    side2_pos = [2, 4, 6, 8]

    s1_vib_cols = [c for c in df.columns if 'Vibration' in c and any(f'position {p}' in c for p in side1_pos)]
    s2_vib_cols = [c for c in df.columns if 'Vibration' in c and any(f'position {p}' in c for p in side2_pos)]
    s1_shk_cols = [c for c in df.columns if 'Shock' in c and any(f'position {p}' in c for p in side1_pos)]
    s2_shk_cols = [c for c in df.columns if 'Shock' in c and any(f'position {p}' in c for p in side2_pos)]

    s1_vib_data = df[s1_vib_cols].values if s1_vib_cols else np.zeros((len(df), 1))
    s2_vib_data = df[s2_vib_cols].values if s2_vib_cols else np.zeros((len(df), 1))
    s1_shk_data = df[s1_shk_cols].values if s1_shk_cols else np.zeros((len(df), 1))
    s2_shk_data = df[s2_shk_cols].values if s2_shk_cols else np.zeros((len(df), 1))

    # 3. Global Left/Right Time-Domain Moments
    feats.update(_compute_channel_moments(s1_vib_data, 's1_vib'))
    feats.update(_compute_channel_moments(s2_vib_data, 's2_vib'))
    feats.update(_compute_channel_moments(s1_shk_data, 's1_shk'))
    feats.update(_compute_channel_moments(s2_shk_data, 's2_shk'))

    # Quantiles for Side 1 vs Side 2
    s1_flat = np.nan_to_num(s1_vib_data.ravel(), nan=0.0)
    s2_flat = np.nan_to_num(s2_vib_data.ravel(), nan=0.0)
    s1_q95, s1_q99 = np.percentile(s1_flat, [95, 99]) if len(s1_flat) > 0 else (0.0, 0.0)
    s2_q95, s2_q99 = np.percentile(s2_flat, [95, 99]) if len(s2_flat) > 0 else (0.0, 0.0)
    feats['s1_vib_q95'] = float(s1_q95)
    feats['s1_vib_q99'] = float(s1_q99)
    feats['s2_vib_q95'] = float(s2_q95)
    feats['s2_vib_q99'] = float(s2_q99)
    feats['asym_vib_q95_diff'] = float(s1_q95 - s2_q95)
    feats['asym_vib_q99_diff'] = float(s1_q99 - s2_q99)

    # 4. Bilateral Asymmetry Features
    s1_vib_rms = feats['s1_vib_rms']
    s2_vib_rms = feats['s2_vib_rms']
    s1_shk_rms = feats['s1_shk_rms']
    s2_shk_rms = feats['s2_shk_rms']

    # Energy and RMS ratios
    feats['asym_vib_rms_ratio'] = float(np.nan_to_num(s1_vib_rms / (s2_vib_rms + 1e-12), nan=1.0))
    feats['asym_vib_energy_ratio'] = float(np.nan_to_num((s1_vib_rms ** 2) / ((s2_vib_rms ** 2) + 1e-12), nan=1.0))
    feats['asym_vib_rms_diff'] = float(s1_vib_rms - s2_vib_rms)
    feats['asym_vib_ptp_diff'] = float(feats['s1_vib_ptp'] - feats['s2_vib_ptp'])
    feats['asym_vib_ptp_ratio'] = float(np.nan_to_num(feats['s1_vib_ptp'] / (feats['s2_vib_ptp'] + 1e-12), nan=1.0))

    feats['asym_shk_rms_ratio'] = float(np.nan_to_num(s1_shk_rms / (s2_shk_rms + 1e-12), nan=1.0))
    feats['asym_shk_energy_ratio'] = float(np.nan_to_num((s1_shk_rms ** 2) / ((s2_shk_rms ** 2) + 1e-12), nan=1.0))
    feats['asym_shk_rms_diff'] = float(s1_shk_rms - s2_shk_rms)
    feats['asym_shk_ptp_diff'] = float(feats['s1_shk_ptp'] - feats['s2_shk_ptp'])
    feats['asym_shk_ptp_ratio'] = float(np.nan_to_num(feats['s1_shk_ptp'] / (feats['s2_shk_ptp'] + 1e-12), nan=1.0))

    # Normalized Asymmetry Index: (S1 - S2) / (S1 + S2 + eps) in range [-1, +1]
    feats['norm_asym_vib_index'] = float(np.nan_to_num((s1_vib_rms - s2_vib_rms) / (s1_vib_rms + s2_vib_rms + 1e-12), nan=0.0))
    feats['norm_asym_shk_index'] = float(np.nan_to_num((s1_shk_rms - s2_shk_rms) / (s1_shk_rms + s2_shk_rms + 1e-12), nan=0.0))

    # 5. Per-Car Localized Asymmetry & Pairwise Cross-Correlations
    car_asym_ratios = []
    car_asym_diffs = []
    correlations = []

    for car_idx in range(1, 9):
        # Channels for this car
        car_s1_cols = [c for c in df.columns if f'car {car_idx}' in c and 'Vibration' in c and any(f'position {p}' in c for p in side1_pos)]
        car_s2_cols = [c for c in df.columns if f'car {car_idx}' in c and 'Vibration' in c and any(f'position {p}' in c for p in side2_pos)]
        car_s1_shk_cols = [c for c in df.columns if f'car {car_idx}' in c and 'Shock' in c and any(f'position {p}' in c for p in side1_pos)]
        car_s2_shk_cols = [c for c in df.columns if f'car {car_idx}' in c and 'Shock' in c and any(f'position {p}' in c for p in side2_pos)]

        if car_s1_cols and car_s2_cols:
            c_s1 = df[car_s1_cols].values
            c_s2 = df[car_s2_cols].values
            c_s1_rms = float(np.sqrt(np.mean(c_s1 ** 2)))
            c_s2_rms = float(np.sqrt(np.mean(c_s2 ** 2)))

            ratio = c_s1_rms / (c_s2_rms + 1e-12)
            car_asym_ratios.append(ratio)
            car_asym_diffs.append(c_s1_rms - c_s2_rms)

            feats[f'car{car_idx}_vib_s1_rms'] = c_s1_rms
            feats[f'car{car_idx}_vib_s2_rms'] = c_s2_rms
            feats[f'car{car_idx}_vib_ratio'] = float(ratio)

            if car_s1_shk_cols and car_s2_cols:
                shk_s1 = df[car_s1_shk_cols].values
                shk_s2 = df[car_s2_shk_cols].values
                shk_s1_rms = float(np.sqrt(np.mean(shk_s1 ** 2)))
                shk_s2_rms = float(np.sqrt(np.mean(shk_s2 ** 2)))
                feats[f'car{car_idx}_shk_s1_rms'] = shk_s1_rms
                feats[f'car{car_idx}_shk_s2_rms'] = shk_s2_rms
                feats[f'car{car_idx}_shk_ratio'] = float(np.nan_to_num(shk_s1_rms / (shk_s2_rms + 1e-12), nan=1.0))

            # Pairwise cross-correlations (Pos 1 vs 2, 3 vs 4, 5 vs 6, 7 vs 8)
            for p1, p2 in [(1, 2), (3, 4), (5, 6), (7, 8)]:
                col1 = [c for c in car_s1_cols if f'position {p1}' in c]
                col2 = [c for c in car_s2_cols if f'position {p2}' in c]
                if col1 and col2:
                    v1 = df[col1[0]].values
                    v2 = df[col2[0]].values
                    std1, std2 = np.std(v1), np.std(v2)
                    if std1 > 1e-12 and std2 > 1e-12:
                        corr = float(np.nan_to_num(np.corrcoef(v1, v2)[0, 1], nan=0.0))
                        correlations.append(corr)

    feats['max_car_asym_ratio'] = float(np.max(car_asym_ratios)) if car_asym_ratios else 1.0
    feats['min_car_asym_ratio'] = float(np.min(car_asym_ratios)) if car_asym_ratios else 1.0
    feats['max_car_asym_diff'] = float(np.max(car_asym_diffs)) if car_asym_diffs else 0.0
    feats['mean_cross_correlation'] = float(np.mean(correlations)) if correlations else 0.0
    feats['min_cross_correlation'] = float(np.min(correlations)) if correlations else 0.0

    # 6. Spectral & Frequency Band Features (Welch PSD & FFT)
    # Average across channels for aggregate spectral analysis
    s1_mean_trace = np.mean(s1_vib_data, axis=1) if s1_vib_data.shape[1] > 0 else np.zeros(len(df))
    s2_mean_trace = np.mean(s2_vib_data, axis=1) if s2_vib_data.shape[1] > 0 else np.zeros(len(df))

    try:
        nperseg = min(1024, len(df))
        freqs, psd_s1 = signal.welch(s1_mean_trace, fs=fs, nperseg=nperseg)
        _, psd_s2 = signal.welch(s2_mean_trace, fs=fs, nperseg=nperseg)

        # Dominant peak frequencies
        feats['s1_dom_freq'] = float(freqs[np.argmax(psd_s1)]) if len(psd_s1) > 0 else 0.0
        feats['s2_dom_freq'] = float(freqs[np.argmax(psd_s2)]) if len(psd_s2) > 0 else 0.0

        # Spectral Centroids
        s1_sum_psd = np.sum(psd_s1)
        s2_sum_psd = np.sum(psd_s2)
        feats['s1_spec_centroid'] = float(np.nan_to_num(np.sum(freqs * psd_s1) / (s1_sum_psd + 1e-12), nan=0.0))
        feats['s2_spec_centroid'] = float(np.nan_to_num(np.sum(freqs * psd_s2) / (s2_sum_psd + 1e-12), nan=0.0))

        # Characteristic Frequency Bands
        # Band 1: 20–100 Hz (track & sleeper passing)
        # Band 2: 100–300 Hz (P2 wheel-rail resonance)
        # Band 3: 300–800 Hz (pinned-pinned corrugation resonance)
        # Band 4: 800–1500 Hz (corrugation wear / high-frequency harmonics)
        # Band 5: 1500–3000 Hz (shock / impact)
        bands = {
            'b1_20_100hz': (20.0, 100.0),
            'b2_100_300hz': (100.0, 300.0),
            'b3_300_800hz': (300.0, 800.0),
            'b4_800_1500hz': (800.0, 1500.0),
            'b5_1500_3000hz': (1500.0, 3000.0)
        }

        for band_name, (low_f, high_f) in bands.items():
            mask = (freqs >= low_f) & (freqs <= high_f)
            e_s1 = float(np.sum(psd_s1[mask])) if np.any(mask) else 0.0
            e_s2 = float(np.sum(psd_s2[mask])) if np.any(mask) else 0.0

            feats[f'{band_name}_s1_energy'] = e_s1
            feats[f'{band_name}_s2_energy'] = e_s2
            feats[f'{band_name}_asym_ratio'] = float(np.nan_to_num(e_s1 / (e_s2 + 1e-12), nan=1.0))
            # Normalized spectral imbalance: (E1 - E2) / (E1 + E2)
            feats[f'{band_name}_spectral_imbalance'] = float(np.nan_to_num((e_s1 - e_s2) / (e_s1 + e_s2 + 1e-12), nan=0.0))

    except Exception:
        # Fallback in case of signal processing failure
        feats['s1_dom_freq'] = 0.0
        feats['s2_dom_freq'] = 0.0
        feats['s1_spec_centroid'] = 0.0
        feats['s2_spec_centroid'] = 0.0
        for band_name in ['b1_20_100hz', 'b2_100_300hz', 'b3_300_800hz', 'b4_800_1500hz', 'b5_1500_3000hz']:
            feats[f'{band_name}_s1_energy'] = 0.0
            feats[f'{band_name}_s2_energy'] = 0.0
            feats[f'{band_name}_asym_ratio'] = 1.0
            feats[f'{band_name}_spectral_imbalance'] = 0.0

    # Final sanitization pass: ensure all values are strictly finite floats
    for k, v in feats.items():
        feats[k] = float(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0))

    return feats
