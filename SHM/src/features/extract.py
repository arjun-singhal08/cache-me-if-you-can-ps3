import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import skew, kurtosis
from tqdm import tqdm
from typing import Dict, List, Optional
import joblib
from pathlib import Path

from .rainflow import rainflow_count, compute_miner_damage


def compute_spectral_features(stress: np.ndarray, fs: float = 1.0) -> Dict[str, float]:
    """Compute spectral features from stress time series."""
    n = len(stress)
    if n < 10:
        return {k: 0.0 for k in ['psd_peak_freq', 'psd_centroid', 'psd_bandwidth', 'spectral_entropy']}
    
    # Compute PSD using Welch's method
    nperseg = min(4096, n // 4)
    freqs, psd = signal.welch(stress, fs=fs, nperseg=nperseg)
    
    # Peak frequency
    peak_idx = np.argmax(psd)
    peak_freq = freqs[peak_idx]
    
    # Spectral centroid
    centroid = np.sum(freqs * psd) / np.sum(psd) if np.sum(psd) > 0 else 0.0
    
    # Spectral bandwidth
    bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * psd) / np.sum(psd)) if np.sum(psd) > 0 else 0.0
    
    # Spectral entropy
    psd_norm = psd / (np.sum(psd) + 1e-12)
    entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-12))
    
    return {
        'psd_peak_freq': float(peak_freq),
        'psd_centroid': float(centroid),
        'psd_bandwidth': float(bandwidth),
        'spectral_entropy': float(entropy)
    }


def compute_statistical_features(stress: np.ndarray) -> Dict[str, float]:
    """Compute statistical features from stress time series."""
    stress = stress[~np.isnan(stress)]
    if len(stress) == 0:
        return {k: 0.0 for k in [
            'mean', 'std', 'rms', 'ptp', 'crest_factor', 
            'skewness', 'kurtosis', 'zero_crossing_rate',
            'p01', 'p05', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95', 'p99'
        ]}
    
    mean_val = np.mean(stress)
    std_val = np.std(stress)
    rms = np.sqrt(np.mean(stress ** 2))
    ptp = np.max(stress) - np.min(stress)
    crest = np.max(np.abs(stress)) / (rms + 1e-12)
    skew_val = skew(stress)
    kurt_val = kurtosis(stress)
    
    # Zero crossing rate
    zero_cross = np.sum(np.diff(np.signbit(stress))) / len(stress)
    
    # Percentiles
    percentiles = np.percentile(stress, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    
    return {
        'mean': float(mean_val),
        'std': float(std_val),
        'rms': float(rms),
        'ptp': float(ptp),
        'crest_factor': float(crest),
        'skewness': float(skew_val),
        'kurtosis': float(kurt_val),
        'zero_crossing_rate': float(zero_cross),
        'p01': float(percentiles[0]),
        'p05': float(percentiles[1]),
        'p10': float(percentiles[2]),
        'p25': float(percentiles[3]),
        'p50': float(percentiles[4]),
        'p75': float(percentiles[5]),
        'p90': float(percentiles[6]),
        'p95': float(percentiles[7]),
        'p99': float(percentiles[8]),
    }


def compute_rainflow_features(stress: np.ndarray, m_values: List[float] = None) -> Dict[str, float]:
    """Compute rainflow-based features including Miner damage for multiple m values."""
    if m_values is None:
        m_values = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    
    amplitudes, means, counts = rainflow_count(stress)
    
    if len(amplitudes) == 0:
        base = {
            'n_cycles': 0.0,
            'amp_mean': 0.0, 'amp_std': 0.0, 'amp_max': 0.0, 'amp_min': 0.0,
            'amp_skew': 0.0, 'amp_kurt': 0.0,
            'mean_stress_mean': 0.0, 'mean_stress_std': 0.0,
        }
        for m in m_values:
            base[f'miner_damage_m{int(m)}'] = 0.0
        return base
    
    # Basic cycle statistics
    n_cycles = np.sum(counts)
    amp_mean = np.average(amplitudes, weights=counts)
    amp_std = np.sqrt(np.average((amplitudes - amp_mean) ** 2, weights=counts))
    amp_max = np.max(amplitudes)
    amp_min = np.min(amplitudes)
    amp_skew = skew(amplitudes)
    amp_kurt = kurtosis(amplitudes)
    
    mean_stress_mean = np.average(means, weights=counts)
    mean_stress_std = np.sqrt(np.average((means - mean_stress_mean) ** 2, weights=counts))
    
    # Miner damage for multiple m values
    miner_damages = {}
    for m in m_values:
        miner_damages[f'miner_damage_m{int(m)}'] = float(compute_miner_damage(amplitudes, counts, m))
    
    # 2D rainflow histogram features (amplitude vs mean stress)
    # Bin amplitudes and mean stresses, take top bins
    n_amp_bins = 10
    n_mean_bins = 10
    amp_bins = np.linspace(amplitudes.min(), amplitudes.max(), n_amp_bins + 1)
    mean_bins = np.linspace(means.min(), means.max(), n_mean_bins + 1)
    
    hist, _, _ = np.histogram2d(amplitudes, means, bins=[amp_bins, mean_bins], weights=counts)
    hist_flat = hist.flatten()
    top_k = min(10, len(hist_flat))
    top_indices = np.argpartition(hist_flat, -top_k)[-top_k:]
    top_values = hist_flat[top_indices]
    
    rf_hist_features = {f'rf_hist_{i}': float(v) for i, v in enumerate(sorted(top_values, reverse=True))}
    
    return {
        'n_cycles': float(n_cycles),
        'amp_mean': float(amp_mean),
        'amp_std': float(amp_std),
        'amp_max': float(amp_max),
        'amp_min': float(amp_min),
        'amp_skew': float(amp_skew),
        'amp_kurt': float(amp_kurt),
        'mean_stress_mean': float(mean_stress_mean),
        'mean_stress_std': float(mean_stress_std),
        **miner_damages,
        **rf_hist_features
    }


def extract_features_from_file(filepath: str, m_values: List[float] = None) -> Dict[str, float]:
    """Extract all features from a single CSV file."""
    # Load stress data (single column, no header)
    stress = pd.read_csv(filepath, header=None).values.flatten().astype(np.float64)
    
    features = {}
    features['file_id'] = Path(filepath).name
    
    # Statistical features
    features.update(compute_statistical_features(stress))
    
    # Spectral features
    features.update(compute_spectral_features(stress))
    
    # Rainflow features (most important)
    features.update(compute_rainflow_features(stress, m_values))
    
    return features


def extract_features_batch(filepaths: List[str], m_values: List[float] = None, n_jobs: int = -1) -> pd.DataFrame:
    """Extract features from multiple files in parallel."""
    if n_jobs == 1:
        results = [extract_features_from_file(fp, m_values) for fp in tqdm(filepaths, desc="Extracting features")]
    else:
        results = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(extract_features_from_file)(fp, m_values) for fp in tqdm(filepaths, desc="Extracting features")
        )
    return pd.DataFrame(results)


def prepare_features(df: pd.DataFrame, target_col: str = 'damage') -> tuple:
    """Prepare feature matrix X and target y from DataFrame."""
    # Identify feature columns (exclude file_id and target)
    exclude_cols = ['file_id', target_col]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32) if target_col in df.columns else None
    
    return X, y, feature_cols