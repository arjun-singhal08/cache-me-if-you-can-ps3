# features/extract.py
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.signal import welch
from typing import Dict, List
from .align import interpolate_to_position_grid

def spectral_entropy(psd: np.ndarray) -> float:
    """Compute spectral entropy of power spectral density."""
    psd_norm = psd / (np.sum(psd) + 1e-10)
    return -np.sum(psd_norm * np.log(psd_norm + 1e-10))

def compute_rise_time(signal: np.ndarray, grid: np.ndarray) -> float:
    """Compute 10%-90% rise time."""
    if len(signal) < 2:
        return 0.0
    min_val, max_val = np.min(signal), np.max(signal)
    if max_val - min_val < 1e-6:
        return 0.0
    target_10 = min_val + 0.1 * (max_val - min_val)
    target_90 = min_val + 0.9 * (max_val - min_val)
    idx_10 = np.where(signal >= target_10)[0]
    idx_90 = np.where(signal >= target_90)[0]
    if len(idx_10) == 0 or len(idx_90) == 0:
        return 0.0
    return (grid[idx_90[0]] - grid[idx_10[0]]) / (grid[-1] - grid[0]) if grid[-1] > grid[0] else 0.0

def compute_settling_time(signal: np.ndarray, grid: np.ndarray, threshold: float = 0.05) -> float:
    """Compute time to settle within threshold% of final value."""
    if len(signal) < 2:
        return 0.0
    final_val = signal[-1]
    target_band = threshold * abs(final_val) if final_val != 0 else threshold
    settled_idx = np.where(np.abs(signal - final_val) <= target_band)[0]
    if len(settled_idx) == 0:
        return 1.0
    first_settled = settled_idx[0]
    return grid[first_settled] / grid[-1] if grid[-1] > 0 else 0.0

def compute_overshoot(signal: np.ndarray) -> float:
    """Compute overshoot percentage."""
    if len(signal) < 2:
        return 0.0
    steady_state = signal[-1]
    peak = np.max(signal)
    if steady_state == 0:
        return 0.0
    return max(0, (peak - steady_state) / abs(steady_state))

def compute_phase_lag(sig1: np.ndarray, sig2: np.ndarray) -> float:
    """Compute phase lag between two signals using cross-correlation."""
    if len(sig1) != len(sig2) or len(sig1) < 2:
        return 0.0
    # Normalize
    s1 = (sig1 - np.mean(sig1)) / (np.std(sig1) + 1e-10)
    s2 = (sig2 - np.mean(sig2)) / (np.std(sig2) + 1e-10)
    corr = np.correlate(s1, s2, mode='full')
    lag = np.argmax(corr) - (len(sig1) - 1)
    return lag / len(sig1)  # Normalized lag

def extract_statistical_features(signal: np.ndarray, prefix: str) -> Dict[str, float]:
    """Extract statistical features from a signal."""
    return {
        f'{prefix}_mean': float(np.mean(signal)),
        f'{prefix}_std': float(np.std(signal)),
        f'{prefix}_min': float(np.min(signal)),
        f'{prefix}_max': float(np.max(signal)),
        f'{prefix}_median': float(np.median(signal)),
        f'{prefix}_skew': float(skew(signal)) if len(signal) > 2 else 0.0,
        f'{prefix}_kurt': float(kurtosis(signal)) if len(signal) > 3 else 0.0,
        f'{prefix}_range': float(np.max(signal) - np.min(signal)),
        f'{prefix}_energy': float(np.sum(signal ** 2)),
    }

def extract_spectral_features(signal: np.ndarray, fs: float = 50.0, prefix: str = '') -> Dict[str, float]:
    """Extract spectral features from a signal."""
    if len(signal) < 4:
        return {f'{prefix}_spec_entropy': 0.0, f'{prefix}_dom_freq': 0.0, f'{prefix}_top5_power': 0.0}
    
    nperseg = min(64, len(signal))
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    
    return {
        f'{prefix}_spec_entropy': float(spectral_entropy(psd)),
        f'{prefix}_dom_freq': float(freqs[np.argmax(psd)]),
        f'{prefix}_top5_power': float(np.sum(np.sort(psd)[-5:])),
    }

def extract_segment_features(seg_df: pd.DataFrame, operation: str) -> Dict[str, float]:
    """
    Extract comprehensive features from a door cycle segment.
    
    Args:
        seg_df: DataFrame slice for one segment
        operation: 'Open' or 'Close'
    
    Returns:
        Dictionary of features
    """
    features = {}
    
    # Raw signals
    current = seg_df['Motor current(mA)'].values.astype(float)
    voltage = seg_df['Motor Voltage(10mV)'].values.astype(float)
    bemf = seg_df['Motor electrodynamic force'].values.astype(float)
    position = seg_df['Door leaf position'].values.astype(float)
    time = np.arange(len(seg_df)) * 0.02  # 20ms per sample
    
    # --- Statistical features (time-domain) ---
    for name, sig in [('current', current), ('voltage', voltage), 
                      ('bemf', bemf), ('position', position)]:
        features.update(extract_statistical_features(sig, name))
    
    # --- Spectral features ---
    for name, sig in [('current', current), ('voltage', voltage), ('bemf', bemf)]:
        features.update(extract_spectral_features(sig, prefix=name))
    
    # --- Physics-based features (position-aligned) ---
    pos_grid = np.linspace(position.min(), position.max(), 50)
    if len(np.unique(position)) > 3:
        curr_interp = interpolate_to_position_grid(current, position, pos_grid)
        volt_interp = interpolate_to_position_grid(voltage, position, pos_grid)
        bemf_interp = interpolate_to_position_grid(bemf, position, pos_grid)
        
        features['current_integral'] = float(np.trapezoid(curr_interp, pos_grid))
        features['voltage_integral'] = float(np.trapezoid(volt_interp, pos_grid))
        features['bemf_integral'] = float(np.trapezoid(bemf_interp, pos_grid))
        features['peak_current_pos'] = float(np.max(curr_interp))
        features['peak_voltage_pos'] = float(np.max(volt_interp))
        features['peak_bemf_pos'] = float(np.max(bemf_interp))
        
        # Shape features on position-aligned signals
        features['current_rise_time'] = compute_rise_time(curr_interp, pos_grid)
        features['current_settling_time'] = compute_settling_time(curr_interp, pos_grid)
        features['current_overshoot'] = compute_overshoot(curr_interp)
    else:
        # Fallback if position doesn't vary enough
        features['current_integral'] = float(np.trapezoid(current, time))
        features['voltage_integral'] = float(np.trapezoid(voltage, time))
        features['bemf_integral'] = float(np.trapezoid(bemf, time))
        features['peak_current_pos'] = float(np.max(current))
        features['peak_voltage_pos'] = float(np.max(voltage))
        features['peak_bemf_pos'] = float(np.max(bemf))
        features['current_rise_time'] = 0.0
        features['current_settling_time'] = 0.0
        features['current_overshoot'] = 0.0
    
    # --- Cross-channel features ---
    features['iv_correlation'] = float(np.corrcoef(current, voltage)[0, 1]) if len(current) > 1 else 0.0
    features['ibemf_correlation'] = float(np.corrcoef(current, bemf)[0, 1]) if len(current) > 1 else 0.0
    features['iv_phase_lag'] = compute_phase_lag(current, voltage)
    features['vbemf_correlation'] = float(np.corrcoef(voltage, bemf)[0, 1]) if len(voltage) > 1 else 0.0
    
    # --- Power/Energy features ---
    features['current_x_voltage_integral'] = float(np.trapezoid(current * voltage / 1000, time))  # Approx power
    features['bemf_x_current_integral'] = float(np.trapezoid(bemf * current / 1000, time))
    
    # --- Door state features ---
    features['door_opened_final'] = float(seg_df['Door Opened'].iloc[-1])
    features['door_locked_final'] = float(seg_df['Door Locked'].iloc[-1])
    
    # --- Temporal features ---
    features['duration_sec'] = len(seg_df) * 0.02
    features['n_samples'] = len(seg_df)
    features['operation'] = 1.0 if operation == 'Open' else 0.0
    
    # --- Position profile features ---
    features['position_start'] = float(position[0])
    features['position_end'] = float(position[-1])
    features['position_range'] = float(np.max(position) - np.min(position))
    
    # Current profile by position quartiles (if position varies)
    if len(np.unique(position)) > 5:
        pos_sorted_idx = np.argsort(position)
        pos_sorted = position[pos_sorted_idx]
        curr_sorted = current[pos_sorted_idx]
        n = len(curr_sorted)
        q1 = curr_sorted[:n//4].mean() if n > 3 else 0
        q2 = curr_sorted[n//4:3*n//4].mean() if n > 3 else 0
        q3 = curr_sorted[3*n//4:].mean() if n > 3 else 0
        features['current_q1'] = float(q1)
        features['current_q2'] = float(q2)
        features['current_q3'] = float(q3)
        features['current_q3_q1_ratio'] = float(q3 / q1) if q1 > 0 else 0.0
    else:
        features['current_q1'] = float(np.mean(current))
        features['current_q2'] = float(np.mean(current))
        features['current_q3'] = float(np.mean(current))
        features['current_q3_q1_ratio'] = 1.0
    
    return features