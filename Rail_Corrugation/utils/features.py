"""
Handcrafted feature extraction for rail corrugation detection.
Extracts envelope spectrum, statistical moments, and spectral features per side.
"""

import numpy as np
from scipy import signal, fft, integrate
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

def trapz(y, x):
    """Wrapper for numpy/scipy trapz."""
    return integrate.trapezoid(y, x)


def compute_envelope_spectrum(signal_data: np.ndarray, fs: float = 10000) -> np.ndarray:
    """
    Compute envelope spectrum for a single channel.
    Corrugation creates periodic impacts at specific frequencies.
    
    Args:
        signal_data: 1D vibration signal
        fs: Sampling frequency (Hz)
    
    Returns:
        Envelope spectrum magnitude
    """
    # Hilbert transform to get analytic signal
    analytic = signal.hilbert(signal_data)
    envelope = np.abs(analytic)
    
    # Remove DC
    envelope = envelope - envelope.mean()
    
    # FFT of envelope
    n = len(envelope)
    env_fft = np.abs(fft.rfft(envelope))
    freqs = fft.rfftfreq(n, 1/fs)
    
    return env_fft, freqs


def extract_spectral_features(signal_data: np.ndarray, fs: float = 10000) -> Dict[str, float]:
    """Extract spectral features from a single channel."""
    n = len(signal_data)
    
    # Power spectral density
    freqs, psd = signal.welch(signal_data, fs=fs, nperseg=min(2048, n//4))
    
    # Total power
    total_power = trapz(psd, freqs)
    
    # Spectral centroid
    centroid = np.sum(freqs * psd) / (np.sum(psd) + 1e-10)
    
    # Spectral bandwidth
    bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * psd) / (np.sum(psd) + 1e-10))
    
    # Spectral rolloff (85%)
    cumsum = np.cumsum(psd)
    rolloff_idx = np.where(cumsum >= 0.85 * cumsum[-1])[0]
    rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs[-1]
    
    # Peak frequency
    peak_freq = freqs[np.argmax(psd)]
    
    # Power in specific bands (corrugation related)
    # Low frequency: 10-100 Hz (wheel-rail resonance)
    low_mask = (freqs >= 10) & (freqs <= 100)
    low_power = trapz(psd[low_mask], freqs[low_mask]) if np.any(low_mask) else 0
    
    # Mid frequency: 100-500 Hz (corrugation harmonics)
    mid_mask = (freqs >= 100) & (freqs <= 500)
    mid_power = trapz(psd[mid_mask], freqs[mid_mask]) if np.any(mid_mask) else 0
    
    # High frequency: 500-2000 Hz (shock)
    high_mask = (freqs >= 500) & (freqs <= 2000)
    high_power = trapz(psd[high_mask], freqs[high_mask]) if np.any(high_mask) else 0
    
    return {
        'total_power': total_power,
        'centroid': centroid,
        'bandwidth': bandwidth,
        'rolloff': rolloff,
        'peak_freq': peak_freq,
        'low_power': low_power,
        'mid_power': mid_power,
        'high_power': high_power,
    }


def extract_statistical_features(signal_data: np.ndarray) -> Dict[str, float]:
    """Extract time-domain statistical features."""
    return {
        'mean': np.mean(signal_data),
        'std': np.std(signal_data),
        'rms': np.sqrt(np.mean(signal_data ** 2)),
        'skew': skew(signal_data),
        'kurtosis': kurtosis(signal_data),
        'crest_factor': np.max(np.abs(signal_data)) / (np.std(signal_data) + 1e-10),
        'impulse_factor': np.max(np.abs(signal_data)) / (np.mean(np.abs(signal_data)) + 1e-10),
        'shape_factor': np.std(signal_data) / (np.mean(np.abs(signal_data)) + 1e-10),
        'clearance_factor': np.max(np.abs(signal_data)) / (np.mean(np.sqrt(np.abs(signal_data))) + 1e-10) ** 2,
    }


def skew(x: np.ndarray) -> float:
    """Compute skewness."""
    mean = np.mean(x)
    std = np.std(x)
    if std < 1e-10:
        return 0.0
    return np.mean(((x - mean) / std) ** 3)


def kurtosis(x: np.ndarray) -> float:
    """Compute excess kurtosis."""
    mean = np.mean(x)
    std = np.std(x)
    if std < 1e-10:
        return 0.0
    return np.mean(((x - mean) / std) ** 4) - 3


def extract_envelope_features(signal_data: np.ndarray, fs: float = 10000) -> Dict[str, float]:
    """Extract features from envelope spectrum."""
    try:
        env_fft, freqs = compute_envelope_spectrum(signal_data, fs)
        
        # Find peaks in envelope spectrum
        # Corrugation wavelengths correspond to specific frequencies
        # Typical corrugation: 20-200mm wavelength
        # At 30 km/h (8.33 m/s): f = v/lambda = 8.33/0.02 to 8.33/0.2 = 41-416 Hz
        # At 80 km/h (22.22 m/s): f = 111-1111 Hz
        
        # Peak in envelope spectrum
        peak_idx = np.argmax(env_fft)
        peak_freq = freqs[peak_idx]
        peak_mag = env_fft[peak_idx]
        
        # Energy in corrugation-relevant bands
        # Band 1: 20-200 Hz (long wavelength corrugation)
        band1_mask = (freqs >= 20) & (freqs <= 200)
        band1_energy = np.sum(env_fft[band1_mask] ** 2) if np.any(band1_mask) else 0
        
        # Band 2: 200-800 Hz (medium wavelength)
        band2_mask = (freqs >= 200) & (freqs <= 800)
        band2_energy = np.sum(env_fft[band2_mask] ** 2) if np.any(band2_mask) else 0
        
        # Band 3: 800-2000 Hz (short wavelength)
        band3_mask = (freqs >= 800) & (freqs <= 2000)
        band3_energy = np.sum(env_fft[band3_mask] ** 2) if np.any(band3_mask) else 0
        
        # Total envelope energy
        total_env_energy = np.sum(env_fft ** 2)
        
        # Normalized band energies
        band1_norm = band1_energy / (total_env_energy + 1e-10)
        band2_norm = band2_energy / (total_env_energy + 1e-10)
        band3_norm = band3_energy / (total_env_energy + 1e-10)
        
        return {
            'env_peak_freq': peak_freq,
            'env_peak_mag': peak_mag,
            'env_band1_energy': band1_norm,
            'env_band2_energy': band2_norm,
            'env_band3_energy': band3_norm,
            'env_total_energy': total_env_energy,
        }
    except Exception:
        return {
            'env_peak_freq': 0, 'env_peak_mag': 0,
            'env_band1_energy': 0, 'env_band2_energy': 0, 'env_band3_energy': 0,
            'env_total_energy': 0,
        }


def extract_cross_channel_features(side_data: np.ndarray) -> Dict[str, float]:
    """
    Extract features from cross-channel relationships.
    side_data: [T, C] or [C, T]
    """
    # Ensure [T, C]
    if side_data.shape[0] < side_data.shape[1]:
        side_data = side_data.T
    
    T, C = side_data.shape
    
    # Mean correlation across channels
    corr_matrix = np.corrcoef(side_data, rowvar=False)
    # Upper triangle (excluding diagonal)
    upper_tri = corr_matrix[np.triu_indices(C, k=1)]
    mean_corr = np.nanmean(upper_tri)
    max_corr = np.nanmax(upper_tri)
    std_corr = np.nanstd(upper_tri)
    
    # Coherence-like measure: average pairwise correlation at lag 0
    # Also check spatial consistency (corrugation affects all channels similarly)
    channel_means = np.mean(side_data, axis=0)
    channel_stds = np.std(side_data, axis=0)
    
    return {
        'mean_cross_corr': mean_corr if not np.isnan(mean_corr) else 0,
        'max_cross_corr': max_corr if not np.isnan(max_corr) else 0,
        'std_cross_corr': std_corr if not np.isnan(std_corr) else 0,
        'channel_mean_std': np.std(channel_means),
        'channel_std_mean': np.mean(channel_stds),
    }


def extract_all_features(
    side_i: np.ndarray, 
    side_ii: np.ndarray,
    fs: float = 10000
) -> np.ndarray:
    """
    Extract all handcrafted features for both sides.
    
    Args:
        side_i: [T, 64] - Side I angle-domain signals
        side_ii: [T, 64] - Side II angle-domain signals
        fs: Original sampling frequency (for spectral features)
    
    Returns:
        Feature vector [n_features]
    """
    features = []
    
    for side_name, side_data in [('side_i', side_i), ('side_ii', side_ii)]:
        # Per-channel features (aggregate statistics across 64 channels)
        spectral_feats = []
        stat_feats = []
        envelope_feats = []
        
        for c in range(side_data.shape[1]):
            ch_data = side_data[:, c]
            
            # Spectral
            spec = extract_spectral_features(ch_data, fs)
            spectral_feats.append(list(spec.values()))
            
            # Statistical
            stat = extract_statistical_features(ch_data)
            stat_feats.append(list(stat.values()))
            
            # Envelope
            env = extract_envelope_features(ch_data, fs)
            envelope_feats.append(list(env.values()))
        
        # Aggregate across channels: mean, std, max, min
        spectral_feats = np.array(spectral_feats)  # [64, n_spec]
        stat_feats = np.array(stat_feats)
        envelope_feats = np.array(envelope_feats)
        
        for feats, prefix in [(spectral_feats, 'spec'), (stat_feats, 'stat'), (envelope_feats, 'env')]:
            if feats.size > 0:
                features.extend([
                    np.nanmean(feats, axis=0),
                    np.nanstd(feats, axis=0),
                    np.nanmax(feats, axis=0),
                    np.nanmin(feats, axis=0),
                ])
        
        # Cross-channel features
        cross = extract_cross_channel_features(side_data)
        features.append(list(cross.values()))
    
    # Side comparison features
    # Difference in energy between sides
    side_i_energy = np.mean(side_i ** 2)
    side_ii_energy = np.mean(side_ii ** 2)
    features.append([side_i_energy, side_ii_energy, side_i_energy - side_ii_energy, side_i_energy / (side_ii_energy + 1e-10)])
    
    # Flatten
    flat_features = []
    for f in features:
        if isinstance(f, (list, np.ndarray)):
            flat_features.extend(np.array(f).flatten())
        else:
            flat_features.append(f)
    
    return np.array(flat_features, dtype=np.float32)


def get_feature_names() -> List[str]:
    """Get feature names for interpretability."""
    # This is approximate - actual order depends on extract_all_features
    base_names = []
    # Per side: spectral (8) * 4 agg + stat (9) * 4 agg + envelope (6) * 4 agg + cross (5) = 32+36+24+5 = 97 per side
    # Side comparison: 4
    # Total: ~198 features
    return [f"feat_{i}" for i in range(200)]  # Approximate


if __name__ == "__main__":
    # Test feature extraction
    import pandas as pd
    df = pd.read_csv('../NebulaX-Hackathon-ProblemStatement/PS3/02_Datasets/Rail_Corrugation/Train/Train1.csv', header=None, low_memory=False)
    data = df.iloc[1:].values.astype(np.float32)
    speed = data[:, 0]
    sensor = data[:, 1:]
    
    from utils.angle_resample import resample_to_angle_domain, compute_angle_from_pulses
    angles = compute_angle_from_pulses(speed)
    resampled = resample_to_angle_domain(sensor, angles, 1800)
    
    side_i = resampled[:, :64]
    side_ii = resampled[:, 64:]
    
    feats = extract_all_features(side_i, side_ii)
    print(f"Feature vector shape: {feats.shape}")
    print(f"Features: {feats[:10]}")