"""
Physics-grounded feature extraction for Rail Corrugation detection - UPGRADED.

Key upgrades for breaking Macro-F1 plateau:
1. Spatial Wavelength Tracking (Order Tracking via Wheel Speed Telemetry)
   - Corrugation is spatial defect with fixed wavelength λ ≈ 20-150 mm
   - f = v/λ, so convert time-domain spectra to spatial wavelength domain
   - Extract energy in physical corrugation bands:
     * Short-pitch: λ ∈ [20, 40] mm
     * Medium-pitch: λ ∈ [40, 80] mm  
     * Long-pitch: λ ∈ [80, 150] mm

2. Cross-Bilateral Differential Features (Direct Left-Right Isolation)
   - Pointwise difference Δx_t = x_SideI(t) - x_SideII(t)
   - Teager-Kaiser Energy Operator (TKEO) asymmetry
   - High-percentile shock peak ratios (95th, 99th)
   - Envelope spectrum bilateral kurtosis/crest factor ratios

3. Existing features preserved + new physics-grounded features
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal, stats, integrate
from scipy.fft import rfft, rfftfreq
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Sampling frequency
FS = 10000  # Hz
N_SAMPLES = 10000
WHEEL_DIAMETER = 0.85  # meters
TEETH_PER_REV = 90

# Channel layout: 129 columns
# Col 0: Rotating speed (pulse train)
# Cols 1-64: Side I (32 axle boxes * 2 = vibration + shock)
# Cols 65-128: Side II (32 axle boxes * 2 = vibration + shock)

SIDE_I_CHANNELS = []
SIDE_II_CHANNELS = []

for car in range(8):
    for pos in [0, 2, 4, 6]:  # Positions 1,3,5,7 (0-indexed)
        base = (car * 8 + pos) * 2 + 1
        SIDE_I_CHANNELS.append(base)
        SIDE_I_CHANNELS.append(base + 1)

for car in range(8):
    for pos in [1, 3, 5, 7]:  # Positions 2,4,6,8 (0-indexed)
        base = (car * 8 + pos) * 2 + 1
        SIDE_II_CHANNELS.append(base)
        SIDE_II_CHANNELS.append(base + 1)

# Physical corrugation wavelength bands (mm)
SPATIAL_WAVELENGTH_BANDS = [
    (20, 40, 'short_pitch'),
    (40, 80, 'medium_pitch'),
    (80, 150, 'long_pitch'),
]

# Time-domain resonance bands (Hz) - kept for reference
RESONANCE_BANDS = [
    (20, 100, 'band1_track_sleeper'),
    (100, 300, 'band2_p2_resonance'),
    (300, 800, 'band3_pinned_pinned'),
    (800, 1500, 'band4_wear_harmonics'),
    (1500, 3000, 'band5_shock'),
]


def compute_velocity_from_speed_pulses(speed_pulses: np.ndarray) -> np.ndarray:
    """
    Compute instantaneous velocity from 90-tooth wheel speed sensor.
    Each transition = 1/180 revolution. Wheel circumference = π * 0.85 m.
    Returns velocity in m/s for each sample.
    """
    transitions = np.diff(speed_pulses.astype(np.int32)) != 0
    transition_indices = np.where(transitions)[0]
    
    if len(transition_indices) < 2:
        return np.full_like(speed_pulses, 10.0)  # fallback ~36 km/h
    
    # Time between transitions
    dt = np.diff(transition_indices) / FS  # seconds
    
    # Velocity for each transition interval: distance / time
    # Distance per transition = wheel_circumference / 180
    dist_per_transition = np.pi * WHEEL_DIAMETER / 180
    velocities = dist_per_transition / dt  # m/s
    
    # Interpolate to all samples
    v = np.zeros_like(speed_pulses, dtype=np.float32)
    v[:transition_indices[0]] = velocities[0] if len(velocities) > 0 else 10.0
    
    for i, (start, end) in enumerate(zip(transition_indices[:-1], transition_indices[1:])):
        if i < len(velocities):
            v[start:end] = velocities[i]
    
    v[transition_indices[-1]:] = velocities[-1] if len(velocities) > 0 else 10.0
    
    return v


def compute_welch_psd(signal_data: np.ndarray, fs: int = FS) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Welch PSD with appropriate windowing."""
    nperseg = min(2048, len(signal_data) // 4)
    freqs, psd = signal.welch(signal_data, fs=fs, nperseg=nperseg, window='hann')
    return freqs, psd


def band_power(freqs: np.ndarray, psd: np.ndarray, fmin: float, fmax: float) -> float:
    """Integrate PSD power in frequency band."""
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return 0.0
    return integrate.trapezoid(psd[mask], freqs[mask])


def spectral_centroid(freqs: np.ndarray, psd: np.ndarray) -> float:
    return np.sum(freqs * psd) / (np.sum(psd) + 1e-10)


def spectral_bandwidth(freqs: np.ndarray, psd: np.ndarray, centroid: float) -> float:
    return np.sqrt(np.sum(((freqs - centroid) ** 2) * psd) / (np.sum(psd) + 1e-10))


def spectral_rolloff(freqs: np.ndarray, psd: np.ndarray, threshold: float = 0.85) -> float:
    cumsum = np.cumsum(psd)
    total = cumsum[-1]
    idx = np.where(cumsum >= threshold * total)[0]
    return freqs[idx[0]] if len(idx) > 0 else freqs[-1]


# ============ SPATIAL WAVELENGTH TRACKING (ORDER TRACKING) ============

def compute_spatial_spectrum(signal_data: np.ndarray, velocity: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert time-domain signal to spatial wavelength domain using order tracking.
    
    For each sample, we have instantaneous velocity v(t). The spatial wavelength
    is λ = v / f. We resample the signal to constant spatial increments.
    
    Returns:
        wavelengths: spatial wavelengths in mm
        spatial_psd: PSD in spatial domain (energy per wavelength bin)
    """
    # Resample to constant spatial increments
    # Total distance traveled
    distance = np.cumsum(velocity) / FS  # meters
    total_distance = distance[-1]
    
    if total_distance < 0.1:
        return np.array([20, 150]), np.array([0.0, 0.0])
    
    # Resample to constant spatial step (1 mm steps)
    spatial_step = 0.001  # 1 mm
    n_spatial = int(total_distance / spatial_step)
    if n_spatial < 10:
        n_spatial = 10
    
    spatial_grid = np.linspace(0, total_distance, n_spatial)
    
    # Interpolate signal to spatial grid
    try:
        interp = np.interp(spatial_grid, distance, signal_data)
    except ValueError:
        return np.array([20, 150]), np.array([0.0, 0.0])
    
    # Compute PSD in spatial domain
    # Spatial frequency in cycles/meter
    spatial_fs = 1.0 / spatial_step  # samples per meter
    spatial_freqs, spatial_psd = signal.welch(interp, fs=spatial_fs, nperseg=min(1024, n_spatial//4))
    
    # Convert spatial frequency to wavelength (mm)
    # wavelength (m) = 1 / spatial_freq (cycles/m) = spatial_step * N / k
    wavelengths_m = 1.0 / (spatial_freqs + 1e-10)
    wavelengths_mm = wavelengths_m * 1000
    
    return wavelengths_mm, spatial_psd


def extract_spatial_wavelength_features(signal_data: np.ndarray, velocity: np.ndarray) -> Dict[str, float]:
    """
    Extract energy in physical corrugation wavelength bands.
    These bands are speed-invariant because corrugation is a spatial defect.
    """
    wavelengths, spatial_psd = compute_spatial_spectrum(signal_data, velocity)
    
    features = {}
    
    # Energy in physical corrugation wavelength bands
    total_spatial_energy = integrate.trapezoid(spatial_psd, wavelengths)
    
    for wmin, wmax, name in SPATIAL_WAVELENGTH_BANDS:
        mask = (wavelengths >= wmin) & (wavelengths <= wmax)
        if np.any(mask):
            band_energy = integrate.trapezoid(spatial_psd[mask], wavelengths[mask])
            features[f'spatial_{name}_energy'] = band_energy
            features[f'spatial_{name}_ratio'] = band_energy / (total_spatial_energy + 1e-10)
        else:
            features[f'spatial_{name}_energy'] = 0.0
            features[f'spatial_{name}_ratio'] = 0.0
    
    # Peak spatial wavelength
    peak_idx = np.argmax(spatial_psd)
    features['spatial_peak_wavelength'] = wavelengths[peak_idx]
    features['spatial_peak_energy'] = spatial_psd[peak_idx]
    
    # Dominant wavelength band indicator
    band_energies = [features.get(f'spatial_{name}_energy', 0) for _, _, name in SPATIAL_WAVELENGTH_BANDS]
    if sum(band_energies) > 0:
        features['spatial_dominant_band'] = np.argmax(band_energies)
    else:
        features['spatial_dominant_band'] = 0
    
    return features


# ============ CROSS-BILATERAL DIFFERENTIAL FEATURES ============

def teager_kaiser_energy(signal_data: np.ndarray) -> np.ndarray:
    """
    Teager-Kaiser Energy Operator for instantaneous energy estimation.
    Ψ[x(n)] = x(n)^2 - x(n-1)*x(n+1)
    """
    tkeo = np.zeros_like(signal_data)
    tkeo[1:-1] = signal_data[1:-1]**2 - signal_data[:-2] * signal_data[2:]
    tkeo[0] = signal_data[0]**2
    tkeo[-1] = signal_data[-1]**2
    return np.maximum(tkeo, 0)


def extract_cross_bilateral_features(side_i_data: np.ndarray, side_ii_data: np.ndarray) -> Dict[str, float]:
    """
    Direct Left-Right differential features for corrugation isolation.
    These explicitly encode the bilateral asymmetry rather than letting models infer it.
    """
    # Ensure (N, C) format
    if side_i_data.shape[0] == 64:
        side_i_data = side_i_data.T
    if side_ii_data.shape[0] == 64:
        side_ii_data = side_ii_data.T
    
    N, C = side_i_data.shape
    features = {}
    
    # 1. Pointwise difference Δx_t = x_SideI(t) - x_SideII(t)
    diff_signals = side_i_data - side_ii_data
    sum_signals = side_i_data + side_ii_data
    
    diff_rms = np.sqrt(np.mean(diff_signals**2, axis=0))
    sum_rms = np.sqrt(np.mean(sum_signals**2, axis=0))
    
    features['diff_rms_mean'] = np.nanmean(diff_rms)
    features['diff_rms_std'] = np.nanstd(diff_rms)
    features['diff_rms_max'] = np.nanmax(diff_rms)
    
    # Normalized difference: (I - II) / (I + II)
    norm_diff = diff_rms / (sum_rms + 1e-10)
    features['norm_diff_mean'] = np.nanmean(norm_diff)
    features['norm_diff_std'] = np.nanstd(norm_diff)
    features['norm_diff_max'] = np.nanmax(np.abs(norm_diff))
    
    # 2. Teager-Kaiser Energy Operator (TKEO) asymmetry
    tkeo_asym = []
    for c in range(C):
        tkeo_i = teager_kaiser_energy(side_i_data[:, c])
        tkeo_ii = teager_kaiser_energy(side_ii_data[:, c])
        
        tkeo_energy_i = np.mean(tkeo_i)
        tkeo_energy_ii = np.mean(tkeo_ii)
        
        # TKEO asymmetry: (I - II) / (I + II)
        asym = (tkeo_energy_i - tkeo_energy_ii) / (tkeo_energy_i + tkeo_energy_ii + 1e-10)
        tkeo_asym.append(asym)
    
    features['tkeo_asym_mean'] = np.nanmean(tkeo_asym)
    features['tkeo_asym_std'] = np.nanstd(tkeo_asym)
    features['tkeo_asym_max'] = np.nanmax(np.abs(tkeo_asym))
    
    # 3. High-percentile shock peak ratios (95th, 99th percentile)
    # Only for shock channels (odd indices)
    shock_peak_ratios_95 = []
    shock_peak_ratios_99 = []
    
    for c in range(1, C, 2):  # shock channels only
        peak_i_95 = np.percentile(np.abs(side_i_data[:, c]), 95)
        peak_ii_95 = np.percentile(np.abs(side_ii_data[:, c]), 95)
        peak_i_99 = np.percentile(np.abs(side_i_data[:, c]), 99)
        peak_ii_99 = np.percentile(np.abs(side_ii_data[:, c]), 99)
        
        shock_peak_ratios_95.append(peak_i_95 / (peak_ii_95 + 1e-10))
        shock_peak_ratios_99.append(peak_i_99 / (peak_ii_99 + 1e-10))
    
    features['shock_peak_95_ratio_mean'] = np.nanmean(shock_peak_ratios_95)
    features['shock_peak_95_ratio_std'] = np.nanstd(shock_peak_ratios_95)
    features['shock_peak_95_ratio_max'] = np.nanmax(shock_peak_ratios_95)
    features['shock_peak_99_ratio_mean'] = np.nanmean(shock_peak_ratios_99)
    features['shock_peak_99_ratio_std'] = np.nanstd(shock_peak_ratios_99)
    features['shock_peak_99_ratio_max'] = np.nanmax(shock_peak_ratios_99)
    
    # 4. Envelope spectrum bilateral kurtosis and crest factor ratios
    env_kurtosis_ratios = []
    env_crest_ratios = []
    
    for c in range(1, C, 2):  # shock channels only
        # Hilbert envelope
        analytic_i = signal.hilbert(side_i_data[:, c])
        analytic_ii = signal.hilbert(side_ii_data[:, c])
        
        env_i = np.abs(analytic_i)
        env_ii = np.abs(analytic_ii)
        
        # Kurtosis of envelope
        kurt_i = stats.kurtosis(env_i)
        kurt_ii = stats.kurtosis(env_ii)
        env_kurtosis_ratios.append(kurt_i / (kurt_ii + 1e-10))
        
        # Crest factor of envelope
        crest_i = np.max(env_i) / (np.std(env_i) + 1e-10)
        crest_ii = np.max(env_ii) / (np.std(env_ii) + 1e-10)
        env_crest_ratios.append(crest_i / (crest_ii + 1e-10))
    
    features['env_kurtosis_ratio_mean'] = np.nanmean(env_kurtosis_ratios)
    features['env_kurtosis_ratio_std'] = np.nanstd(env_kurtosis_ratios)
    features['env_crest_ratio_mean'] = np.nanmean(env_crest_ratios)
    features['env_crest_ratio_std'] = np.nanstd(env_crest_ratios)
    
    return features


# ============ EXISTING FEATURE FUNCTIONS (preserved) ============

def compute_envelope_features(signal_data: np.ndarray, fs: int = FS) -> Dict[str, float]:
    analytic = signal.hilbert(signal_data)
    envelope = np.abs(analytic)
    envelope = envelope - envelope.mean()
    
    n = len(envelope)
    env_fft = np.abs(rfft(envelope))
    env_freqs = rfftfreq(n, 1/fs)
    
    peak_idx = np.argmax(env_fft)
    peak_freq = env_freqs[peak_idx]
    peak_mag = env_fft[peak_idx]
    
    band1_mask = (env_freqs >= 20) & (env_freqs <= 200)
    band2_mask = (env_freqs >= 200) & (env_freqs <= 800)
    band3_mask = (env_freqs >= 800) & (env_freqs <= 2000)
    
    total_env = np.sum(env_fft ** 2)
    band1_e = np.sum(env_fft[band1_mask] ** 2) / (total_env + 1e-10)
    band2_e = np.sum(env_fft[band2_mask] ** 2) / (total_env + 1e-10)
    band3_e = np.sum(env_fft[band3_mask] ** 2) / (total_env + 1e-10)
    
    return {
        'env_peak_freq': peak_freq,
        'env_peak_mag': peak_mag,
        'env_band1_ratio': band1_e,
        'env_band2_ratio': band2_e,
        'env_band3_ratio': band3_e,
    }


def compute_statistical_features(signal_data: np.ndarray) -> Dict[str, float]:
    return {
        'mean': np.mean(signal_data),
        'std': np.std(signal_data),
        'rms': np.sqrt(np.mean(signal_data ** 2)),
        'skew': stats.skew(signal_data),
        'kurtosis': stats.kurtosis(signal_data),
        'crest_factor': np.max(np.abs(signal_data)) / (np.std(signal_data) + 1e-10),
        'impulse_factor': np.max(np.abs(signal_data)) / (np.mean(np.abs(signal_data)) + 1e-10),
        'clearance_factor': np.max(np.abs(signal_data)) / (np.mean(np.sqrt(np.abs(signal_data))) + 1e-10) ** 2,
    }


def compute_spectral_features(signal_data: np.ndarray, fs: int = FS) -> Dict[str, float]:
    freqs, psd = compute_welch_psd(signal_data, fs)
    
    features = {}
    
    for fmin, fmax, name in RESONANCE_BANDS:
        features[f'{name}_power'] = band_power(freqs, psd, fmin, fmax)
    
    total_power = integrate.trapezoid(psd, freqs)
    features['total_power'] = total_power
    
    for fmin, fmax, name in RESONANCE_BANDS:
        features[f'{name}_ratio'] = features[f'{name}_power'] / (total_power + 1e-10)
    
    centroid = spectral_centroid(freqs, psd)
    features['spectral_centroid'] = centroid
    features['spectral_bandwidth'] = spectral_bandwidth(freqs, psd, centroid)
    features['spectral_rolloff'] = spectral_rolloff(freqs, psd)
    features['peak_freq'] = freqs[np.argmax(psd)]
    
    return features


def extract_side_features(side_data: np.ndarray, side_name: str, velocity: np.ndarray = None) -> Dict[str, float]:
    """
    Extract aggregated features for one side (I or II).
    Now includes spatial wavelength features if velocity provided.
    """
    if side_data.shape[0] == 64:
        side_data = side_data.T
    
    N, C = side_data.shape
    assert C == 64, f"Expected 64 channels, got {C}"
    
    all_stat = []
    all_spectral = []
    all_envelope = []
    all_spatial = []
    
    for c in range(C):
        ch_data = side_data[:, c]
        
        # Statistical
        stat = compute_statistical_features(ch_data)
        all_stat.append(list(stat.values()))
        
        # Spectral
        spectral = compute_spectral_features(ch_data)
        all_spectral.append(list(spectral.values()))
        
        # Envelope (shock channels only)
        if c % 2 == 1:
            env = compute_envelope_features(ch_data)
            all_envelope.append(list(env.values()))
        
        # Spatial wavelength features (shock channels only, need velocity)
        if c % 2 == 1 and velocity is not None:
            spatial = extract_spatial_wavelength_features(ch_data, velocity)
            all_spatial.append(list(spatial.values()))
    
    all_stat = np.array(all_stat)
    all_spectral = np.array(all_spectral)
    all_envelope = np.array(all_envelope)
    all_spatial = np.array(all_spatial) if len(all_spatial) > 0 else np.array([])
    
    features = {}
    
    stat_names = ['mean', 'std', 'rms', 'skew', 'kurtosis', 'crest_factor', 'impulse_factor', 'clearance_factor']
    for i, name in enumerate(stat_names):
        features[f'{side_name}_{name}_mean'] = np.nanmean(all_stat[:, i])
        features[f'{side_name}_{name}_std'] = np.nanstd(all_stat[:, i])
        features[f'{side_name}_{name}_max'] = np.nanmax(all_stat[:, i])
        features[f'{side_name}_{name}_min'] = np.nanmin(all_stat[:, i])
    
    spectral_names = [b[2] + '_power' for b in RESONANCE_BANDS] + \
                     [b[2] + '_ratio' for b in RESONANCE_BANDS] + \
                     ['total_power', 'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 'peak_freq']
    for i, name in enumerate(spectral_names):
        features[f'{side_name}_{name}_mean'] = np.nanmean(all_spectral[:, i])
        features[f'{side_name}_{name}_std'] = np.nanstd(all_spectral[:, i])
    
    env_names = ['env_peak_freq', 'env_peak_mag', 'env_band1_ratio', 'env_band2_ratio', 'env_band3_ratio']
    for i, name in enumerate(env_names):
        features[f'{side_name}_{name}_mean'] = np.nanmean(all_envelope[:, i])
        features[f'{side_name}_{name}_std'] = np.nanstd(all_envelope[:, i])
    
    # Spatial wavelength aggregation
    if len(all_spatial) > 0:
        spatial_names = [f'spatial_{b[2]}_energy' for b in SPATIAL_WAVELENGTH_BANDS] + \
                       [f'spatial_{b[2]}_ratio' for b in SPATIAL_WAVELENGTH_BANDS] + \
                       ['spatial_peak_wavelength', 'spatial_peak_energy', 'spatial_dominant_band']
        for i, name in enumerate(spatial_names):
            if i < all_spatial.shape[1]:
                features[f'{side_name}_{name}_mean'] = np.nanmean(all_spatial[:, i])
                features[f'{side_name}_{name}_std'] = np.nanstd(all_spatial[:, i])
    
    return features


def extract_bilateral_asymmetry(side_i_data: np.ndarray, side_ii_data: np.ndarray) -> Dict[str, float]:
    if side_i_data.shape[0] == 64:
        side_i_data = side_i_data.T
    if side_ii_data.shape[0] == 64:
        side_ii_data = side_ii_data.T
    
    features = {}
    
    rms_ratios = []
    energy_ratios = []
    asymmetry_indices = []
    cross_corrs = []
    
    for c in range(64):
        ch_i = side_i_data[:, c]
        ch_ii = side_ii_data[:, c]
        
        rms_i = np.sqrt(np.mean(ch_i ** 2))
        rms_ii = np.sqrt(np.mean(ch_ii ** 2))
        energy_i = np.sum(ch_i ** 2)
        energy_ii = np.sum(ch_ii ** 2)
        
        rms_ratios.append(rms_i / (rms_ii + 1e-10))
        energy_ratios.append(energy_i / (energy_ii + 1e-10))
        asym = (rms_i - rms_ii) / (rms_i + rms_ii + 1e-10)
        asymmetry_indices.append(asym)
        cross_corrs.append(np.corrcoef(ch_i, ch_ii)[0, 1])
    
    features['asym_rms_ratio_mean'] = np.nanmean(rms_ratios)
    features['asym_rms_ratio_std'] = np.nanstd(rms_ratios)
    features['asym_rms_ratio_max'] = np.nanmax(rms_ratios)
    features['asym_rms_ratio_min'] = np.nanmin(rms_ratios)
    
    features['asym_energy_ratio_mean'] = np.nanmean(energy_ratios)
    features['asym_energy_ratio_std'] = np.nanstd(energy_ratios)
    features['asym_energy_ratio_max'] = np.nanmax(energy_ratios)
    
    features['asym_index_mean'] = np.nanmean(asymmetry_indices)
    features['asym_index_std'] = np.nanstd(asymmetry_indices)
    features['asym_index_max'] = np.nanmax(np.abs(asymmetry_indices))
    
    features['asym_cross_corr_mean'] = np.nanmean(cross_corrs)
    features['asym_cross_corr_std'] = np.nanstd(cross_corrs)
    features['asym_cross_corr_min'] = np.nanmin(cross_corrs)
    
    side_i_total = np.mean(side_i_data ** 2)
    side_ii_total = np.mean(side_ii_data ** 2)
    features['side_i_total_energy'] = side_i_total
    features['side_ii_total_energy'] = side_ii_total
    features['total_energy_ratio'] = side_i_total / (side_ii_total + 1e-10)
    features['total_asym_index'] = (side_i_total - side_ii_total) / (side_i_total + side_ii_total + 1e-10)
    
    car_asym = []
    for car in range(8):
        car_rms_i = []
        car_rms_ii = []
        for pos in range(4):
            idx = car * 4 + pos
            ch_i = side_i_data[:, idx]
            ch_ii = side_ii_data[:, idx]
            rms_i = np.sqrt(np.mean(ch_i ** 2))
            rms_ii = np.sqrt(np.mean(ch_ii ** 2))
            car_rms_i.append(rms_i)
            car_rms_ii.append(rms_ii)
        car_asym.append(np.mean(np.array(car_rms_i) / (np.array(car_rms_ii) + 1e-10)))
    
    features['car_asym_mean'] = np.mean(car_asym)
    features['car_asym_std'] = np.std(car_asym)
    features['car_asym_max'] = np.max(car_asym)
    
    return features


def extract_speed_features(speed_data: np.ndarray) -> Dict[str, float]:
    speed_mean = np.mean(speed_data)
    speed_std = np.std(speed_data)
    
    transitions = np.diff(speed_data.astype(int)) != 0
    n_transitions = np.sum(transitions)
    estimated_rpm = (n_transitions / 180) * 6
    
    return {
        'speed_mean': speed_mean,
        'speed_std': speed_std,
        'estimated_rpm': estimated_rpm,
        'n_transitions': n_transitions,
    }


def extract_all_features(data: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """
    Main feature extraction with all upgrades:
    1. Spatial wavelength tracking (order tracking)
    2. Cross-bilateral differential features
    3. Existing bilateral asymmetry + spectral + statistical + envelope
    """
    features = {}
    
    # Speed features
    speed_feats = extract_speed_features(data[:, 0])
    features.update(speed_feats)
    
    # Compute instantaneous velocity for spatial tracking
    velocity = compute_velocity_from_speed_pulses(data[:, 0])
    
    # Side I and Side II data
    side_i = data[:, SIDE_I_CHANNELS]
    side_ii = data[:, SIDE_II_CHANNELS]
    
    # Side-specific features (with spatial tracking)
    side_i_feats = extract_side_features(side_i, 'side_i', velocity)
    side_ii_feats = extract_side_features(side_ii, 'side_ii', velocity)
    features.update(side_i_feats)
    features.update(side_ii_feats)
    
    # Bilateral asymmetry (existing)
    asym_feats = extract_bilateral_asymmetry(side_i, side_ii)
    features.update(asym_feats)
    
    # NEW: Cross-bilateral differential features
    cross_bilateral = extract_cross_bilateral_features(side_i, side_ii)
    features.update(cross_bilateral)
    
    # Convert to array in consistent order
    feature_names = sorted(features.keys())
    feature_vector = np.array([features[name] for name in feature_names], dtype=np.float32)
    
    return feature_vector, feature_names


def load_and_extract(filepath: str) -> Tuple[np.ndarray, List[str]]:
    df = pd.read_csv(filepath, header=None, low_memory=False)
    data = df.iloc[1:].values.astype(np.float32)
    return extract_all_features(data)


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[2]
    candidates = [
        base_dir / "PS3" / "02_Datasets" / "Rail_Corrugation" / "Train" / "Train1.csv",
        base_dir / "data" / "Rail_Corrugation" / "Train" / "Train1.csv",
        base_dir.parent / "NebulaX-Hackathon-ProblemStatement" / "PS3" / "02_Datasets" / "Rail_Corrugation" / "Train" / "Train1.csv",
        Path.home() / "NebulaX-Hackathon-ProblemStatement" / "PS3" / "02_Datasets" / "Rail_Corrugation" / "Train" / "Train1.csv",
    ]
    test_file = next((str(p) for p in candidates if p.exists()), None)
    if test_file:
        feats, names = load_and_extract(test_file)
        print(f"Feature vector shape: {feats.shape}")
        print(f"Number of features: {len(names)}")
        print(f"Feature names: {names[:20]}...")
        print(f"Any NaN: {np.any(np.isnan(feats))}")
        print(f"Any Inf: {np.any(np.isinf(feats))}")
    else:
        print("Sample Train1.csv not found locally; skipping extraction run.")