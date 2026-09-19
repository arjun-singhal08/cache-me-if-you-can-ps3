"""
ASTM E1049-85 Rainflow Cycle Counting with Mean-Stress Correction
Multi-Exponent Palmgren-Miner Damage + Spectral Fatigue Estimators
"""
import numpy as np
from numba import njit, prange
from typing import Tuple, Dict, List, Optional
from scipy import signal
from scipy.stats import moment
import warnings
warnings.filterwarnings('ignore')


@njit(cache=True)
def _rainflow_astm(stress: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    ASTM E1049-85 compliant 4-point rainflow cycle counting.
    Returns: (ranges, means, counts) where range = peak - valley
    """
    n = len(stress)
    if n < 4:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    # Find turning points (peaks and valleys)
    tp = np.zeros(n, dtype=np.bool_)
    tp[0] = True
    tp[-1] = True

    for i in range(1, n - 1):
        if (stress[i] > stress[i - 1] and stress[i] > stress[i + 1]) or \
           (stress[i] < stress[i - 1] and stress[i] < stress[i + 1]):
            tp[i] = True

    turning_points = stress[tp]
    m = len(turning_points)

    if m < 3:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    # 4-point rainflow counting
    stack = np.zeros(m, dtype=np.float64)
    stack_idx = 0
    
    ranges_list = []
    means_list = []
    counts_list = []

    for i in range(m):
        stack[stack_idx] = turning_points[i]
        stack_idx += 1

        while stack_idx >= 3:
            s3 = stack[stack_idx - 1]
            s2 = stack[stack_idx - 2]
            s1 = stack[stack_idx - 3]
            s0 = stack[stack_idx - 4] if stack_idx >= 4 else 0.0

            range_23 = abs(s3 - s2)
            range_12 = abs(s2 - s1)
            range_01 = abs(s1 - s0) if stack_idx >= 4 else 1e30

            if range_12 <= range_23 and range_12 <= range_01:
                rng = range_12
                mean_stress = (s2 + s1) / 2.0
                ranges_list.append(rng)
                means_list.append(mean_stress)
                counts_list.append(1.0)

                # Remove middle point
                stack[stack_idx - 2] = stack[stack_idx - 1]
                stack_idx -= 1
            else:
                break

    # Residual cycles (half cycles)
    while stack_idx >= 2:
        s2 = stack[stack_idx - 1]
        s1 = stack[stack_idx - 2]
        rng = abs(s2 - s1)
        mean_stress = (s2 + s1) / 2.0
        ranges_list.append(rng)
        means_list.append(mean_stress)
        counts_list.append(0.5)
        stack_idx -= 1

    return np.array(ranges_list, dtype=np.float64), np.array(means_list, dtype=np.float64), np.array(counts_list, dtype=np.float64)


def rainflow_astm(stress: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ASTM E1049-85 rainflow counting with NaN handling."""
    stress = np.asarray(stress, dtype=np.float64)
    stress = stress[~np.isnan(stress)]
    if len(stress) < 4:
        return np.array([]), np.array([]), np.array([])
    return _rainflow_astm(stress)


def goodman_correction(ranges: np.ndarray, means: np.ndarray, sigma_uts: float) -> np.ndarray:
    """
    Goodman mean-stress correction:
    Δσ_eff = Δσ / (1 - σ_mean / σ_uts)
    """
    # Handle compressive mean stress (no correction needed)
    mask = means > 0
    corrected = ranges.copy()
    corrected[mask] = ranges[mask] / (1.0 - means[mask] / sigma_uts)
    # Cap at reasonable value to avoid numerical issues
    corrected = np.clip(corrected, 0, ranges * 10.0)
    return corrected


def gerber_correction(ranges: np.ndarray, means: np.ndarray, sigma_uts: float) -> np.ndarray:
    """
    Gerber parabolic mean-stress correction:
    Δσ_gerber = Δσ / (1 - (σ_mean / σ_uts)^2)
    """
    mask = means > 0
    corrected = ranges.copy()
    corrected[mask] = ranges[mask] / (1.0 - (means[mask] / sigma_uts) ** 2)
    corrected = np.clip(corrected, 0, ranges * 10.0)
    return corrected


def swt_correction(ranges: np.ndarray, means: np.ndarray) -> np.ndarray:
    """
    Smith-Watson-Topper (SWT) parameter:
    σ_max = σ_mean + σ_a = σ_mean + Δσ/2
    SWT = σ_max * σ_a = σ_max * (Δσ/2)
    Equivalent fully-reversed amplitude: Δσ_swt = 2 * sqrt(SWT)
    """
    stress_amplitude = ranges / 2.0
    sigma_max = means + stress_amplitude
    swt = sigma_max * stress_amplitude
    return 2.0 * np.sqrt(np.maximum(swt, 0))


def morrow_correction(ranges: np.ndarray, means: np.ndarray, sigma_f: float) -> np.ndarray:
    """
    Morrow mean-stress correction using fatigue strength coefficient.
    """
    mask = means > 0
    corrected = ranges.copy()
    corrected[mask] = ranges[mask] * (1.0 - means[mask] / sigma_f)
    corrected = np.clip(corrected, 0, ranges * 2.0)
    return corrected


def compute_miner_damage(ranges: np.ndarray, counts: np.ndarray, m: float, C: float = 1.0) -> float:
    """Standard Palmgren-Miner linear damage accumulation."""
    if len(ranges) == 0:
        return 0.0
    return np.sum(counts * (ranges ** m)) / C


def compute_multi_exponent_miner(
    ranges: np.ndarray,
    means: np.ndarray,
    counts: np.ndarray,
    m_values: List[float],
    sigma_uts: float,
    corrections: List[str] = ['none', 'goodman', 'gerber', 'swt']
) -> Dict[str, float]:
    """
    Compute Miner damage for multiple S-N exponents and mean-stress corrections.
    """
    results = {}
    
    for corr_name in corrections:
        if corr_name == 'none':
            eff_ranges = ranges
        elif corr_name == 'goodman':
            eff_ranges = goodman_correction(ranges, means, sigma_uts)
        elif corr_name == 'gerber':
            eff_ranges = gerber_correction(ranges, means, sigma_uts)
        elif corr_name == 'swt':
            eff_ranges = swt_correction(ranges, means)
        else:
            continue
            
        for m in m_values:
            key = f'miner_{corr_name}_m{m:.1f}'
            results[key] = compute_miner_damage(eff_ranges, counts, m)
    
    return results


def compute_spectral_moments(stress: np.ndarray, fs: float = 1.0) -> Dict[str, float]:
    """
    Compute spectral moments for narrow-band and Dirlik estimators.
    m_k = ∫ f^k * PSD(f) df
    """
    n = len(stress)
    if n < 10:
        return {k: 0.0 for k in ['m0', 'm1', 'm2', 'm4']}
    
    nperseg = min(4096, n // 4)
    freqs, psd = signal.welch(stress, fs=fs, nperseg=nperseg)
    
    from scipy.integrate import trapezoid
    m0 = trapezoid(psd, freqs)
    m1 = trapezoid(freqs * psd, freqs)
    m2 = trapezoid(freqs**2 * psd, freqs)
    m4 = trapezoid(freqs**4 * psd, freqs)
    
    return {'m0': m0, 'm1': m1, 'm2': m2, 'm4': m4}


def rayleigh_damage(moments: Dict[str, float], m: float, C: float = 1.0) -> float:
    """
    Narrow-band Rayleigh approximation for fatigue damage.
    D = (1/C) * (2*sqrt(m0))^m * Γ(1 + m/2) * f0 * T
    where f0 = sqrt(m2/m0) / (2π)
    """
    m0 = moments['m0']
    m2 = moments['m2']
    
    if m0 <= 0 or m2 <= 0:
        return 0.0
    
    sigma_rms = np.sqrt(m0)
    f0 = np.sqrt(m2 / m0) / (2 * np.pi)
    
    from scipy.special import gamma
    damage_rate = (1/C) * (2 * sigma_rms) ** m * gamma(1 + m/2) * f0
    return damage_rate


def dirlik_damage(moments: Dict[str, float], m: float, C: float = 1.0) -> float:
    """
    Dirlik's empirical spectral fatigue method.
    More accurate for wide-band processes.
    """
    m0 = moments['m0']
    m1 = moments['m1']
    m2 = moments['m2']
    m4 = moments['m4']
    
    if m0 <= 0:
        return 0.0
    
    # Dirlik parameters
    xm = m1 / m0 * np.sqrt(m2 / m4)
    alpha2 = xm - (xm**2) / (1 + xm**2)
    alpha1 = 1 - alpha2 - (1 - xm) / (1 + xm**2)
    
    gamma = m2 / (m0 * np.sqrt(m4))
    z = m0 * np.sqrt(m4) / m2 if m2 > 0 else 0
    
    if z <= 0 or gamma <= 0:
        return 0.0
    
    # Dirlik PDF parameters
    R = alpha1 / (gamma * z) + alpha2 * (1 - alpha1 - alpha2) / (1 - alpha2) * alpha2
    
    # Damage calculation using Dirlik's method
    # D = f0 * ∫ (S^m / C) * p(S) dS
    # where p(S) is Dirlik's PDF
    
    # Approximate using characteristic values
    f0 = np.sqrt(m2 / m0) / (2 * np.pi)
    
    # Use multiple stress ranges to approximate the integral
    S_ranges = np.logspace(np.log10(0.1 * np.sqrt(m0)), np.log10(5 * np.sqrt(m0)), 100)
    
    # Dirlik PDF
    pdf = (alpha1 / z * np.exp(-S_ranges / z) + 
           alpha2 * S_ranges / (gamma * m0) * np.exp(-S_ranges**2 / (2 * gamma**2 * m0)) +
           (1 - alpha1 - alpha2) * S_ranges / m0 * np.exp(-S_ranges**2 / (2 * m0)))
    
    from scipy.special import gamma as gamma_fn
    from scipy.integrate import trapezoid
    damage_rate = f0 / C * trapezoid(S_ranges**m * pdf, S_ranges)
    return damage_rate


def tovo_benasciutto_damage(moments: Dict[str, float], m: float, C: float = 1.0) -> float:
    """
    Tovo-Benasciutto bi-modal spectral method.
    """
    m0 = moments['m0']
    m1 = moments['m1']
    m2 = moments['m2']
    m4 = moments['m4']
    
    if m0 <= 0:
        return 0.0
    
    # Bandwidth parameter
    alpha = m2 / np.sqrt(m0 * m4)
    
    if alpha <= 0 or alpha >= 1:
        return 0.0
    
    # Narrow-band and wide-band components
    f0 = np.sqrt(m2 / m0) / (2 * np.pi)
    
    # Tovo-Benasciutto combination
    b = alpha * (1 - alpha)
    from scipy.special import gamma as gamma_fn
    damage_rate = f0 / C * (2 * np.sqrt(m0))**m * (
        (1 - b) * alpha**m * gamma_fn(1 + m/2) + 
        b * gamma_fn(1 + m)
    )
    
    return damage_rate


def extract_physics_features_from_stress(
    stress: np.ndarray,
    m_values: List[float] = None,
    sigma_uts_range: Tuple[float, float] = (600.0, 900.0),
    fs: float = 1.0
) -> Dict[str, float]:
    """
    Extract comprehensive physics-based fatigue features from stress time series.
    """
    if m_values is None:
        m_values = [3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0, 10.0]
    
    features = {}
    
    # Basic statistics
    stress_clean = stress[~np.isnan(stress)]
    if len(stress_clean) == 0:
        # Return zeros for all features
        return _get_zero_features(m_values)
    
    features['mean'] = float(np.mean(stress_clean))
    features['std'] = float(np.std(stress_clean))
    features['rms'] = float(np.sqrt(np.mean(stress_clean**2)))
    features['skew'] = float(np.mean(((stress_clean - features['mean']) / features['std'])**3)) if features['std'] > 0 else 0.0
    features['kurt'] = float(np.mean(((stress_clean - features['mean']) / features['std'])**4)) if features['std'] > 0 else 0.0
    features['max'] = float(np.max(stress_clean))
    features['min'] = float(np.min(stress_clean))
    features['ptp'] = float(features['max'] - features['min'])
    
    # Rainflow counting
    ranges, means, counts = rainflow_astm(stress_clean)
    features['n_cycles'] = float(np.sum(counts))
    
    if len(ranges) > 0:
        features['rf_range_mean'] = float(np.average(ranges, weights=counts))
        features['rf_range_std'] = float(np.sqrt(np.average((ranges - features['rf_range_mean'])**2, weights=counts)))
        features['rf_range_max'] = float(np.max(ranges))
        features['rf_mean_mean'] = float(np.average(means, weights=counts))
        features['rf_mean_std'] = float(np.sqrt(np.average((means - features['rf_mean_mean'])**2, weights=counts)))
    else:
        features['rf_range_mean'] = 0.0
        features['rf_range_std'] = 0.0
        features['rf_range_max'] = 0.0
        features['rf_mean_mean'] = 0.0
        features['rf_mean_std'] = 0.0
    
    # Spectral moments
    moments = compute_spectral_moments(stress_clean, fs)
    features['m0'] = float(moments['m0']) if not np.isnan(moments['m0']) else 0.0
    features['m1'] = float(moments['m1']) if not np.isnan(moments['m1']) else 0.0
    features['m2'] = float(moments['m2']) if not np.isnan(moments['m2']) else 0.0
    features['m4'] = float(moments['m4']) if not np.isnan(moments['m4']) else 0.0
    
    # Multi-exponent Miner with mean-stress corrections
    sigma_uts_opt = optimize_sigma_uts(ranges, means, counts, m_values, sigma_uts_range)
    
    miner_results = compute_multi_exponent_miner(
        ranges, means, counts, m_values, sigma_uts_opt,
        corrections=['none', 'goodman', 'gerber', 'swt']
    )
    features.update(miner_results)
    features['sigma_uts_opt'] = float(sigma_uts_opt)
    
    # Spectral fatigue estimators
    for m in m_values:
        features[f'rayleigh_m{m:.1f}'] = float(rayleigh_damage(moments, m))
        features[f'dirlik_m{m:.1f}'] = float(dirlik_damage(moments, m))
        features[f'tovo_m{m:.1f}'] = float(tovo_benasciutto_damage(moments, m))
    
    # Ensure no NaN values
    for k, v in features.items():
        if isinstance(v, float) and np.isnan(v):
            features[k] = 0.0
    
    return features


def _get_zero_features(m_values: List[float]) -> Dict[str, float]:
    """Return zero-filled features dict."""
    features = {
        'mean': 0.0, 'std': 0.0, 'rms': 0.0, 'skew': 0.0, 'kurt': 0.0,
        'max': 0.0, 'min': 0.0, 'ptp': 0.0, 'n_cycles': 0.0,
        'rf_range_mean': 0.0, 'rf_range_std': 0.0, 'rf_range_max': 0.0,
        'rf_mean_mean': 0.0, 'rf_mean_std': 0.0,
        'm0': 0.0, 'm1': 0.0, 'm2': 0.0, 'm4': 0.0,
        'sigma_uts_opt': 750.0
    }
    for m in m_values:
        for corr in ['none', 'goodman', 'gerber', 'swt']:
            features[f'miner_{corr}_m{m:.1f}'] = 0.0
        features[f'rayleigh_m{m:.1f}'] = 0.0
        features[f'dirlik_m{m:.1f}'] = 0.0
        features[f'tovo_m{m:.1f}'] = 0.0
    return features


def optimize_sigma_uts(
    ranges: np.ndarray,
    means: np.ndarray,
    counts: np.ndarray,
    m_values: List[float],
    sigma_uts_range: Tuple[float, float]
) -> float:
    """
    Optimize ultimate tensile strength for best mean-stress correction.
    For now, return mid-range value. Can be optimized per-file if labels available.
    """
    return (sigma_uts_range[0] + sigma_uts_range[1]) / 2.0


def extract_features_batch(
    filepaths: List[str],
    m_values: List[float] = None,
    sigma_uts_range: Tuple[float, float] = (600.0, 900.0),
    n_jobs: int = -1,
    fs: float = 1.0
) -> pd.DataFrame:
    """
    Extract physics features from multiple files in parallel.
    """
    import pandas as pd
    from joblib import Parallel, delayed
    from tqdm import tqdm
    
    if m_values is None:
        m_values = [3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0, 10.0]
    
    def process_file(fp: str) -> Dict[str, float]:
        stress = np.loadtxt(fp, delimiter=',').astype(np.float64)
        feats = extract_physics_features_from_stress(
            stress, m_values, sigma_uts_range, fs
        )
        from pathlib import Path
        feats['file_id'] = Path(fp).name
        return feats
    
    if n_jobs == 1:
        results = [process_file(fp) for fp in tqdm(filepaths, desc="Extracting physics features")]
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_file)(fp) for fp in tqdm(filepaths, desc="Extracting physics features")
        )
    
    return pd.DataFrame(results)