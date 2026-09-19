import numpy as np
from scipy.optimize import minimize
from typing import Tuple, List, Optional
from .rainflow import rainflow_count


def compute_miner_damage_differentiable(amplitudes: np.ndarray, counts: np.ndarray, m: float, C: float) -> float:
    """Compute Miner damage for given m, C."""
    if len(amplitudes) == 0:
        return 0.0
    return np.sum(counts * (amplitudes ** m)) / C


def fit_sn_parameters_joint(
    filepaths: List[str],
    damages: np.ndarray,
    m_init: float = 5.0,
    C_init: float = 1.0,
    bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    method: str = 'L-BFGS-B'
) -> Tuple[float, float]:
    """
    Fit S-N curve parameters (m, C) by minimizing MAPE on training data.
    Uses joint optimization with gradient-based method.
    """
    if bounds is None:
        bounds = ((2.0, 15.0), (1e-6, 1e6))
    
    # Pre-compute rainflow cycles for all files
    print("Pre-computing rainflow cycles for S-N fitting...")
    all_amplitudes = []
    all_counts = []
    
    for fp in filepaths:
        stress = np.loadtxt(fp, delimiter=',').astype(np.float64)
        amps, means, counts = rainflow_count(stress)
        all_amplitudes.append(amps)
        all_counts.append(counts)
    
    def objective(params: np.ndarray) -> float:
        m, C = params
        if m <= 0 or C <= 0:
            return 1e6
        
        preds = []
        for amps, counts in zip(all_amplitudes, all_counts):
            if len(amps) == 0:
                preds.append(0.0)
            else:
                dmg = compute_miner_damage_differentiable(amps, counts, m, C)
                preds.append(dmg)
        
        preds = np.array(preds)
        # MAPE with protection
        mask = damages > 1e-10
        if not np.any(mask):
            return np.mean(np.abs(preds - damages))
        mape = np.mean(np.abs(damages[mask] - preds[mask]) / np.abs(damages[mask]))
        return mape
    
    result = minimize(
        objective,
        x0=[m_init, C_init],
        bounds=bounds,
        method=method,
        options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
    )
    
    m_opt, C_opt = result.x
    print(f"S-N joint fit: m={m_opt:.4f}, C={C_opt:.4f}, MAPE={result.fun:.4f}")
    
    return m_opt, C_opt


def fit_sn_parameters_log_space(
    filepaths: List[str],
    damages: np.ndarray,
    m_init: float = 5.0,
    logC_init: float = 0.0,
    bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
) -> Tuple[float, float]:
    """
    Fit S-N parameters in log space for better numerical stability.
    Optimizes (m, log10(C)).
    """
    if bounds is None:
        bounds = ((2.0, 15.0), (-6.0, 6.0))
    
    # Pre-compute rainflow cycles
    all_amplitudes = []
    all_counts = []
    
    for fp in filepaths:
        stress = np.loadtxt(fp, delimiter=',').astype(np.float64)
        amps, means, counts = rainflow_count(stress)
        all_amplitudes.append(amps)
        all_counts.append(counts)
    
    def objective(params: np.ndarray) -> float:
        m, logC = params
        C = 10 ** logC
        if m <= 0 or C <= 0:
            return 1e6
        
        preds = []
        for amps, counts in zip(all_amplitudes, all_counts):
            if len(amps) == 0:
                preds.append(0.0)
            else:
                dmg = compute_miner_damage_differentiable(amps, counts, m, C)
                preds.append(dmg)
        
        preds = np.array(preds)
        mask = damages > 1e-10
        if not np.any(mask):
            return np.mean(np.abs(preds - damages))
        mape = np.mean(np.abs(damages[mask] - preds[mask]) / np.abs(damages[mask]))
        return mape
    
    result = minimize(
        objective,
        x0=[m_init, logC_init],
        bounds=bounds,
        method='L-BFGS-B',
        options={'maxiter': 200, 'ftol': 1e-10}
    )
    
    m_opt, logC_opt = result.x
    C_opt = 10 ** logC_opt
    print(f"S-N log-space fit: m={m_opt:.4f}, C={C_opt:.4f}, MAPE={result.fun:.4f}")
    
    return m_opt, C_opt


def compute_miner_features_optimized(
    filepaths: List[str],
    m: float,
    C: float
) -> np.ndarray:
    """Compute optimized Miner damage features for all files."""
    n_files = len(filepaths)
    features = np.zeros((n_files, 1))
    
    for i, fp in enumerate(filepaths):
        stress = np.loadtxt(fp, delimiter=',').astype(np.float64)
        amps, means, counts = rainflow_count(stress)
        if len(amps) > 0:
            features[i, 0] = compute_miner_damage_differentiable(amps, counts, m, C)
    
    return features


def compute_miner_features_multi_m(
    filepaths: List[str],
    m_values: List[float],
    C: float = 1.0
) -> np.ndarray:
    """Compute Miner damage for multiple m values (fixed C)."""
    n_files = len(filepaths)
    n_m = len(m_values)
    features = np.zeros((n_files, n_m))
    
    for i, fp in enumerate(filepaths):
        stress = np.loadtxt(fp, delimiter=',').astype(np.float64)
        amps, means, counts = rainflow_count(stress)
        if len(amps) > 0:
            for j, m in enumerate(m_values):
                features[i, j] = compute_miner_damage_differentiable(amps, counts, m, C)
    
    return features


def get_physics_baseline_predictions(
    filepaths: List[str],
    m: float,
    C: float
) -> np.ndarray:
    """Get pure physics-based predictions using fitted S-N parameters."""
    preds = []
    for fp in filepaths:
        stress = np.loadtxt(fp, delimiter=',').astype(np.float64)
        amps, means, counts = rainflow_count(stress)
        if len(amps) > 0:
            dmg = compute_miner_damage_differentiable(amps, counts, m, C)
            preds.append(dmg)
        else:
            preds.append(0.0)
    return np.array(preds)