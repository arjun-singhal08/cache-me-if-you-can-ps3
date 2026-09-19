import numpy as np
from scipy.optimize import minimize
from typing import Tuple, List, Optional
from .rainflow import rainflow_count, compute_miner_damage_vectorized


def fit_sn_parameters(
    filepaths: List[str],
    damages: np.ndarray,
    m_init: float = 5.0,
    C_init: float = 1.0,
    bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
) -> Tuple[float, float]:
    """
    Fit S-N curve parameters (m, C) by minimizing MAPE on training data.
    
    Miner's rule: D = sum(n_i * (amp_i^m) / C)
    We optimize m, C to match reference damages.
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
                dmg = compute_miner_damage_vectorized(amps, counts, np.array([m]), C)[0]
                preds.append(dmg)
        
        preds = np.array(preds)
        # MAPE
        mask = damages > 1e-10
        if not np.any(mask):
            return np.mean(np.abs(preds - damages))
        mape = np.mean(np.abs(damages[mask] - preds[mask]) / np.abs(damages[mask]))
        return mape
    
    result = minimize(
        objective,
        x0=[m_init, C_init],
        bounds=bounds,
        method='L-BFGS-B',
        options={'maxiter': 100, 'ftol': 1e-8}
    )
    
    m_opt, C_opt = result.x
    print(f"S-N fit: m={m_opt:.4f}, C={C_opt:.4f}, MAPE={result.fun:.4f}")
    
    return m_opt, C_opt


def compute_miner_features_for_m(
    filepaths: List[str],
    m_values: List[float],
    C: float = 1.0
) -> np.ndarray:
    """Compute Miner damage features for multiple m values across all files."""
    n_files = len(filepaths)
    n_m = len(m_values)
    features = np.zeros((n_files, n_m))
    
    for i, fp in enumerate(filepaths):
        stress = np.loadtxt(fp, delimiter=',').astype(np.float64)
        amps, means, counts = rainflow_count(stress)
        if len(amps) > 0:
            damages = compute_miner_damage_vectorized(amps, counts, np.array(m_values), C)
            features[i, :] = damages
    
    return features


def find_optimal_m_via_cv(
    filepaths: List[str],
    damages: np.ndarray,
    m_candidates: List[float] = None
) -> float:
    """Find best single m value via cross-validation."""
    if m_candidates is None:
        m_candidates = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    
    best_m = m_candidates[0]
    best_mape = float('inf')
    
    for m in m_candidates:
        miner_features = compute_miner_features_for_m(filepaths, [m])
        preds = miner_features.flatten()
        
        mask = damages > 1e-10
        if np.any(mask):
            mape = np.mean(np.abs(damages[mask] - preds[mask]) / np.abs(damages[mask]))
            print(f"  m={m}: MAPE={mape:.4f}")
            if mape < best_mape:
                best_mape = mape
                best_m = m
    
    print(f"Best m: {best_m} (MAPE={best_mape:.4f})")
    return best_m