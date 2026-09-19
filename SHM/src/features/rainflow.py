import numpy as np
from numba import njit
from typing import Tuple, List


@njit(cache=True)
def _rainflow_numba(stress):
    """
    Numba-accelerated rainflow counting algorithm (4-point method).
    Returns: (amplitudes, mean_stresses, counts)
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

    # Rainflow 4-point counting
    stack = np.zeros(m, dtype=np.float64)
    stack_idx = 0
    
    # Pre-allocate lists (use Python lists for numba compatibility)
    amplitudes_list = []
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
                amp = range_12 / 2.0
                mean_stress = (s2 + s1) / 2.0
                amplitudes_list.append(amp)
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
        amp = abs(s2 - s1) / 2.0
        mean_stress = (s2 + s1) / 2.0
        amplitudes_list.append(amp)
        means_list.append(mean_stress)
        counts_list.append(0.5)
        stack_idx -= 1

    # Convert lists to arrays
    amplitudes = np.array(amplitudes_list, dtype=np.float64)
    means = np.array(means_list, dtype=np.float64)
    counts = np.array(counts_list, dtype=np.float64)

    return amplitudes, means, counts


def rainflow_count(stress: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rainflow cycle counting for a stress time series.
    
    Parameters
    ----------
    stress : np.ndarray
        1D array of stress values (float32 or float64)
    
    Returns
    -------
    amplitudes : np.ndarray
        Stress amplitudes for each cycle
    mean_stresses : np.ndarray
        Mean stresses for each cycle
    counts : np.ndarray
        Cycle counts (1.0 for full cycles, 0.5 for half cycles)
    """
    stress = np.asarray(stress, dtype=np.float64)
    # Remove NaNs if any
    stress = stress[~np.isnan(stress)]
    if len(stress) < 4:
        return np.array([]), np.array([]), np.array([])
    return _rainflow_numba(stress)


def rainflow_count_downsampled(stress: np.ndarray, factor: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rainflow on downsampled signal for speed."""
    if factor > 1:
        stress = stress[::factor]
    return rainflow_count(stress)


def compute_miner_damage(amplitudes: np.ndarray, counts: np.ndarray, m: float, C: float = 1.0) -> float:
    """
    Compute Miner's linear cumulative damage.
    D = sum(count_i * (amp_i^m) / C)
    """
    if len(amplitudes) == 0:
        return 0.0
    return np.sum(counts * (amplitudes ** m)) / C


def compute_miner_damage_vectorized(amplitudes: np.ndarray, counts: np.ndarray, m_values: np.ndarray, C: float = 1.0) -> np.ndarray:
    """Compute damage for multiple m values at once."""
    if len(amplitudes) == 0:
        return np.zeros(len(m_values))
    # Shape: (n_m_values, n_cycles) -> sum over cycles
    return np.sum(counts * (amplitudes[:, None] ** m_values).T, axis=1) / C