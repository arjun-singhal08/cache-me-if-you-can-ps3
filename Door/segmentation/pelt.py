# segmentation/pelt.py
"""
PELT (Pruned Exact Linear Time) change point detection implementation.
Based on Killick, Fearnhead, and Eckley (2012) "Optimal detection of changepoints with a linear computational cost."
"""
import numpy as np
from typing import List, Tuple

class PELT:
    """PELT algorithm for change point detection with L2 cost."""
    
    def __init__(self, penalty: float = 10.0, min_segment_len: int = 1):
        self.penalty = penalty
        self.min_segment_len = min_segment_len
    
    def _cost(self, signal: np.ndarray, start: int, end: int) -> float:
        """L2 cost (sum of squared errors from mean) for segment [start:end)."""
        if end - start < self.min_segment_len:
            return np.inf
        segment = signal[start:end]
        mean = np.mean(segment)
        return np.sum((segment - mean) ** 2)
    
    def fit_predict(self, signal: np.ndarray) -> List[int]:
        """
        Detect change points in signal.
        Returns list of change point indices (end of each segment).
        """
        n = len(signal)
        if n < 2 * self.min_segment_len:
            return []
        
        # Precompute cumulative sums for O(1) segment cost
        cumsum = np.cumsum(signal)
        cumsum2 = np.cumsum(signal ** 2)
        
        def segment_cost(start: int, end: int) -> float:
            if end - start < self.min_segment_len:
                return np.inf
            n_seg = end - start
            sum_x = cumsum[end - 1] - (cumsum[start - 1] if start > 0 else 0)
            sum_x2 = cumsum2[end - 1] - (cumsum2[start - 1] if start > 0 else 0)
            mean = sum_x / n_seg
            return sum_x2 - n_seg * mean * mean
        
        # DP arrays
        F = np.full(n + 1, np.inf)  # F[t] = min cost for signal[:t]
        F[0] = -self.penalty  # Base case
        cp = np.full(n + 1, -1, dtype=int)  # Backtrack: last change point before t
        R = [0]  # Set of candidate last change points
        
        for t in range(1, n + 1):
            # Find best last change point in R
            best_cost = np.inf
            best_k = -1
            
            for k in R:
                cost = F[k] + segment_cost(k, t) + self.penalty
                if cost < best_cost:
                    best_cost = cost
                    best_k = k
            
            F[t] = best_cost
            cp[t] = best_k
            
            # Prune R: remove k where F[k] + cost(k, t') + penalty >= F[t] + penalty for all future t'
            # Simplified pruning condition from paper
            new_R = []
            for k in R:
                if F[k] + segment_cost(k, t) < F[t]:
                    new_R.append(k)
            new_R.append(t)
            R = new_R
        
        # Backtrack to find change points
        change_points = []
        t = n
        while t > 0:
            k = cp[t]
            if k >= 0:
                change_points.append(t)
            t = k
        
        change_points.reverse()
        return change_points[:-1]  # Remove the last one (end of signal)


def detect_position_changes(position: np.ndarray, 
                            penalty: float = 10.0,
                            min_segment_len: int = 50,
                            threshold: float = 50.0) -> List[int]:
    """
    Detect door position changes using PELT on position derivative.
    
    Args:
        position: Door leaf position signal
        penalty: PELT penalty parameter
        min_segment_len: Minimum segment length in samples
        threshold: Minimum position change to consider as valid
    
    Returns:
        List of change point indices
    """
    # Compute position derivative (velocity)
    pos_diff = np.abs(np.diff(position, prepend=position[0]))
    
    # Apply PELT on the derivative
    pelt = PELT(penalty=penalty, min_segment_len=min_segment_len)
    change_points = pelt.fit_predict(pos_diff)
    
    # Filter by actual position change magnitude
    valid_cps = []
    for cp in change_points:
        if cp > 0 and cp < len(position):
            pos_change = abs(position[cp] - position[cp - 1])
            # Also check larger window
            window = min(20, cp, len(position) - cp)
            if window > 0:
                pos_change = max(pos_change, abs(np.mean(position[cp:cp+window]) - np.mean(position[cp-window:cp])))
            if pos_change >= threshold:
                valid_cps.append(cp)
    
    return valid_cps


def refine_boundaries_with_current(current: np.ndarray,
                                    position: np.ndarray,
                                    change_points: List[int],
                                    window_ms: int = 500,
                                    sample_rate_hz: float = 50.0) -> List[Tuple[int, int]]:
    """
    Refine segment boundaries using motor current derivative.
    
    Args:
        current: Motor current signal
        position: Door position signal
        change_points: Initial change points from position
        window_ms: Refinement window in milliseconds
        sample_rate_hz: Sampling rate in Hz
    
    Returns:
        List of (start_idx, end_idx) tuples
    """
    window_samples = int(window_ms * sample_rate_hz / 1000)
    segments = []
    
    # Compute current derivative
    curr_diff = np.abs(np.diff(current, prepend=current[0]))
    
    for i, cp in enumerate(change_points):
        # Determine if this is opening (pos increasing) or closing (pos decreasing)
        if cp > 0:
            pos_before = np.mean(position[max(0, cp-10):cp])
            pos_after = np.mean(position[cp:min(len(position), cp+10)])
            is_opening = pos_after > pos_before
        else:
            is_opening = True
        
        # Refine start: look for current spike before position change
        start_search = max(0, cp - window_samples)
        start_idx = cp
        max_deriv = 0
        for j in range(start_search, cp):
            if curr_diff[j] > max_deriv:
                max_deriv = curr_diff[j]
                start_idx = j
        
        # Refine end: look for position settling after change
        next_cp = change_points[i + 1] if i + 1 < len(change_points) else len(position)
        end_search = min(len(position), cp + window_samples)
        end_idx = cp
        # Find where position derivative becomes small
        pos_deriv = np.abs(np.diff(position, prepend=position[0]))
        for j in range(cp, min(end_search, next_cp)):
            if pos_deriv[j] < 5:  # Threshold for "settled"
                end_idx = j
                break
        else:
            end_idx = min(cp + window_samples, next_cp)
        
        if end_idx > start_idx + 20:  # Minimum segment length
            segments.append((start_idx, end_idx))
    
    return segments