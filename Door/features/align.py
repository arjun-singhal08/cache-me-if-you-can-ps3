# features/align.py
import numpy as np
from typing import Tuple

def interpolate_to_position_grid(signal: np.ndarray, 
                                  position: np.ndarray, 
                                  pos_grid: np.ndarray) -> np.ndarray:
    """
    Interpolate signal values to a common position grid.
    
    Args:
        signal: Signal values (current, voltage, bemf)
        position: Corresponding position values
        pos_grid: Target position grid (monotonic)
    
    Returns:
        Interpolated signal on pos_grid
    """
    if len(signal) != len(position):
        raise ValueError("Signal and position must have same length")
    
    if len(np.unique(position)) < 2:
        # Position doesn't change, return constant
        return np.full_like(pos_grid, np.mean(signal))
    
    # Sort by position
    sort_idx = np.argsort(position)
    pos_sorted = position[sort_idx]
    sig_sorted = signal[sort_idx]
    
    # Remove duplicate positions (keep first)
    _, unique_idx = np.unique(pos_sorted, return_index=True)
    pos_sorted = pos_sorted[unique_idx]
    sig_sorted = sig_sorted[unique_idx]
    
    if len(pos_sorted) < 2:
        return np.full_like(pos_grid, sig_sorted[0])
    
    # Interpolate
    interp = np.interp(pos_grid, pos_sorted, sig_sorted, 
                       left=sig_sorted[0], right=sig_sorted[-1])
    
    return interp


def align_segments_to_grid(segments_data: list, n_grid: int = 50) -> Tuple[np.ndarray, list]:
    """
    Align multiple segments to a common position grid.
    
    Args:
        segments_data: List of dicts with 'current', 'voltage', 'bemf', 'position'
        n_grid: Number of grid points
    
    Returns:
        pos_grid, list of aligned feature dicts
    """
    # Determine global position range
    all_positions = np.concatenate([s['position'] for s in segments_data])
    pos_min, pos_max = all_positions.min(), all_positions.max()
    
    if pos_max - pos_min < 10:
        pos_min, pos_max = 0, 700
    
    pos_grid = np.linspace(pos_min, pos_max, n_grid)
    
    aligned = []
    for seg in segments_data:
        aligned_seg = {
            'current': interpolate_to_position_grid(seg['current'], seg['position'], pos_grid),
            'voltage': interpolate_to_position_grid(seg['voltage'], seg['position'], pos_grid),
            'bemf': interpolate_to_position_grid(seg['bemf'], seg['position'], pos_grid),
        }
        aligned.append(aligned_seg)
    
    return pos_grid, aligned