"""
Angle-domain resampling for rail corrugation detection.
Converts time-domain vibration signals to angle-domain using wheel rotation pulses.
"""

import numpy as np
from scipy import interpolate
from typing import Tuple, Optional


def compute_angle_from_pulses(pulses: np.ndarray, teeth_per_rev: int = 90) -> np.ndarray:
    """
    Compute cumulative angle (radians) from binary pulse train.
    
    Each tooth passing the sensor creates a transition. With 90 teeth,
    each transition = 1/180 revolution = pi/90 radians.
    
    Args:
        pulses: Binary array (0/1) from speed sensor, shape [T]
        teeth_per_rev: Number of teeth on wheel (default 90)
    
    Returns:
        Cumulative angle in radians, shape [T]
    """
    # Detect transitions (both rising and falling edges)
    transitions = np.diff(pulses.astype(np.int32)) != 0
    
    # Each transition = 1/(2*teeth) revolution
    angle_per_transition = np.pi / teeth_per_rev  # radians
    
    # Cumulative angle
    cumulative = np.cumsum(transitions) * angle_per_transition
    
    # Prepend 0 for first sample
    angle = np.concatenate([[0.0], cumulative])
    
    return angle


def resample_to_angle_domain(
    signals: np.ndarray, 
    angles: np.ndarray,
    target_num_samples: int = 1800,
    angle_range: Optional[Tuple[float, float]] = None
) -> np.ndarray:
    """
    Resample signals from time-domain to constant-angle domain.
    
    Args:
        signals: [T, C] or [C, T] - vibration signals
        angles: [T] - cumulative angle in radians (monotonic non-decreasing)
        target_num_samples: Number of output samples per revolution * num_revs
        angle_range: (min_angle, max_angle) to use. If None, use full range.
    
    Returns:
        Resampled signals [target_num_samples, C]
    """
    # Ensure signals is [T, C]
    if signals.shape[0] != len(angles):
        signals = signals.T  # [C, T] -> [T, C]
    
    T, C = signals.shape
    
    # Handle edge case: constant speed (no transitions)
    if angles[-1] <= angles[0] + 1e-6:
        # Return interpolated constant signal
        return np.tile(signals.mean(axis=0), (target_num_samples, 1))
    
    # Determine angle range
    if angle_range is None:
        min_angle = angles[0]
        max_angle = angles[-1]
    else:
        min_angle, max_angle = angle_range
    
    # Create target angle grid (uniform spacing)
    target_angles = np.linspace(min_angle, max_angle, target_num_samples)
    
    # Interpolate each channel
    resampled = np.zeros((target_num_samples, C), dtype=np.float32)
    
    for c in range(C):
        # Use linear interpolation
        valid = np.isfinite(signals[:, c]) & np.isfinite(angles)
        if valid.sum() < 4:
            resampled[:, c] = 0.0
            continue
        
        sig_valid = signals[valid, c]
        ang_valid = angles[valid]
        
        # Remove duplicate angles (flat regions where speed=0)
        # interp1d requires strictly increasing x values
        _, unique_idx = np.unique(ang_valid, return_index=True)
        sig_unique = sig_valid[unique_idx]
        ang_unique = ang_valid[unique_idx]
        
        if len(ang_unique) < 2:
            resampled[:, c] = sig_unique[0] if len(sig_unique) > 0 else 0.0
            continue
        
        try:
            f = interpolate.interp1d(
                ang_unique, 
                sig_unique,
                kind='linear',
                bounds_error=False,
                fill_value='extrapolate'
            )
            resampled[:, c] = f(target_angles).astype(np.float32)
        except ValueError:
            # Fallback: nearest neighbor
            indices = np.searchsorted(ang_unique, target_angles)
            indices = np.clip(indices, 0, len(ang_unique) - 1)
            resampled[:, c] = sig_unique[indices].astype(np.float32)
    
    return resampled  # [target_num_samples, C]


def preprocess_file(
    filepath: str,
    target_num_samples: int = 1800,
    teeth_per_rev: int = 90
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and preprocess a single CSV file.
    
    Returns:
        side_i: [target_num_samples, 64] - Side I channels (vib + shock for pos 1,3,5,7)
        side_ii: [target_num_samples, 64] - Side II channels (vib + shock for pos 2,4,6,8)
    """
    import pandas as pd
    
    # Load CSV - first row is header, rest is data
    df = pd.read_csv(filepath, header=None, low_memory=False)
    
    # First row contains column names
    headers = df.iloc[0].values
    data = df.iloc[1:].values.astype(np.float32)  # [T, 129]
    
    # Column 0: speed pulses
    speed_pulses = data[:, 0]
    
    # Columns 1-128: 64 axle boxes * 2 (vib, shock)
    # Order: Car1-P1-vib, Car1-P1-shock, Car1-P2-vib, Car1-P2-shock, ...
    sensor_data = data[:, 1:]  # [T, 128]
    
    # Compute angle from speed pulses
    angles = compute_angle_from_pulses(speed_pulses, teeth_per_rev)
    
    # Resample to angle domain
    resampled = resample_to_angle_domain(
        sensor_data, angles, target_num_samples
    )  # [target_num_samples, 128]
    
    # Split into Side I and Side II
    # 8 cars * 8 positions * 2 (vib, shock) = 128 channels
    # Side I: positions 1,3,5,7 (0-indexed: 0,2,4,6)
    # Side II: positions 2,4,6,8 (0-indexed: 1,3,5,7)
    side_i_channels = []
    side_ii_channels = []
    
    for car in range(8):
        for pos in range(8):
            base_idx = (car * 8 + pos) * 2  # vib, shock
            vib_idx = base_idx
            shock_idx = base_idx + 1
            
            if pos in [0, 2, 4, 6]:  # Side I
                side_i_channels.append(vib_idx)
                side_i_channels.append(shock_idx)
            else:  # Side II
                side_ii_channels.append(vib_idx)
                side_ii_channels.append(shock_idx)
    
    side_i = resampled[:, side_i_channels]   # [target_num_samples, 64]
    side_ii = resampled[:, side_ii_channels] # [target_num_samples, 64]
    
    return side_i, side_ii


if __name__ == "__main__":
    # Test with a sample file
    test_file = "NebulaX-Hackathon-ProblemStatement/PS3/02_Datasets/Rail_Corrugation/Train/Train1.csv"
    side_i, side_ii = preprocess_file(test_file)
    print(f"Side I shape: {side_i.shape}")
    print(f"Side II shape: {side_ii.shape}")
    print(f"Side I range: [{side_i.min():.4f}, {side_i.max():.4f}]")
    print(f"Side II range: [{side_ii.min():.4f}, {side_ii.max():.4f}]")