# segmentation/detector.py
import numpy as np
import pandas as pd
from typing import List, Tuple

class DoorSegmenter:
    """Door cycle segmentation using door state transitions + position sub-segmentation."""
    
    def __init__(self, config: dict):
        self.min_segment_len = config.get('min_segment_len', 80)
        self.max_segment_len = config.get('max_segment_len', 350)
        self.refinement_window_ms = config.get('refinement_window_ms', 500)
        self.min_gap = config.get('min_gap_between_segments', 100)
        self.position_threshold = config.get('position_threshold', 100)  # For sub-segment detection
        self.sample_rate = 50.0  # Hz
        
    def fit(self, df: pd.DataFrame, segments_df: pd.DataFrame):
        """Learn optimal parameters from training data."""
        pass
    
    def predict(self, df: pd.DataFrame) -> List[Tuple[pd.Timestamp, pd.Timestamp, str]]:
        """
        Detect door cycles in continuous stream.
        1. Find major cycles using door state flag transitions
        2. Subdivide major cycles using position derivative thresholds
        3. Handle edge cases: start/end of data
        """
        timestamps = df['dt'].values
        door_opening = df['Door is opening'].values
        door_closing = df['Door is closing'].values
        open_cmd = df['Open command'].values
        close_cmd = df['Close command'].values
        position = df['Door leaf position'].values
        current = df['Motor current(mA)'].values
        
        # Step 1: Detect major cycle transitions
        opening_diff = np.diff(door_opening.astype(int), prepend=door_opening[0])
        closing_diff = np.diff(door_closing.astype(int), prepend=door_closing[0])
        open_cmd_diff = np.diff(open_cmd.astype(int), prepend=open_cmd[0])
        close_cmd_diff = np.diff(close_cmd.astype(int), prepend=close_cmd[0])
        
        # Major cycle starts/ends
        open_starts = np.where((opening_diff == 1) | (open_cmd_diff == 1))[0]
        open_ends = np.where(opening_diff == -1)[0]
        close_starts = np.where((closing_diff == 1) | (close_cmd_diff == 1))[0]
        close_ends = np.where(closing_diff == -1)[0]
        
        # Handle edge case: data starts with door already in motion
        if door_opening[0] == 1:
            open_starts = np.concatenate([[0], open_starts])
        if door_closing[0] == 1:
            close_starts = np.concatenate([[0], close_starts])
        
        # Handle edge case: data ends with door still in motion
        if door_opening[-1] == 1:
            open_ends = np.concatenate([open_ends, [len(door_opening) - 1]])
        if door_closing[-1] == 1:
            close_ends = np.concatenate([close_ends, [len(door_closing) - 1]])
        
        # Pair major cycles
        major_cycles = []  # (start_idx, end_idx, operation)
        
        for start_idx in open_starts:
            matching_ends = open_ends[open_ends > start_idx]
            if len(matching_ends) > 0:
                end_idx = matching_ends[0]
                if end_idx - start_idx >= self.min_segment_len:
                    major_cycles.append((start_idx, end_idx, 'Open'))
        
        for start_idx in close_starts:
            matching_ends = close_ends[close_ends > start_idx]
            if len(matching_ends) > 0:
                end_idx = matching_ends[0]
                if end_idx - start_idx >= self.min_segment_len:
                    major_cycles.append((start_idx, end_idx, 'Close'))
        
        # Sort by start time
        major_cycles.sort(key=lambda x: x[0])
        
        # Step 2: Subdivide each major cycle using position derivative
        all_segments = []
        for start_idx, end_idx, operation in major_cycles:
            sub_segments = self._subdivide_cycle(
                position, current, timestamps, start_idx, end_idx, operation
            )
            all_segments.extend(sub_segments)
        
        # Sort all segments by start time
        all_segments.sort(key=lambda x: x[0])
        
        # Merge segments that are too close (same operation)
        merged = self._merge_close_segments(all_segments, timestamps, position)
        
        return merged
    
    def _subdivide_cycle(self, position: np.ndarray, current: np.ndarray,
                         timestamps: np.ndarray, start_idx: int, end_idx: int,
                         operation: str) -> List[Tuple[pd.Timestamp, pd.Timestamp, str]]:
        """Subdivide a major cycle into segments using position derivative."""
        if end_idx - start_idx < self.min_segment_len:
            return []
        
        cycle_pos = position[start_idx:end_idx]
        cycle_current = current[start_idx:end_idx]
        cycle_ts = timestamps[start_idx:end_idx]
        
        # Compute position derivative
        pos_diff = np.diff(cycle_pos, prepend=cycle_pos[0])
        
        if operation == 'Open':
            # Look for position drops (close->open transitions within cycle)
            # A drop of > threshold indicates a sub-segment boundary
            sub_boundaries = np.where(pos_diff < -self.position_threshold)[0]
        else:  # Close
            # Look for position rises (open->close transitions within cycle)
            sub_boundaries = np.where(pos_diff > self.position_threshold)[0]
        
        # Filter boundaries that are too close to start/end or each other
        filtered_boundaries = [0]  # Start of cycle
        for b in sub_boundaries:
            if b > self.min_segment_len // 2 and b < len(cycle_pos) - self.min_segment_len // 2:
                if not filtered_boundaries or b - filtered_boundaries[-1] > self.min_segment_len // 2:
                    filtered_boundaries.append(b)
        filtered_boundaries.append(len(cycle_pos))  # End of cycle
        
        # Create segments from boundaries
        segments = []
        for i in range(len(filtered_boundaries) - 1):
            sub_start = start_idx + filtered_boundaries[i]
            sub_end = start_idx + filtered_boundaries[i + 1]
            
            if sub_end - sub_start >= self.min_segment_len:
                # Refine boundaries
                sub_start, sub_end = self._refine_boundaries(
                    current, position, sub_start, sub_end
                )
                
                if sub_end - sub_start >= self.min_segment_len:
                    segments.append((
                        timestamps[sub_start],
                        timestamps[sub_end - 1],
                        operation
                    ))
        
        return segments
    
    def _refine_boundaries(self, current: np.ndarray, position: np.ndarray,
                           start_idx: int, end_idx: int) -> Tuple[int, int]:
        """Refine segment boundaries using current derivative and position settling."""
        window = int(self.refinement_window_ms * self.sample_rate / 1000)
        
        # Refine start: find current derivative peak near start
        curr_diff = np.abs(np.diff(current, prepend=current[0]))
        search_start = max(0, start_idx - window)
        search_end = min(len(current), start_idx + window)
        
        if search_end > search_start:
            local_max = np.argmax(curr_diff[search_start:search_end])
            new_start = search_start + local_max
            if new_start < end_idx - 20:
                start_idx = new_start
        
        # Refine end: find position settling near end
        pos_diff = np.abs(np.diff(position, prepend=position[0]))
        search_start = max(0, end_idx - window)
        search_end = min(len(position), end_idx + window)
        
        settled_idx = end_idx
        for i in range(search_start, search_end):
            if pos_diff[i] < 5:
                settled_idx = i
                break
        
        if settled_idx > start_idx + 20:
            end_idx = settled_idx
        
        return start_idx, end_idx
    
    def _merge_close_segments(self, segments: List[Tuple[pd.Timestamp, pd.Timestamp, str]],
                              timestamps: np.ndarray, position: np.ndarray) -> List[Tuple[pd.Timestamp, pd.Timestamp, str]]:
        """Merge adjacent segments of same operation with small gaps."""
        if not segments:
            return []
        
        def to_ns(dt):
            if hasattr(dt, 'value'):
                return dt.value
            return pd.Timestamp(dt).value
        
        merged = [segments[0]]
        for start, end, op in segments[1:]:
            last_start, last_end, last_op = merged[-1]
            
            # Check gap in nanoseconds
            gap = (to_ns(start) - to_ns(last_end)) / 1e6  # ms
            
            if gap < self.min_gap and op == last_op:
                # Merge: extend last segment
                merged[-1] = (last_start, end, op)
            else:
                merged.append((start, end, op))
        
        return merged