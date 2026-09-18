import numpy as np
import pandas as pd
from typing import Tuple, Optional

def validate_input_signal(file_or_data) -> Tuple[bool, str, Optional[np.ndarray]]:
    """
    Validates that the input data is a valid 1D dynamic stress time series.
    
    Returns:
        is_valid: bool
        message: str
        signal_array: np.ndarray or None
    """
    try:
        if isinstance(file_or_data, str):
            df = pd.read_csv(file_or_data, header=None)
            signal = df.iloc[:, 0].values
        elif isinstance(file_or_data, pd.DataFrame):
            signal = file_or_data.iloc[:, 0].values
        elif isinstance(file_or_data, pd.Series):
            signal = file_or_data.values
        elif hasattr(file_or_data, 'read'): # UploadedFile (BytesIO/StringIO)
            df = pd.read_csv(file_or_data, header=None)
            signal = df.iloc[:, 0].values
        else:
            signal = np.array(file_or_data)
            
        if len(signal) == 0:
            return False, "Input file is empty.", None
            
        signal = signal.astype(np.float64)
        
        # Check for NaN / Infs
        nan_count = np.isnan(signal).sum()
        inf_count = np.isinf(signal).sum()
        
        if nan_count > 0 or inf_count > 0:
            return False, f"Signal contains invalid values ({nan_count} NaNs, {inf_count} Infs).", None
            
        if len(signal) < 100:
            return False, f"Signal has only {len(signal)} samples, which is too short for reliable fatigue cycle analysis.", None
            
        return True, f"Valid 1D dynamic stress signal with {len(signal):,} measurements.", signal
        
    except Exception as e:
        return False, f"Error validating signal: {str(e)}", None
