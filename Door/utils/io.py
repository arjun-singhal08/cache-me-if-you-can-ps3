# utils/io.py
import pandas as pd
import numpy as np
from typing import List, Tuple

def parse_datetime(s: str) -> pd.Timestamp:
    """Parse datetime string like '2023-7-5-0-0-0-0' to pandas Timestamp."""
    parts = s.split('-')
    return pd.Timestamp(
        year=int(parts[0]),
        month=int(parts[1]),
        day=int(parts[2]),
        hour=int(parts[3]),
        minute=int(parts[4]),
        second=int(parts[5]),
        microsecond=int(parts[6]) * 1000
    )

def load_train_data(data_path: str, segments_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load training data and segment labels."""
    df = pd.read_csv(data_path)
    segments = pd.read_csv(segments_path)
    
    df['dt'] = df['Datetime'].apply(parse_datetime)
    segments['start_dt'] = segments['start_time'].apply(parse_datetime)
    segments['end_dt'] = segments['end_time'].apply(parse_datetime)
    
    return df, segments

def load_test_data(data_path: str) -> pd.DataFrame:
    """Load test data."""
    df = pd.read_csv(data_path)
    df['dt'] = df['Datetime'].apply(parse_datetime)
    return df

def format_datetime(dt) -> str:
    """Format Timestamp back to the required string format."""
    if hasattr(dt, 'year'):
        # pandas Timestamp
        return f"{dt.year}-{dt.month}-{dt.day}-{dt.hour}-{dt.minute}-{dt.second}-{dt.microsecond // 1000}"
    else:
        # numpy datetime64
        ts = pd.Timestamp(dt)
        return f"{ts.year}-{ts.month}-{ts.day}-{ts.hour}-{ts.minute}-{ts.second}-{ts.microsecond // 1000}"

def save_predictions(predictions: List[Tuple[pd.Timestamp, pd.Timestamp, str]], output_path: str):
    """Save predictions to CSV in required format."""
    rows = []
    for start, end, label in predictions:
        rows.append({
            'start_time': format_datetime(start),
            'end_time': format_datetime(end),
            'prediction': label
        })
    pd.DataFrame(rows).to_csv(output_path, index=False)