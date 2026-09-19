import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import joblib


def load_train_labels(labels_path: str) -> pd.DataFrame:
    """Load training labels CSV."""
    df = pd.read_csv(labels_path)
    # Rename 'filename' to 'file_id' for consistency
    if 'filename' in df.columns:
        df = df.rename(columns={'filename': 'file_id'})
    return df


def get_filepaths(data_dir: str, pattern: str = "*.csv") -> List[str]:
    """Get sorted list of CSV file paths."""
    path = Path(data_dir)
    files = sorted(path.glob(pattern))
    return [str(f) for f in files]


def load_and_merge_features(
    feature_csv: str,
    labels_csv: str
) -> pd.DataFrame:
    """Load features and merge with labels."""
    features_df = pd.read_csv(feature_csv)
    labels_df = pd.read_csv(labels_csv)
    merged = features_df.merge(labels_df, on='file_id', how='left')
    return merged


def save_features(df: pd.DataFrame, output_path: str):
    """Save features DataFrame to CSV."""
    df.to_csv(output_path, index=False)
    print(f"Saved features to {output_path} ({df.shape[0]} rows, {df.shape[1]} cols)")


def load_features(feature_csv: str) -> pd.DataFrame:
    """Load features from CSV."""
    return pd.read_csv(feature_csv)


def prepare_submission(predictions: np.ndarray, test_filepaths: List[str], output_path: str):
    """Create submission CSV in required format."""
    file_ids = [Path(fp).name for fp in test_filepaths]
    sub_df = pd.DataFrame({
        'file_id': file_ids,
        'prediction': predictions
    })
    sub_df.to_csv(output_path, index=False)
    print(f"Saved submission to {output_path}")
    return sub_df


def load_model(model_path: str):
    """Load trained model from disk."""
    return joblib.load(model_path)


def save_model(model, model_path: str):
    """Save trained model to disk."""
    joblib.dump(model, model_path)
    print(f"Saved model to {model_path}")