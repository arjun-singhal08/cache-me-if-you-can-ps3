import os
import glob
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import numpy as np
import pandas as pd

def find_shm_data_dir() -> Path:
    """
    Robustly locate the SHM dataset directory across different machines,
    repo structures, and operating systems using pathlib.Path.
    """
    # 1. Check environment variable override
    if "SHM_DATA_DIR" in os.environ:
        p = Path(os.environ["SHM_DATA_DIR"]).resolve()
        if p.exists():
            return p

    # 2. Check candidate relative search paths
    base_dir = Path(__file__).resolve().parents[2]  # repo root
    candidates = [
        base_dir / "data" / "SHM",
        base_dir / "PS3" / "02_Datasets" / "SHM",
        base_dir.parent / "NebulaX-Hackathon-ProblemStatement" / "PS3" / "02_Datasets" / "SHM",
        base_dir.parent / "PS3" / "02_Datasets" / "SHM",
        Path.home() / "NebulaX-Hackathon-ProblemStatement" / "PS3" / "02_Datasets" / "SHM",
        Path.cwd() / "data" / "SHM",
        Path.cwd() / "PS3" / "02_Datasets" / "SHM",
    ]

    for candidate in candidates:
        if candidate.exists() and (candidate / "Train_Labels.csv").exists():
            return candidate.resolve()

    # Fallback to base_dir / 'data' / 'SHM' if not found yet
    return (base_dir / "data" / "SHM").resolve()


def load_single_signal(file_path: Path) -> np.ndarray:
    """
    Load a single 1D dynamic stress time-series CSV file into a float64 numpy array.
    Validates non-emptiness, finiteness, and 1D dimensionality.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Signal file not found: {file_path}")

    df = pd.read_csv(file_path, header=None)
    signal = df.iloc[:, 0].values.astype(np.float64)

    if len(signal) == 0:
        raise ValueError(f"Signal file is empty: {file_path}")
    if np.isnan(signal).any() or np.isinf(signal).any():
        raise ValueError(f"Signal contains NaN or Inf values: {file_path}")

    return signal


def load_training_dataset(data_dir: Optional[Path] = None) -> Tuple[List[Path], pd.DataFrame]:
    """
    Discovers all train CSV files (train01.csv - train64.csv) and reads Train_Labels.csv.

    Returns:
        train_file_paths: List[Path]
        df_labels: pd.DataFrame with columns ['filename', 'damage']
    """
    if data_dir is None:
        data_dir = find_shm_data_dir()

    train_dir = data_dir / "Train"
    labels_file = data_dir / "Train_Labels.csv"

    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found at: {train_dir}")
    if not labels_file.exists():
        raise FileNotFoundError(f"Train_Labels.csv not found at: {labels_file}")

    train_files = sorted(list(train_dir.glob("*.csv")))
    df_labels = pd.read_csv(labels_file)

    # Standardize column naming if needed
    if "file_id" in df_labels.columns and "filename" not in df_labels.columns:
        df_labels = df_labels.rename(columns={"file_id": "filename"})

    return train_files, df_labels


def load_test_dataset(data_dir: Optional[Path] = None) -> List[Path]:
    """
    Discovers all held-out test CSV files (test01.csv - test16.csv).

    Returns:
        test_file_paths: List[Path]
    """
    if data_dir is None:
        data_dir = find_shm_data_dir()

    test_dir = data_dir / "Test"
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found at: {test_dir}")

    test_files = sorted(list(test_dir.glob("*.csv")))
    return test_files
