import os
import re
from pathlib import Path
from typing import Tuple, List, Optional, Union
import numpy as np
import pandas as pd

def natural_sort_key(p: Union[str, Path]):
    name = Path(p).name
    match = re.search(r'\d+', name)
    return int(match.group()) if match else name

def find_rail_data_dir() -> Path:
    """
    Robustly locate the Rail Corrugation dataset directory across different machines,
    repo structures, and operating systems using pathlib.Path.
    """
    # 1. Environment variable override
    if "RAIL_DATA_DIR" in os.environ:
        p = Path(os.environ["RAIL_DATA_DIR"]).resolve()
        if p.exists():
            return p

    # 2. Candidate search locations
    base_dir = Path(__file__).resolve().parents[2]  # repo root
    candidates = [
        base_dir / "data" / "Rail_Corrugation",
        base_dir / "PS3" / "02_Datasets" / "Rail_Corrugation",
        base_dir.parent / "NebulaX-Hackathon-ProblemStatement" / "PS3" / "02_Datasets" / "Rail_Corrugation",
        base_dir.parent / "PS3" / "02_Datasets" / "Rail_Corrugation",
        Path.home() / "NebulaX-Hackathon-ProblemStatement" / "PS3" / "02_Datasets" / "Rail_Corrugation",
        Path.cwd() / "data" / "Rail_Corrugation",
        Path.cwd() / "PS3" / "02_Datasets" / "Rail_Corrugation",
    ]

    for candidate in candidates:
        if candidate.exists() and (candidate / "Train_Labels.csv").exists():
            return candidate.resolve()

    return (base_dir / "data" / "Rail_Corrugation").resolve()


def validate_vibration_data(df: pd.DataFrame, file_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Validate multi-channel vibration recording structure.
    Checks:
    - Empty file / 0 rows
    - Minimum channel count (at least 129 columns: 1 speed + 128 vibration/shock channels)
    - NaN and Inf sanitization
    - Zero variance edge conditions
    """
    if df.empty or len(df) == 0:
        raise ValueError(f"File is empty: {file_path}")

    if df.shape[1] < 129:
        raise ValueError(
            f"Invalid channel count for rail vibration recording in {file_path}. "
            f"Expected 129 channels, found {df.shape[1]}"
        )

    # Check for NaN / Infs and sanitize
    if df.isna().any().any() or np.isinf(df.values).any():
        df = df.fillna(0.0)
        df = df.replace([np.inf, -np.inf], 0.0)

    return df


def load_single_file(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Loads and validates a single rail corrugation multi-channel CSV file.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Rail vibration file not found: {file_path}")

    df = pd.read_csv(file_path)
    return validate_vibration_data(df, file_path)


def load_training_dataset(data_dir: Optional[Path] = None) -> Tuple[List[Path], pd.DataFrame]:
    """
    Discovers all train CSV files (Train1.csv - Train272.csv) and reads Train_Labels.csv.

    Returns:
        train_file_paths: List[Path] sorted naturally
        df_labels: pd.DataFrame with columns ['filename', 'label']
    """
    if data_dir is None:
        data_dir = find_rail_data_dir()

    train_dir = data_dir / "Train"
    labels_file = data_dir / "Train_Labels.csv"

    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found at: {train_dir}")
    if not labels_file.exists():
        raise FileNotFoundError(f"Train_Labels.csv not found at: {labels_file}")

    train_files = sorted(list(train_dir.glob("*.csv")), key=natural_sort_key)
    df_labels = pd.read_csv(labels_file)

    # Standardize column naming if needed
    if "file_id" in df_labels.columns and "filename" not in df_labels.columns:
        df_labels = df_labels.rename(columns={"file_id": "filename"})
    if "prediction" in df_labels.columns and "label" not in df_labels.columns:
        df_labels = df_labels.rename(columns={"prediction": "label"})

    return train_files, df_labels


def load_test_dataset(data_dir: Optional[Path] = None) -> List[Path]:
    """
    Discovers all held-out test CSV files (Test1.csv - Test68.csv).

    Returns:
        test_file_paths: List[Path] sorted naturally
    """
    if data_dir is None:
        data_dir = find_rail_data_dir()

    test_dir = data_dir / "Test"
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found at: {test_dir}")

    test_files = sorted(list(test_dir.glob("*.csv")), key=natural_sort_key)
    return test_files
