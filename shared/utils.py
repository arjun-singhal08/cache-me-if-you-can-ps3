"""
Shared utilities for NebulaX PS3 subsystems.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any


def load_yaml_config(path: Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    import yaml
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def save_predictions_csv(df: pd.DataFrame, path: Path):
    """Save predictions CSV with proper formatting."""
    df.to_csv(path, index=False)
    print(f"Predictions saved to {path}")


def validate_acv_submission(df: pd.DataFrame) -> bool:
    """Validate ACV submission format."""
    required_cols = {'file_id', 'ranked_cars'}
    if not required_cols.issubset(set(df.columns)):
        print(f"Missing required columns. Need: {required_cols}")
        return False
    
    for _, row in df.iterrows():
        cars = row['ranked_cars'].split('|')
        if len(cars) != 8:
            print(f"Expected 8 cars, got {len(cars)} for {row['file_id']}")
            return False
        if len(set(cars)) != 8:
            print(f"Duplicate car IDs in {row['file_id']}")
            return False
    
    print("ACV submission format valid!")
    return True


def validate_door_submission(df: pd.DataFrame) -> bool:
    """Validate Door submission format."""
    required_cols = {'start_time', 'end_time', 'prediction'}
    if not required_cols.issubset(set(df.columns)):
        print(f"Missing required columns. Need: {required_cols}")
        return False
    
    valid_labels = {'Normal', 'Abnormal resistance'}
    invalid = set(df['prediction'].unique()) - valid_labels
    if invalid:
        print(f"Invalid prediction labels: {invalid}")
        return False
    
    print("Door submission format valid!")
    return True


def create_submission_zip(submission_dir: Path, output_zip: Path):
    """Create predictions.zip for competition submission."""
    import zipfile
    
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        for pred_file in submission_dir.glob("*_predictions.csv"):
            zf.write(pred_file, pred_file.name)
    
    print(f"Submission zip created: {output_zip}")
    print(f"Contents: {[f.name for f in submission_dir.glob('*_predictions.csv')]}")