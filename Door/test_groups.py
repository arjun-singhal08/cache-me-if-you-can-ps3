import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add Door directory to path for imports
door_dir = Path(__file__).resolve().parent
if str(door_dir) not in sys.path:
    sys.path.insert(0, str(door_dir))

try:
    import pytest
except ImportError:
    pytest = None

# Candidate paths for Door dataset
candidate_dirs = [
    door_dir.parent / "PS3" / "02_Datasets" / "Door",
    door_dir.parents[1] / "NebulaX-Hackathon-ProblemStatement" / "PS3" / "02_Datasets" / "Door",
    door_dir.parent / "data" / "Door",
    Path.home() / "NebulaX-Hackathon-ProblemStatement" / "PS3" / "02_Datasets" / "Door",
]

data_dir = next((p for p in candidate_dirs if (p / "Train.csv").exists() and (p / "Train_Segments_Answer.csv").exists()), None)


def test_door_groups():
    """Verify group splitting logic, gracefully skipping if raw dataset is absent."""
    if data_dir is None:
        if pytest is not None:
            pytest.skip("Raw Door dataset not present; skipping dataset-loading assertions.")
        else:
            print("Raw Door dataset not present; skipping dataset-loading assertions.")
            return

    from utils.io import load_train_data
    from features.extract import extract_segment_features

    train, segments = load_train_data(
        str(data_dir / "Train.csv"),
        str(data_dir / "Train_Segments_Answer.csv")
    )

    seg_ids = segments['segment_id'].tolist()
    groups = np.array([f"{segments.iloc[i]['operation']}_{segments.iloc[i]['status']}" for i in range(len(seg_ids))])
    print('Groups:', np.unique(groups))
    print('Counts:', pd.Series(groups).value_counts())

    from sklearn.model_selection import GroupKFold
    cv = GroupKFold(n_splits=5)
    for fold, (train_idx, val_idx) in enumerate(cv.split(np.zeros(len(groups)), np.zeros(len(groups)), groups)):
        print(f'Fold {fold}: train={len(train_idx)}, val={len(val_idx)}, train groups={np.unique(groups[train_idx])}, val groups={np.unique(groups[val_idx])}')


if __name__ == '__main__':
    test_door_groups()