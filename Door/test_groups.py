import sys
sys.path.insert(0, r'C:\Users\Arjun Singhal\NebulaX-Hackathon-ProblemStatement\PS3\02_Datasets\Door\app')
import numpy as np
import pandas as pd
from utils.io import load_train_data
from features.extract import extract_segment_features

train, segments = load_train_data(
    r'C:\Users\Arjun Singhal\NebulaX-Hackathon-ProblemStatement\PS3\02_Datasets\Door\Train.csv',
    r'C:\Users\Arjun Singhal\NebulaX-Hackathon-ProblemStatement\PS3\02_Datasets\Door\Train_Segments_Answer.csv'
)

seg_ids = segments['segment_id'].tolist()
groups = np.array([f"{segments.iloc[i]['operation']}_{segments.iloc[i]['status']}" for i in range(len(seg_ids))])
print('Groups:', np.unique(groups))
print('Counts:', pd.Series(groups).value_counts())

from sklearn.model_selection import GroupKFold
cv = GroupKFold(n_splits=5)
try:
    for fold, (train_idx, val_idx) in enumerate(cv.split(np.zeros(len(groups)), np.zeros(len(groups)), groups)):
        print(f'Fold {fold}: train={len(train_idx)}, val={len(val_idx)}, train groups={np.unique(groups[train_idx])}, val groups={np.unique(groups[val_idx])}')
except Exception as e:
    print('Error:', e)