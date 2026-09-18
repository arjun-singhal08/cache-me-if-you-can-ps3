import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def clean_series(s: pd.Series) -> np.ndarray:
    """Clean a series: handle missing, inf, convert to numeric."""
    s = pd.to_numeric(s, errors='coerce')
    s = s.replace([np.inf, -np.inf], np.nan)
    return s.values


def safe_stat(func, arr: np.ndarray, default: float = 0.0) -> float:
    """Safely compute statistic with default for empty/all-NaN arrays."""
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return default
    try:
        return float(func(arr))
    except:
        return default


def compute_statistical_features(arr: np.ndarray) -> Dict[str, float]:
    """Compute core statistical features for a 1D array (fast)."""
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return {f'stat_{k}': 0.0 for k in 
                ['mean', 'std', 'min', 'max', 'median', 'q25', 'q75', 'iqr', 
                 'cv', 'range', 'mad']}
    
    mean_val = np.mean(arr)
    std_val = np.std(arr)
    median_val = np.median(arr)
    q25 = np.percentile(arr, 25)
    q75 = np.percentile(arr, 75)
    
    features = {
        'stat_mean': mean_val,
        'stat_std': std_val,
        'stat_min': np.min(arr),
        'stat_max': np.max(arr),
        'stat_median': median_val,
        'stat_q25': q25,
        'stat_q75': q75,
        'stat_iqr': q75 - q25,
        'stat_cv': std_val / (mean_val + 1e-8),
        'stat_range': np.max(arr) - np.min(arr),
        'stat_mad': np.median(np.abs(arr - median_val))
    }
    
    return features


def compute_trend_features(arr: np.ndarray) -> Dict[str, float]:
    """Compute simple linear trend (fast)."""
    arr = arr[~np.isnan(arr)]
    if len(arr) < 3:
        return {'trend_slope': 0.0}
    
    x = np.arange(len(arr))
    slope, _, _, _, _ = stats.linregress(x, arr)
    return {'trend_slope': slope}


def compute_categorical_features(s: pd.Series) -> Dict[str, float]:
    """Compute features for categorical/mode columns (fast)."""
    s_clean = s.dropna()
    if len(s_clean) == 0:
        return {'cat_nunique': 0, 'cat_mode_freq': 0.0, 'cat_entropy': 0.0,
                'cat_transitions': 0.0}
    
    features = {}
    features['cat_nunique'] = float(s_clean.nunique())
    
    mode_val = s_clean.mode()
    if len(mode_val) > 0:
        features['cat_mode_freq'] = float((s_clean == mode_val[0]).mean())
    else:
        features['cat_mode_freq'] = 0.0
    
    vc = s_clean.value_counts(normalize=True)
    features['cat_entropy'] = float(-(vc * np.log(vc + 1e-10)).sum())
    
    features['cat_transitions'] = float((s_clean != s_clean.shift()).sum() / len(s_clean))
    
    return features


def extract_car_features(car_df: pd.DataFrame, car_id: str, 
                         all_car_dfs: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    """Extract all features for a single car, including cross-car features (optimized)."""
    features = {}
    features['car_id'] = car_id
    
    # Get numeric and categorical columns
    numeric_cols = car_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = car_df.select_dtypes(include=['object']).columns.tolist()
    
    # Remove ID columns
    id_cols = ['Car model', 'Train number', 'Time']
    numeric_cols = [c for c in numeric_cols if c not in id_cols]
    cat_cols = [c for c in cat_cols if c not in id_cols]
    
    # ===== Per-parameter statistical features =====
    for col in numeric_cols:
        arr = clean_series(car_df[col])
        prefix = col.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
        
        # Statistical (fast)
        stat_feats = compute_statistical_features(arr)
        for k, v in stat_feats.items():
            features[f'{prefix}_{k}'] = v
            
        # Trend (fast)
        trend_feats = compute_trend_features(arr)
        for k, v in trend_feats.items():
            features[f'{prefix}_{k}'] = v
    
    # Categorical features
    for col in cat_cols:
        cat_feats = compute_categorical_features(car_df[col])
        prefix = col.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
        for k, v in cat_feats.items():
            features[f'{prefix}_{k}'] = v
    
    # ===== Physics-informed features (if pressure data available) =====
    features.update(extract_physics_features(car_df))
    
    # ===== Cross-car differential features =====
    features.update(extract_cross_car_features(car_df, car_id, all_car_dfs, numeric_cols))
    
    # ===== Operating phase features =====
    features.update(extract_phase_features(car_df))
    
    return features


def extract_physics_features(car_df: pd.DataFrame) -> Dict[str, float]:
    """Extract physics-informed features for refrigerant leak detection."""
    features = {}
    
    # Try to find pressure columns (may have different names)
    hp_cols = [c for c in car_df.columns if 'High Pressure' in c or 'high pressure' in c.lower()]
    lp_cols = [c for c in car_df.columns if 'Low Pressure' in c or 'low pressure' in c.lower()]
    indoor_cols = [c for c in car_df.columns if 'Indoor' in c or 'Cabin' in c or 'Passenger' in c]
    outdoor_cols = [c for c in car_df.columns if 'Outdoor' in c or 'Fresh Air' in c]
    cool_set_cols = [c for c in car_df.columns if 'Control Temperature (Cooling)' in c]
    heat_set_cols = [c for c in car_df.columns if 'Control Temperature (Heating)' in c]
    comp_cols = [c for c in car_df.columns if 'Compressor' in c and 'Running' in c]
    
    # Pressure ratio features (if available) - key for leak detection
    if hp_cols and lp_cols:
        hp = clean_series(car_df[hp_cols[0]])
        lp = clean_series(car_df[lp_cols[0]])
        valid = ~np.isnan(hp) & ~np.isnan(lp) & (lp > 0)
        if np.sum(valid) > 10:
            pr = hp[valid] / lp[valid]
            features['physics_pressure_ratio_mean'] = np.mean(pr)
            features['physics_pressure_ratio_std'] = np.std(pr)
            features['physics_pressure_ratio_trend'] = compute_trend_features(pr)['trend_slope']
            features['physics_high_pressure_mean'] = np.mean(hp[valid])
            features['physics_low_pressure_mean'] = np.mean(lp[valid])
            features['physics_high_pressure_std'] = np.std(hp[valid])
            features['physics_low_pressure_std'] = np.std(lp[valid])
    
    # Delta-T (cooling capacity proxy)
    if indoor_cols and outdoor_cols:
        indoor = clean_series(car_df[indoor_cols[0]])
        outdoor = clean_series(car_df[outdoor_cols[0]])
        valid = ~np.isnan(indoor) & ~np.isnan(outdoor)
        if np.sum(valid) > 10:
            dt = indoor[valid] - outdoor[valid]
            features['physics_delta_t_mean'] = np.mean(dt)
            features['physics_delta_t_std'] = np.std(dt)
            features['physics_delta_t_min'] = np.min(dt)
            features['physics_delta_t_trend'] = compute_trend_features(dt)['trend_slope']
    
    # Control deviation (indoor vs setpoint)
    if indoor_cols and cool_set_cols:
        indoor = clean_series(car_df[indoor_cols[0]])
        cool_set = clean_series(car_df[cool_set_cols[0]])
        valid = ~np.isnan(indoor) & ~np.isnan(cool_set)
        if np.sum(valid) > 10:
            dev = indoor[valid] - cool_set[valid]
            features['physics_cool_dev_mean'] = np.mean(dev)
            features['physics_cool_dev_std'] = np.std(dev)
            features['physics_cool_dev_max'] = np.max(dev)
    
    # Compressor runtime fraction
    if comp_cols:
        comp = clean_series(car_df[comp_cols[0]])
        valid = ~np.isnan(comp)
        if np.sum(valid) > 0:
            features['physics_comp_runtime_frac'] = float(np.mean(comp[valid] > 0.5))
    
    return features


def extract_cross_car_features(car_df: pd.DataFrame, car_id: str,
                                all_car_dfs: Dict[str, pd.DataFrame],
                                numeric_cols: List[str]) -> Dict[str, float]:
    """Extract cross-car differential features (key for localisation)."""
    features = {}
    other_cars = [cid for cid in all_car_dfs.keys() if cid != car_id]
    
    if not other_cars:
        return features
    
    # Find columns that exist in ALL cars
    common_cols = set(numeric_cols)
    for oc in other_cars:
        oc_cols = set(all_car_dfs[oc].select_dtypes(include=[np.number]).columns.tolist())
        oc_cols = [c for c in oc_cols if c not in ['Car model', 'Train number', 'Time']]
        common_cols &= set(oc_cols)
    
    common_cols = list(common_cols)
    
    # For each numeric parameter, compute car's z-score and rank within fleet
    zscores = []
    extremeness_vals = []
    
    for col in common_cols:
        car_vals = clean_series(car_df[col])
        car_mean = np.nanmean(car_vals)
        
        # Collect fleet values
        fleet_means = []
        fleet_stds = []
        for oc in other_cars:
            oc_vals = clean_series(all_car_dfs[oc][col])
            fleet_means.append(np.nanmean(oc_vals))
            fleet_stds.append(np.nanstd(oc_vals) + 1e-8)
        
        if fleet_means:
            fleet_mean = np.mean(fleet_means)
            fleet_std = np.mean(fleet_stds)
            
            # Z-score relative to fleet
            z_score = (car_mean - fleet_mean) / (fleet_std + 1e-8)
            prefix = col.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
            features[f'{prefix}_fleet_zscore'] = z_score
            zscores.append(z_score)
            
            # Rank (how many cars are higher/lower)
            all_means = [car_mean] + fleet_means
            rank_high = sum(1 for m in all_means if m > car_mean)
            rank_low = sum(1 for m in all_means if m < car_mean)
            features[f'{prefix}_fleet_rank_high'] = float(rank_high)
            features[f'{prefix}_fleet_rank_low'] = float(rank_low)
            ext = max(rank_high, rank_low) / len(all_means)
            features[f'{prefix}_fleet_extremeness'] = ext
            extremeness_vals.append(ext)
    
    # Aggregate cross-car features
    if zscores:
        features['crosscar_max_abs_zscore'] = np.max(np.abs(zscores))
        features['crosscar_mean_abs_zscore'] = np.mean(np.abs(zscores))
        features['crosscar_zscore_std'] = np.std(zscores)
    
    if extremeness_vals:
        features['crosscar_max_extremeness'] = np.max(extremeness_vals)
        features['crosscar_mean_extremeness'] = np.mean(extremeness_vals)
    
    return features


def extract_phase_features(car_df: pd.DataFrame) -> Dict[str, float]:
    """Extract features per operating phase (cooling/heating/off)."""
    features = {}
    
    # Find mode columns
    mode_cols = [c for c in car_df.columns if 'Running Mode' in c or 'Setting Mode' in c]
    temp_cols = [c for c in car_df.columns if 'Temperature' in c and 'Control' not in c]
    
    if not mode_cols or not temp_cols:
        return features
    
    mode_col = mode_cols[0]
    temp_col = temp_cols[0]
    
    modes = car_df[mode_col].fillna('UNKNOWN')
    temps = clean_series(car_df[temp_col])
    
    unique_modes = modes.unique()
    for mode in unique_modes:
        mask = (modes == mode)
        mode_temps = temps[mask.values]
        mode_temps = mode_temps[~np.isnan(mode_temps)]
        
        if len(mode_temps) > 5:
            prefix = f'phase_{str(mode).replace(" ", "_").replace("/", "_")}'
            features[f'{prefix}_temp_mean'] = np.mean(mode_temps)
            features[f'{prefix}_temp_std'] = np.std(mode_temps)
            features[f'{prefix}_duration_frac'] = float(mask.mean())
    
    return features


def build_feature_matrix(train_cases: List[Tuple], 
                         test_df: Optional[pd.DataFrame] = None) -> Tuple:
    """Build feature matrix for all train cases and optionally test.
    
    Returns:
        X_train: (n_cases * 8, n_features) feature matrix
        y_train: (n_cases * 8,) binary labels (1 for faulty car)
        groups_train: (n_cases * 8,) case indices for LOCO-CV
        car_ids_train: (n_cases * 8,) car identifiers
        feature_names: list of feature names
        X_test, test_car_ids (if test_df provided)
    """
    all_features = []
    all_labels = []
    all_groups = []
    all_car_ids = []
    
    for case_idx, (case_id, df, faulty_car) in enumerate(train_cases):
        # Parse car dataframes directly
        car_dfs = {}
        car_params = {}
        for col in df.columns:
            if col.startswith('Car ') and ' - ' in col:
                cid, param = col.split(' - ', 1)
                cid = cid.replace('Car ', '')
                if cid not in car_params:
                    car_params[cid] = []
                car_params[cid].append(param)
        
        id_cols = ['Car model', 'Train number', 'Time']
        for car_id, params in car_params.items():
            car_cols = id_cols + [f'Car {car_id} - {p}' for p in params]
            car_df = df[car_cols].copy()
            car_df.columns = id_cols + params
            car_dfs[car_id] = car_df
        
        for car_id in sorted(car_dfs.keys()):
            car_df = car_dfs[car_id]
            feats = extract_car_features(car_df, car_id, car_dfs)
            
            all_features.append(feats)
            # Compare with zero-padded faulty car ID
            faulty_car_str = str(faulty_car).zfill(2)
            all_labels.append(1 if car_id == faulty_car_str else 0)
            all_groups.append(case_idx)
            all_car_ids.append(f"{case_id}_{car_id}")
    
    # Convert to DataFrame
    feat_df = pd.DataFrame(all_features)
    
    # Ensure consistent columns
    feat_df = feat_df.fillna(0)
    
    # Replace inf
    feat_df = feat_df.replace([np.inf, -np.inf], 0)
    
    feature_names = [c for c in feat_df.columns if c != 'car_id']
    X = feat_df[feature_names].values
    y = np.array(all_labels)
    groups = np.array(all_groups)
    car_ids = np.array(all_car_ids)
    
    # Test features
    X_test = None
    test_car_ids = None
    if test_df is not None:
        # Parse test car dataframes
        test_car_dfs = {}
        test_car_params = {}
        for col in test_df.columns:
            if col.startswith('Car ') and ' - ' in col:
                cid, param = col.split(' - ', 1)
                cid = cid.replace('Car ', '')
                if cid not in test_car_params:
                    test_car_params[cid] = []
                test_car_params[cid].append(param)
        
        id_cols = ['Car model', 'Train number', 'Time']
        for car_id, params in test_car_params.items():
            car_cols = id_cols + [f'Car {car_id} - {p}' for p in params]
            car_df = test_df[car_cols].copy()
            car_df.columns = id_cols + params
            test_car_dfs[car_id] = car_df
        
        test_features = []
        test_car_ids_list = []
        
        for car_id in sorted(test_car_dfs.keys()):
            car_df = test_car_dfs[car_id]
            feats = extract_car_features(car_df, car_id, test_car_dfs)
            test_features.append(feats)
            test_car_ids_list.append(car_id)
        
        test_feat_df = pd.DataFrame(test_features)
        # Align columns with training
        for col in feature_names:
            if col not in test_feat_df.columns:
                test_feat_df[col] = 0
        test_feat_df = test_feat_df[feature_names].fillna(0).replace([np.inf, -np.inf], 0)
        
        X_test = test_feat_df.values
        test_car_ids = np.array(test_car_ids_list)
    
    return X, y, groups, car_ids, feature_names, X_test, test_car_ids


if __name__ == "__main__":
    from data_loader import load_data
    import time
    
    print("Loading data...")
    data = load_data()
    
    print("Building feature matrix...")
    start = time.time()
    X, y, groups, car_ids, feature_names, X_test, test_car_ids = build_feature_matrix(
        data['train_cases'], data['test_df']
    )
    elapsed = time.time() - start
    
    print(f"Feature extraction: {elapsed:.2f}s")
    print(f"Train: X={X.shape}, y={y.shape}, groups={groups.shape}")
    print(f"Features: {len(feature_names)}")
    print(f"Test: X_test={X_test.shape if X_test is not None else None}")
    print(f"Faulty cars in train: {np.sum(y)} / {len(y)}")
    print(f"Feature sample: {feature_names[:20]}")