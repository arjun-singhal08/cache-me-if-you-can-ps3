import os
import glob
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks
import rainflow
from tqdm import tqdm
from joblib import Parallel, delayed

def extract_features_from_signal(signal: np.ndarray) -> dict:
    """
    Extract physics-informed, statistical, spectral, and Rainflow fatigue features
    from a 1D dynamic stress time series array.
    """
    feats = {}
    
    # Ensure 1D float array
    signal = signal.astype(np.float64)
    n_samples = len(signal)
    feats['n_samples'] = n_samples
    
    # 1. Statistical / Time-Domain Features
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    min_val = np.min(signal)
    max_val = np.max(signal)
    ptp_val = max_val - min_val
    rms_val = np.sqrt(np.mean(signal ** 2))
    
    feats['mean'] = mean_val
    feats['std'] = std_val
    feats['min'] = min_val
    feats['max'] = max_val
    feats['ptp'] = ptp_val
    feats['rms'] = rms_val
    feats['crest_factor'] = (max_val / rms_val) if rms_val > 1e-8 else 0.0
    feats['peak_factor'] = (max(abs(min_val), abs(max_val)) / rms_val) if rms_val > 1e-8 else 0.0
    
    feats['skewness'] = float(stats.skew(signal))
    feats['kurtosis'] = float(stats.kurtosis(signal))
    feats['mad'] = np.mean(np.abs(signal - mean_val))
    
    # Quantiles
    q1, q5, q25, q50, q75, q95, q99 = np.percentile(signal, [1, 5, 25, 50, 75, 95, 99])
    feats['q01'] = q1
    feats['q05'] = q5
    feats['q25'] = q25
    feats['q50'] = q50
    feats['q75'] = q75
    feats['q95'] = q95
    feats['q99'] = q99
    feats['iqr'] = q75 - q25
    
    # Energy
    feats['energy'] = np.sum(signal ** 2)
    
    # 2. Rainflow Cycle Counting & Fatigue Features (Miner's linear damage rule proxies)
    try:
        # Fast prominence-filtered extrema for Rainflow cycle counting
        prominence = 0.01 * ptp_val if ptp_val > 1e-6 else 1e-4
        peaks, _ = find_peaks(signal, prominence=prominence)
        valleys, _ = find_peaks(-signal, prominence=prominence)
        
        extrema_idx = np.sort(np.concatenate(([0], peaks, valleys, [len(signal)-1])))
        extrema_sig = signal[extrema_idx]
        
        cycles = list(rainflow.count_cycles(extrema_sig))
        if cycles:
            ranges = np.array([c[0] for c in cycles])
            counts = np.array([c[1] for c in cycles])
            
            feats['rf_count'] = len(cycles)
            feats['rf_total_cycles'] = float(np.sum(counts))
            feats['rf_max_range'] = float(np.max(ranges)) if len(ranges) > 0 else 0.0
            feats['rf_mean_range'] = float(np.average(ranges, weights=counts)) if len(ranges) > 0 else 0.0
            feats['rf_std_range'] = float(np.std(ranges)) if len(ranges) > 0 else 0.0
            
            # S-N curve power accumulators: sum(n_i * (delta_sigma)^m) for m in [3, 3.5, 4, 5]
            feats['rf_damage_m3'] = float(np.sum(counts * (ranges ** 3)))
            feats['rf_damage_m3_5'] = float(np.sum(counts * (ranges ** 3.5)))
            feats['rf_damage_m4'] = float(np.sum(counts * (ranges ** 4)))
            feats['rf_damage_m5'] = float(np.sum(counts * (ranges ** 5)))
            
            # Stress range histogram (10 bins relative to max range)
            hist, _ = np.histogram(ranges, bins=10, weights=counts)
            for i, h in enumerate(hist):
                feats[f'rf_bin_{i}'] = float(h)
        else:
            feats['rf_count'] = 0
            feats['rf_total_cycles'] = 0.0
            feats['rf_max_range'] = 0.0
            feats['rf_mean_range'] = 0.0
            feats['rf_std_range'] = 0.0
            feats['rf_damage_m3'] = 0.0
            feats['rf_damage_m3_5'] = 0.0
            feats['rf_damage_m4'] = 0.0
            feats['rf_damage_m5'] = 0.0
            for i in range(10):
                feats[f'rf_bin_{i}'] = 0.0
    except Exception as e:
        print(f"Warning in rainflow extraction: {e}")
        feats['rf_count'] = 0
        feats['rf_total_cycles'] = 0.0
        feats['rf_max_range'] = 0.0
        feats['rf_mean_range'] = 0.0
        feats['rf_std_range'] = 0.0
        feats['rf_damage_m3'] = 0.0
        feats['rf_damage_m3_5'] = 0.0
        feats['rf_damage_m4'] = 0.0
        feats['rf_damage_m5'] = 0.0
        for i in range(10):
            feats[f'rf_bin_{i}'] = 0.0

    # 3. Frequency-Domain / Spectral Features (FFT)
    fft_vals = np.abs(np.fft.rfft(signal))
    feats['fft_mean'] = float(np.mean(fft_vals))
    feats['fft_std'] = float(np.std(fft_vals))
    feats['fft_max'] = float(np.max(fft_vals))
    feats['fft_skew'] = float(stats.skew(fft_vals))
    feats['fft_kurtosis'] = float(stats.kurtosis(fft_vals))
    
    # 5 Spectral Bands
    n_fft = len(fft_vals)
    band_len = max(1, n_fft // 5)
    for b in range(5):
        band_vals = fft_vals[b*band_len : (b+1)*band_len]
        feats[f'fft_band_{b}_energy'] = float(np.sum(band_vals ** 2)) if len(band_vals) > 0 else 0.0

    return feats

def _process_single_file(filepath: str) -> dict:
    filename = os.path.basename(filepath)
    df_sig = pd.read_csv(filepath, header=None)
    signal = df_sig.iloc[:, 0].values
    feats = extract_features_from_signal(signal)
    feats['filename'] = filename
    return feats

def process_directory(data_dir: str, n_jobs: int = -1) -> pd.DataFrame:
    """
    Process all .csv files in data_dir in parallel and return a DataFrame of features.
    """
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    print(f"Found {len(files)} CSV files in {data_dir}. Processing in parallel...")
    
    rows = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_process_single_file)(fp) for fp in tqdm(files, desc=f"Extracting {os.path.basename(data_dir)}")
    )
        
    res_df = pd.DataFrame(rows)
    cols = ['filename'] + [c for c in res_df.columns if c != 'filename']
    return res_df[cols]

if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parents[1]
    candidates_train = [
        base_dir / "PS3" / "02_Datasets" / "SHM" / "Train",
        base_dir / "data" / "SHM" / "Train",
        base_dir.parent / "NebulaX-Hackathon-ProblemStatement" / "PS3" / "02_Datasets" / "SHM" / "Train",
        Path.home() / "NebulaX-Hackathon-ProblemStatement" / "PS3" / "02_Datasets" / "SHM" / "Train",
    ]
    train_dir = next((str(p) for p in candidates_train if p.exists()), str(candidates_train[0]))

    candidates_test = [
        base_dir / "PS3" / "02_Datasets" / "SHM" / "Test",
        base_dir / "data" / "SHM" / "Test",
        base_dir.parent / "NebulaX-Hackathon-ProblemStatement" / "PS3" / "02_Datasets" / "SHM" / "Test",
        Path.home() / "NebulaX-Hackathon-ProblemStatement" / "PS3" / "02_Datasets" / "SHM" / "Test",
    ]
    test_dir = next((str(p) for p in candidates_test if p.exists()), str(candidates_test[0]))

    output_dir = "data_processed"
    os.makedirs(output_dir, exist_ok=True)
    
    if os.path.exists(train_dir):
        df_train_feats = process_directory(train_dir)
        df_train_feats.to_csv(os.path.join(output_dir, "train_features.csv"), index=False)
        print(f"Saved train features to {output_dir}/train_features.csv with shape {df_train_feats.shape}")
        
    if os.path.exists(test_dir):
        df_test_feats = process_directory(test_dir)
        df_test_feats.to_csv(os.path.join(output_dir, "test_features.csv"), index=False)
        print(f"Saved test features to {output_dir}/test_features.csv with shape {df_test_feats.shape}")

