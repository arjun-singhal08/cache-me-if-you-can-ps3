import os
import sys
import zipfile
import pandas as pd
import numpy as np

def validate_shm(pred_file="predictions/shm_predictions.csv", test_dir=r"C:\Users\Arjun Singhal\NebulaX-Hackathon-ProblemStatement\PS3\02_Datasets\SHM\Test"):
    print("=" * 60)
    print("VALIDATING SHM PREDICTION FILE")
    print("=" * 60)
    
    if not os.path.exists(pred_file):
        print(f"[FAIL] Prediction file not found: {pred_file}")
        return False
        
    try:
        df = pd.read_csv(pred_file)
    except Exception as e:
        print(f"[FAIL] Could not parse CSV: {e}")
        return False
        
    # Check column names
    expected_cols = ['file_id', 'prediction']
    if list(df.columns) != expected_cols:
        print(f"[FAIL] Invalid columns. Expected {expected_cols}, got {list(df.columns)}")
        return False
    print("[PASS] Header check passed: ['file_id', 'prediction']")
    
    # Check expected test files
    if os.path.exists(test_dir):
        expected_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.csv')])
        pred_files = sorted(df['file_id'].tolist())
        if pred_files != expected_files:
            print(f"[FAIL] Mismatch in file_id rows. Expected {len(expected_files)} files, got {len(pred_files)}.")
            missing = set(expected_files) - set(pred_files)
            extra = set(pred_files) - set(expected_files)
            if missing: print(f"  Missing: {missing}")
            if extra: print(f"  Extra: {extra}")
            return False
        print(f"[PASS] Row count and file_id check passed: All {len(expected_files)} test files present.")
    else:
        if len(df) != 16:
            print(f"[FAIL] Expected 16 rows, got {len(df)}")
            return False
            
    # Check numeric predictions
    if df['prediction'].isnull().any():
        print("[FAIL] Found null or NaN values in 'prediction' column.")
        return False
        
    if not np.issubdtype(df['prediction'].dtype, np.number):
        print(f"[FAIL] Predictions are not numeric: dtype is {df['prediction'].dtype}")
        return False
        
    if (df['prediction'] <= 0).any():
        print("[FAIL] Found non-positive damage prediction values (fatigue damage must be > 0).")
        return False
        
    print("[PASS] Prediction values check passed: All values are valid positive numbers.")
    print(f"\n[INFO] Preview of predictions:\n{df.head()}")
    print("\n[SUCCESS] SHM PREDICTION SUBMISSION IS 100% VALID!")
    return True

def package_and_validate_zip():
    print("\n" + "=" * 60)
    print("PACKAGING & VALIDATING PREDICTIONS.ZIP")
    print("=" * 60)
    
    zip_name = "predictions.zip"
    pred_dir = "predictions"
    
    files_to_zip = [f for f in os.listdir(pred_dir) if f.endswith('_predictions.csv')]
    if not files_to_zip:
        print(f"[FAIL] No prediction CSVs found in {pred_dir}")
        return False
        
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in files_to_zip:
            file_path = os.path.join(pred_dir, f)
            zf.write(file_path, arcname=f)
            print(f"  Added {f} to {zip_name}")
            
    # Test opening and verifying zip contents
    with zipfile.ZipFile(zip_name, 'r') as zf:
        namelist = zf.namelist()
        print(f"[PASS] Successfully opened {zip_name}. Contents: {namelist}")
        for name in namelist:
            if "/" in name or "\\" in name:
                print(f"[FAIL] File {name} is in a subfolder inside zip. Must be at top level!")
                return False
                
    print(f"[SUCCESS] {zip_name} PACKAGED AND TESTED SUCCESSFULLY!")
    return True

if __name__ == '__main__':
    shm_ok = validate_shm()
    zip_ok = package_and_validate_zip()
    
    if shm_ok and zip_ok:
        print("\n[RESULT] ALL VALIDATION CHECKS PASSED PERFECTLY!")
        sys.exit(0)
    else:
        print("\n[RESULT] VALIDATION CHECKS FAILED.")
        sys.exit(1)
