import os
import sys
import zipfile
from pathlib import Path
import pandas as pd
import numpy as np

def validate_shm(pred_file="predictions/shm_predictions.csv", test_dir=None):
    print("=" * 65)
    print("STRICT SUBMISSION VALIDATION — SHM SUBSYSTEM")
    print("=" * 65)
    
    pred_path = Path(pred_file)
    if not pred_path.exists():
        print(f"[FAIL] Prediction file not found: {pred_path}")
        return False
        
    try:
        df = pd.read_csv(pred_path)
    except Exception as e:
        print(f"[FAIL] Could not parse CSV file: {e}")
        return False

    # 1. Exact Column Names & Ordering
    expected_cols = ['file_id', 'prediction']
    if list(df.columns) != expected_cols:
        print(f"[FAIL] Column mismatch. Expected exactly {expected_cols}, got {list(df.columns)}")
        return False
    print("[PASS] Header schema check passed: ['file_id', 'prediction']")

    # 2. Row Count & Unique Filenames
    if len(df) != 16:
        print(f"[FAIL] Row count mismatch. Expected exactly 16 test predictions, got {len(df)}")
        return False
    print("[PASS] Row count check passed: Exactly 16 rows found.")

    if not df['file_id'].is_unique:
        print("[FAIL] Duplicate file_id entries found in submission file.")
        return False
    print("[PASS] Uniqueness check passed: All file_id entries are unique.")

    expected_file_ids = [f"test{i:02d}.csv" for i in range(1, 17)]
    actual_file_ids = sorted(df['file_id'].tolist())
    if actual_file_ids != expected_file_ids:
        print(f"[FAIL] Missing or unexpected test file IDs.")
        print(f"  Expected: {expected_file_ids}")
        print(f"  Actual:   {actual_file_ids}")
        return False
    print("[PASS] File ID matching check passed: Exactly test01.csv to test16.csv present.")

    # 3. Numeric Float Types & Non-null / Finite Values
    if df['prediction'].isna().any():
        print("[FAIL] Found NaN/null values in 'prediction' column.")
        return False

    if np.isinf(df['prediction'].values).any():
        print("[FAIL] Found infinite (Inf/-Inf) values in 'prediction' column.")
        return False

    if not np.issubdtype(df['prediction'].dtype, np.number):
        print(f"[FAIL] 'prediction' column is not numeric. Data type: {df['prediction'].dtype}")
        return False
    print("[PASS] Finite numeric check passed: Zero NaNs, zero Infs, strictly numeric float values.")

    # 4. Physical Damage Value Bounds [0.0, 1.0]
    preds = df['prediction'].values
    if (preds < 0.0).any() or (preds > 1.0).any():
        out_of_bounds = preds[(preds < 0.0) | (preds > 1.0)]
        print(f"[FAIL] Predictions out of physical bounds [0.0, 1.0]: {out_of_bounds}")
        return False
    print("[PASS] Physical bounds check passed: All predictions strictly lie within [0.0, 1.0].")

    print("\n[INFO] Validated Submission Preview:")
    print(df.to_string(index=False))
    print("\n[SUCCESS] SHM SUBMISSION IS 100% VALID & COMPLIANT WITH COMPETITION RULES!")
    return True


def package_and_validate_zip():
    print("\n" + "=" * 65)
    print("PACKAGING & VALIDATING PREDICTIONS.ZIP ARCHIVE")
    print("=" * 65)

    base_dir = Path(__file__).resolve().parents[1]
    zip_path = base_dir / "predictions.zip"
    pred_dir = base_dir / "predictions"

    csv_files = list(pred_dir.glob("*_predictions.csv"))
    if not csv_files:
        print(f"[FAIL] No prediction CSVs found in {pred_dir}")
        return False

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in csv_files:
            # Must be placed at root of ZIP with no subdirectories
            zf.write(f, arcname=f.name)
            print(f"  Archived {f.name} into {zip_path.name}")

    with zipfile.ZipFile(zip_path, 'r') as zf:
        namelist = zf.namelist()
        print(f"[PASS] Successfully inspected {zip_path.name}. Contents: {namelist}")
        for name in namelist:
            if "/" in name or "\\" in name:
                print(f"[FAIL] Archive contains nested folders: {name}. Must be flat at root level!")
                return False

    print(f"[SUCCESS] {zip_path.name} PACKAGED AND TESTED SUCCESSFULLY!")
    return True


if __name__ == '__main__':
    shm_ok = validate_shm()
    zip_ok = package_and_validate_zip()

    if shm_ok and zip_ok:
        print("\n[RESULT] ALL VALIDATION & ARCHIVING CHECKS PASSED PERFECTLY!")
        sys.exit(0)
    else:
        print("\n[RESULT] VALIDATION FAILED.")
        sys.exit(1)
