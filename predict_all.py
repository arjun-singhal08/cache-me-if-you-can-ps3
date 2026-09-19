#!/usr/bin/env python3
"""
Unified Prediction & Validation Pipeline for NebulaX PS3 - Train Condition Monitoring
Handles all 4 subsystems sequentially:
  1. SHM (Structural Health Monitoring - Fatigue Damage)
  2. ACV (Refrigerant Leak Localisation)
  3. Rail_Corrugation (Bilateral Track Fault Classification)
  4. Door (Door Operating Mechanism Anomaly Segmentation & Classification)

Validates CSV schemas and row counts, then bundles a flat predictions.zip archive.
"""

import os
import sys
import zipfile
import argparse
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def find_dataset_dir(subsystem: str, custom_input: str = None) -> Path:
    """Find dataset folder or file for a given subsystem with fallback candidates."""
    if custom_input:
        p = Path(custom_input)
        if p.exists():
            return p.resolve()

    candidates = [
        REPO_ROOT / "PS3" / "02_Datasets",
        REPO_ROOT.parent / "NebulaX-Hackathon-ProblemStatement" / "PS3" / "02_Datasets",
        REPO_ROOT / "data",
        Path.home() / "NebulaX-Hackathon-ProblemStatement" / "PS3" / "02_Datasets",
    ]

    for base in candidates:
        if subsystem.lower() == "shm":
            target = base / "SHM" / "Test"
            if target.exists():
                return target
        elif subsystem.lower() == "acv":
            target = base / "ACV" / "Test"
            if target.exists():
                return target
        elif subsystem.lower() in ("rail", "rail_corrugation"):
            target = base / "Rail_Corrugation" / "Test"
            if target.exists():
                return target
        elif subsystem.lower() == "door":
            target = base / "Door" / "Test.csv"
            if target.exists():
                return target
    return None


def run_shm_prediction(input_target: Path, output_file: Path) -> bool:
    """Run SHM prediction via src.shm.predict."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        from src.shm.predict import generate_shm_predictions
        print(f"[SHM] Running inference on {input_target} -> {output_file}...")
        generate_shm_predictions(input_target=input_target, output_path=output_file)
        return True
    except Exception as e:
        print(f"[SHM] Python API execution failed ({e}), attempting CLI fallback...")
        cmd = [
            sys.executable, str(REPO_ROOT / "predict.py"),
            "--subsystem", "shm",
            "--input", str(input_target),
            "--output", str(output_file)
        ]
        res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"[SHM ERROR] {res.stderr}")
            return False
        print(res.stdout)
        return True


def run_rail_prediction(input_target: Path, output_file: Path) -> bool:
    """Run Rail Corrugation prediction via src.rail.predict."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        from src.rail.predict import generate_rail_predictions
        print(f"[Rail] Running inference on {input_target} -> {output_file}...")
        generate_rail_predictions(input_path=input_target, output_path=output_file)
        return True
    except Exception as e:
        print(f"[Rail] Python API execution failed ({e}), attempting CLI fallback...")
        cmd = [
            sys.executable, str(REPO_ROOT / "predict.py"),
            "--subsystem", "rail",
            "--input", str(input_target),
            "--output", str(output_file)
        ]
        res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"[Rail ERROR] {res.stderr}")
            return False
        print(res.stdout)
        return True


def run_acv_prediction(input_dir: Path, output_file: Path) -> bool:
    """Run ACV prediction via ACV/predict.py."""
    acv_script = REPO_ROOT / "ACV" / "predict.py"
    if not acv_script.exists():
        print(f"[ACV ERROR] Script not found: {acv_script}")
        return False

    output_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(acv_script),
        "--input", str(input_dir),
        "--output", str(output_file),
        "--model-dir", str(REPO_ROOT / "ACV" / "models")
    ]
    print(f"[ACV] Running: {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=REPO_ROOT / "ACV", capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[ACV ERROR] {res.stderr}")
        return False
    print(res.stdout)
    return True


def run_door_prediction(input_file: Path, output_file: Path) -> bool:
    """Run Door prediction via Door/predict.py."""
    door_script = REPO_ROOT / "Door" / "predict.py"
    if not door_script.exists():
        print(f"[Door ERROR] Script not found: {door_script}")
        return False

    output_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(door_script),
        "--input", str(input_file),
        "--output", str(output_file),
        "--model-dir", str(REPO_ROOT / "Door" / "model")
    ]
    print(f"[Door] Running: {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=REPO_ROOT / "Door", capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[Door ERROR] {res.stderr}")
        return False
    print(res.stdout)
    return True


def validate_csv(csv_path: Path, subsystem: str) -> bool:
    """Validate prediction CSV schema, non-nulls, row counts, and value constraints."""
    if not csv_path.exists():
        print(f"[{subsystem} FAIL] File not found: {csv_path}")
        return False

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[{subsystem} FAIL] Failed to parse CSV: {e}")
        return False

    sub = subsystem.lower()
    if sub == "shm":
        expected_cols = ['file_id', 'prediction']
        if list(df.columns) != expected_cols:
            print(f"[SHM FAIL] Header mismatch: expected {expected_cols}, got {list(df.columns)}")
            return False
        if len(df) == 0:
            print("[SHM FAIL] Empty predictions file")
            return False
        if df['prediction'].isna().any() or np.isinf(df['prediction'].values).any():
            print("[SHM FAIL] Contains NaN or Inf values")
            return False
        preds = df['prediction'].values
        if (preds < 0.0).any() or (preds > 1.0).any():
            print("[SHM FAIL] Values outside [0.0, 1.0]")
            return False
        print(f"[SHM PASS] Validated {len(df)} rows. Columns: {list(df.columns)}")

    elif sub in ("rail", "rail_corrugation"):
        expected_cols = ['file_id', 'prediction']
        if list(df.columns) != expected_cols:
            print(f"[Rail FAIL] Header mismatch: expected {expected_cols}, got {list(df.columns)}")
            return False
        if len(df) == 0:
            print("[Rail FAIL] Empty predictions file")
            return False
        if df['prediction'].isna().any():
            print("[Rail FAIL] Contains NaN values")
            return False
        valid_classes = {'Normal', 'Side I', 'Side II'}
        if not set(df['prediction'].unique()).issubset(valid_classes):
            print(f"[Rail FAIL] Invalid classes: {set(df['prediction'].unique()) - valid_classes}")
            return False
        print(f"[Rail PASS] Validated {len(df)} rows. Columns: {list(df.columns)}")

    elif sub == "door":
        expected_cols = ['start_time', 'end_time', 'prediction']
        if list(df.columns) != expected_cols:
            print(f"[Door FAIL] Header mismatch: expected {expected_cols}, got {list(df.columns)}")
            return False
        if len(df) == 0:
            print("[Door FAIL] Empty predictions file")
            return False
        if df.isna().any().any():
            print("[Door FAIL] Contains NaN values")
            return False
        print(f"[Door PASS] Validated {len(df)} rows. Columns: {list(df.columns)}")

    elif sub == "acv":
        expected_cols = ['file_id', 'ranked_cars']
        if list(df.columns) != expected_cols:
            print(f"[ACV FAIL] Header mismatch: expected {expected_cols}, got {list(df.columns)}")
            return False
        if len(df) == 0:
            print("[ACV FAIL] Empty predictions file")
            return False
        if df.isna().any().any():
            print("[ACV FAIL] Contains NaN values")
            return False
        for val in df['ranked_cars']:
            cars = [c.strip() for c in str(val).split('|')]
            if len(cars) != 8:
                print(f"[ACV FAIL] Expected 8 ranked cars, found {len(cars)} in '{val}'")
                return False
            expected_set = {'01', '02', '03', '04', '05', '06', '07', '08'}
            if set(cars) != expected_set:
                print(f"[ACV FAIL] Invalid car set: {set(cars)} != {expected_set}")
                return False
        print(f"[ACV PASS] Validated {len(df)} rows. Columns: {list(df.columns)}")

    return True


def build_flat_zip(output_dir: Path, zip_file: Path) -> bool:
    """Build flat predictions.zip strictly containing the 4 required CSVs at the root."""
    target_files = [
        'shm_predictions.csv',
        'rail_predictions.csv',
        'door_predictions.csv',
        'acv_predictions.csv'
    ]

    for fname in target_files:
        src = output_dir / fname
        if not src.exists():
            print(f"[ZIP FAIL] Required file missing: {src}")
            return False

    with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fname in target_files:
            zf.write(output_dir / fname, arcname=fname)
            print(f"  Archived {fname} -> {zip_file.name}")

    with zipfile.ZipFile(zip_file, 'r') as zf:
        namelist = zf.namelist()
        print(f"[ZIP PASS] Successfully built {zip_file.name}. Contents: {namelist}")
        for name in namelist:
            if "/" in name or "\\" in name:
                print(f"[ZIP FAIL] Found nested path {name}. Must be flat at root.")
                return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="NebulaX PS3 - Unified 4-Subsystem Prediction & Archiving Pipeline"
    )
    parser.add_argument(
        "--subsystem",
        choices=["all", "shm", "rail", "rail_corrugation", "door", "acv"],
        default="all",
        help="Subsystem to run (default: all)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input path for a single subsystem"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path for a single subsystem"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Base datasets directory containing subsystem subfolders"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "predictions"),
        help="Output directory for prediction CSVs (default: predictions/)"
    )
    parser.add_argument(
        "--infer",
        action="store_true",
        help="Run full model inference on raw input datasets (default validates existing predictions if present)"
    )
    parser.add_argument(
        "--zip-path",
        type=str,
        default=str(REPO_ROOT / "predictions.zip"),
        help="Destination path for predictions.zip archive (default: predictions.zip)"
    )

    args = parser.parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = Path(args.zip_path).resolve()

    subsystems = ["shm", "acv", "rail", "door"] if args.subsystem == "all" else [args.subsystem.lower()]

    print("=" * 70)
    print("NEBULAX PS3 — UNIFIED PREDICTION & PACKAGING PIPELINE")
    print(f"Subsystems to process: {[s.upper() for s in subsystems]}")
    print(f"Output Directory: {out_dir}")
    print(f"Inference Mode: {'FORCE RECOMPUTE (--infer)' if args.infer else 'VALIDATE & PACKAGE (use --infer to re-run models)'}")
    print("=" * 70)

    success = True

    for sub in subsystems:
        print(f"\n--- Subsystem: {sub.upper()} ---")
        out_csv = out_dir / (
            f"rail_predictions.csv" if sub in ("rail", "rail_corrugation")
            else f"{sub}_predictions.csv"
        )
        if args.output and len(subsystems) == 1:
            out_csv = Path(args.output).resolve()

        # Locate dataset input
        input_p = None
        if args.input and len(subsystems) == 1:
            input_p = Path(args.input)
        elif args.input_dir:
            base = Path(args.input_dir)
            if sub == "shm":
                input_p = base / "SHM" / "Test"
            elif sub == "acv":
                input_p = base / "ACV" / "Test"
            elif sub in ("rail", "rail_corrugation"):
                input_p = base / "Rail_Corrugation" / "Test"
            elif sub == "door":
                input_p = base / "Door" / "Test.csv"
        else:
            input_p = find_dataset_dir(sub)

        # Run inference if requested or if output CSV does not exist yet
        should_run_inference = args.infer or (not out_csv.exists()) or (args.input is not None)

        if should_run_inference:
            if input_p and input_p.exists():
                if sub == "shm":
                    ok = run_shm_prediction(input_p, out_csv)
                elif sub == "acv":
                    ok = run_acv_prediction(input_p, out_csv)
                elif sub in ("rail", "rail_corrugation"):
                    ok = run_rail_prediction(input_p, out_csv)
                elif sub == "door":
                    ok = run_door_prediction(input_p, out_csv)
                else:
                    ok = False
                success &= ok
            else:
                if out_csv.exists():
                    print(f"[INFO] Raw dataset not found for {sub.upper()}; using existing predictions at {out_csv}.")
                else:
                    print(f"[FAIL] Neither raw dataset nor existing predictions found for {sub.upper()}!")
                    success = False
                    continue
        else:
            print(f"[INFO] Using existing prediction artifact at: {out_csv}")

        # Validate generated or existing CSV
        valid = validate_csv(out_csv, sub)
        success &= valid

    # If processing all, build flat zip
    if args.subsystem == "all":
        print("\n" + "=" * 70)
        print("BUILDING & VALIDATING FLAT PREDICTIONS.ZIP")
        print("=" * 70)
        zip_ok = build_flat_zip(out_dir, zip_path)
        success &= zip_ok

    if success:
        print("\n" + "=" * 70)
        print("ALL SUBSYSTEM PREDICTIONS AND PACKAGING SUCCEEDED!")
        print("=" * 70)
        sys.exit(0)
    else:
        print("\n" + "=" * 70)
        print("PIPELINE ENCOUNTERED ERRORS OR VALIDATION FAILURES.")
        print("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()