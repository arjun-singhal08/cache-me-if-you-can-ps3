#!/usr/bin/env python3
"""
Unified prediction script for NebulaX PS3 - Train Condition Monitoring
Supports multiple subsystems: ACV, Door

Usage:
    python predict_all.py --subsystem ACV --input data/ACV/Test --output predictions/acv_predictions.csv
    python predict_all.py --subsystem Door --input data/Door/Test.csv --output predictions/door_predictions.csv
    python predict_all.py --all --input-dir data --output-dir predictions
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def run_acv_prediction(input_dir: str, output_file: str):
    """Run ACV prediction."""
    acv_dir = Path(__file__).parent / "ACV"
    predict_script = acv_dir / "predict.py"
    
    if not predict_script.exists():
        raise FileNotFoundError(f"ACV predict.py not found at {predict_script}")
    
    # ACV predict.py expects --input as directory, --output as file
    cmd = [
        sys.executable, str(predict_script),
        "--input", input_dir,
        "--output", output_file,
        "--model-dir", str(acv_dir / "models")
    ]
    
    print(f"Running ACV prediction: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=acv_dir, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ACV prediction failed: {result.stderr}")
        return False
    
    print(result.stdout)
    return True

def run_door_prediction(input_file: str, output_file: str):
    """Run Door prediction."""
    door_dir = Path(__file__).parent / "Door"
    predict_script = door_dir / "predict.py"
    
    if not predict_script.exists():
        raise FileNotFoundError(f"Door predict.py not found at {predict_script}")
    
    # Door predict.py expects --input as CSV file, --output as file
    cmd = [
        sys.executable, str(predict_script),
        "--input", input_file,
        "--output", output_file,
        "--model-dir", str(door_dir / "model")
    ]
    
    print(f"Running Door prediction: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=door_dir, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Door prediction failed: {result.stderr}")
        return False
    
    print(result.stdout)
    return True

def main():
    parser = argparse.ArgumentParser(description='NebulaX PS3 - Unified Prediction')
    parser.add_argument('--subsystem', choices=['ACV', 'Door', 'all'], 
                       default='all', help='Subsystem to run prediction for')
    parser.add_argument('--input', help='Input file/directory (for single subsystem)')
    parser.add_argument('--output', help='Output file (for single subsystem)')
    parser.add_argument('--input-dir', help='Input directory containing subsystem data (for --all)')
    parser.add_argument('--output-dir', help='Output directory for predictions (for --all)')
    
    args = parser.parse_args()
    
    if args.subsystem == 'all':
        if not args.input_dir or not args.output_dir:
            parser.error("--all requires --input-dir and --output-dir")
        
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        success = True
        
        # ACV
        acv_test_dir = input_dir / "ACV" / "Test"
        if acv_test_dir.exists():
            print("\n" + "="*60)
            print("Running ACV subsystem prediction...")
            print("="*60)
            success &= run_acv_prediction(
                str(acv_test_dir),
                str(output_dir / "acv_predictions.csv")
            )
        else:
            print(f"Warning: ACV test directory not found at {acv_test_dir}")
        
        # Door
        door_test_file = input_dir / "Door" / "Test.csv"
        if door_test_file.exists():
            print("\n" + "="*60)
            print("Running Door subsystem prediction...")
            print("="*60)
            success &= run_door_prediction(
                str(door_test_file),
                str(output_dir / "door_predictions.csv")
            )
        else:
            print(f"Warning: Door test file not found at {door_test_file}")
        
        if success:
            print("\n" + "="*60)
            print("All predictions completed successfully!")
            print(f"Output directory: {output_dir}")
            print("="*60)
        else:
            print("\nSome predictions failed!")
            sys.exit(1)
    
    elif args.subsystem == 'ACV':
        if not args.input or not args.output:
            parser.error("ACV requires --input and --output")
        run_acv_prediction(args.input, args.output)
    
    elif args.subsystem == 'Door':
        if not args.input or not args.output:
            parser.error("Door requires --input and --output")
        run_door_prediction(args.input, args.output)

if __name__ == "__main__":
    main()