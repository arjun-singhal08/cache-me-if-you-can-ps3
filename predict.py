#!/usr/bin/env python3
"""
LTA NebulaX PS3 Subsystem Predictor CLI - Routes to subsystem-specific predict.py
Usage: python predict.py --subsystem <shm|acv|rail|door> --input <path> --output <path> [--model-dir <path>]
"""

import sys
import argparse
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

SUBSYSTEM_SCRIPTS = {
    "shm": REPO_ROOT / "SHM" / "predict.py",
    "acv": REPO_ROOT / "ACV" / "predict.py",
    "rail": REPO_ROOT / "Rail_Corrugation" / "predict.py",
    "door": REPO_ROOT / "Door" / "predict.py",
}

SUBSYSTEM_MODEL_DIRS = {
    "shm": REPO_ROOT / "SHM" / "models",
    "acv": REPO_ROOT / "ACV" / "models",
    "rail": REPO_ROOT / "Rail_Corrugation" / "models",
    "door": REPO_ROOT / "Door" / "model",
}

def main():
    parser = argparse.ArgumentParser(description="LTA NebulaX PS3 Subsystem Predictor CLI")
    parser.add_argument("--subsystem", type=str, required=True, choices=["shm", "acv", "rail", "door"], help="Subsystem name")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input directory containing test CSVs or single CSV file")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output predictions CSV file")
    parser.add_argument("--model-dir", "-m", type=str, default=None, help="Path to model artifacts directory")
    
    args = parser.parse_args()
    
    script = SUBSYSTEM_SCRIPTS[args.subsystem]
    if not script.exists():
        print(f"Error: Subsystem script not found: {script}", file=sys.stderr)
        sys.exit(1)
    
    model_dir = args.model_dir or SUBSYSTEM_MODEL_DIRS[args.subsystem]
    
    cmd = [
        sys.executable, str(script),
        "--input", args.input,
        "--output", args.output,
        "--model-dir", str(model_dir),
    ]
    
    # Rail uses --model instead of --model-dir and --config
    if args.subsystem == "rail":
        cmd = [
            sys.executable, str(script),
            "--input", args.input,
            "--output", args.output,
            "--model", str(model_dir / "rail_best_model_final.joblib"),
        ]
    
    print(f"Running {args.subsystem.upper()} prediction...")
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, cwd=script.parent, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    
    sys.exit(result.returncode)

if __name__ == '__main__':
    main()