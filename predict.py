import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.shm.predict import generate_shm_predictions

def main():
    parser = argparse.ArgumentParser(description="LTA NebulaX PS3 Subsystem Predictor CLI")
    parser.add_argument("--subsystem", type=str, default="shm", choices=["shm", "acv", "rail", "door"], help="Subsystem name")
    parser.add_argument("--input", "-i", type=str, default=None, help="Input directory containing test CSVs or single CSV file")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output predictions CSV file")
    parser.add_argument("--model", "-m", type=str, default=None, help="Path to trained model artifact")
    args = parser.parse_args()

    if args.subsystem == "shm":
        generate_shm_predictions(input_target=args.input, output_path=args.output, model_path=args.model)
    else:
        print(f"Subsystem {args.subsystem} scheduled in Development Plan.")

if __name__ == '__main__':
    main()
