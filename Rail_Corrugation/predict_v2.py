"""
Enhanced inference script for rail corrugation detection.
Required CLI: predict.py --input <test_folder> --output <predictions.csv>
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.angle_resample import preprocess_file
from utils.features import extract_all_features
from models.model_v2 import EnhancedRailCorrugationModel


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return ConfigDict(config)


class ConfigDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in self.items():
            if isinstance(v, dict):
                self[k] = ConfigDict(v)
            elif isinstance(v, str):
                self[k] = self._convert_numeric(v)
    
    def _convert_numeric(self, v):
        try:
            if '.' in v or 'e' in v.lower():
                return float(v)
            return int(v)
        except ValueError:
            return v
    
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'ConfigDict' object has no attribute '{key}'")


def load_model(checkpoint_path: str, config, feature_dim: int) -> EnhancedRailCorrugationModel:
    """Load model from checkpoint."""
    model = EnhancedRailCorrugationModel.load_from_checkpoint(
        checkpoint_path, config=config, handcrafted_dim=feature_dim, 
        map_location='cpu', weights_only=False
    )
    model.eval()
    model.freeze()
    return model


def preprocess_test_file(filepath: str, config, norm_stats) -> dict:
    """Preprocess a single test file with handcrafted features."""
    side_i, side_ii = preprocess_file(
        filepath,
        target_num_samples=config.data.target_num_samples
    )
    
    # Extract handcrafted features
    features = extract_all_features(side_i, side_ii)
    
    # Convert to tensors: [T, C] -> [C, T]
    side_i = torch.from_numpy(side_i).float().transpose(0, 1).unsqueeze(0)  # [1, 64, T]
    side_ii = torch.from_numpy(side_ii).float().transpose(0, 1).unsqueeze(0)  # [1, 64, T]
    features = torch.from_numpy(features).float().unsqueeze(0)  # [1, n_features]
    
    # Normalize signals
    mean = norm_stats['mean'].view(1, -1, 1)
    std = norm_stats['std'].view(1, -1, 1)
    side_i = (side_i - mean) / (std + 1e-8)
    side_ii = (side_ii - mean) / (std + 1e-8)
    
    # Normalize features
    f_mean = norm_stats['feature_mean']
    f_std = norm_stats['feature_std']
    features = (features - f_mean) / (f_std + 1e-8)
    
    return {
        "side_i": side_i, 
        "side_ii": side_ii,
        "features_i": features,
        "features_ii": features
    }


def apply_tta(model, data, config) -> list:
    """Apply test-time augmentation."""
    tta_probs = []
    device = next(model.parameters()).device
    
    side_i = data["side_i"].to(device)
    side_ii = data["side_ii"].to(device)
    features_i = data["features_i"].to(device)
    features_ii = data["features_ii"].to(device)
    
    # Original
    with torch.no_grad():
        logits_i, logits_ii, fused = model(side_i, side_ii, features_i, features_ii)
        probs = torch.softmax(fused, dim=1)
        tta_probs.append(probs.cpu())
    
    if not config.ensemble.tta:
        return tta_probs
    
    # Flip augmentation
    if "flip" in config.ensemble.tta_augments:
        with torch.no_grad():
            logits_i, logits_ii, fused = model(
                torch.flip(side_i, dims=[2]),
                torch.flip(side_ii, dims=[2]),
                features_i, features_ii
            )
            probs = torch.softmax(fused, dim=1)
            tta_probs.append(probs.cpu())
    
    # Noise augmentation
    if "noise" in config.ensemble.tta_augments:
        noise_std = config.augment.gaussian_noise_std
        with torch.no_grad():
            noisy_i = side_i + torch.randn_like(side_i) * noise_std
            noisy_ii = side_ii + torch.randn_like(side_ii) * noise_std
            logits_i, logits_ii, fused = model(noisy_i, noisy_ii, features_i, features_ii)
            probs = torch.softmax(fused, dim=1)
            tta_probs.append(probs.cpu())
    
    # Shift augmentation
    if "shift" in config.ensemble.tta_augments:
        shift = config.data.target_num_samples // 20  # ~5%
        with torch.no_grad():
            shifted_i = torch.roll(side_i, shifts=shift, dims=2)
            shifted_ii = torch.roll(side_ii, shifts=shift, dims=2)
            logits_i, logits_ii, fused = model(shifted_i, shifted_ii, features_i, features_ii)
            probs = torch.softmax(fused, dim=1)
            tta_probs.append(probs.cpu())
    
    return tta_probs


def predict_ensemble(model_paths, test_dir, output_csv, config_path):
    """Run ensemble prediction on test files."""
    
    config = load_config(config_path)
    
    # Load normalization stats from first model
    first_checkpoint = torch.load(model_paths[0], map_location='cpu', weights_only=False)
    norm_stats = first_checkpoint.get('norm_stats', None)
    
    if norm_stats is None:
        # Compute from training data if not saved
        print("Warning: No normalization stats in checkpoint. Computing from training data...")
        from data.dataset_v2 import get_dataloaders
        _, _, norm_stats = get_dataloaders(config)
    
    # Get feature dimension from a test file
    test_files = sorted([f for f in os.listdir(test_dir) if f.endswith(".csv")])
    sample_fpath = os.path.join(test_dir, test_files[0])
    sample_data = preprocess_test_file(sample_fpath, config, norm_stats)
    feature_dim = sample_data["features_i"].shape[1]
    print(f"Feature dimension: {feature_dim}")
    
    # Load all models
    models = []
    for path in model_paths:
        print(f"Loading model: {path}")
        model = load_model(path, config, feature_dim)
        models.append(model)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for m in models:
        m.to(device)
    
    # Get test files
    test_files = sorted([f for f in os.listdir(test_dir) if f.endswith(".csv")])
    print(f"Found {len(test_files)} test files")
    
    results = []
    label_map = {0: "Normal", 1: "Side I", 2: "Side II"}
    
    for fname in tqdm(test_files, desc="Predicting"):
        fpath = os.path.join(test_dir, fname)
        
        # Preprocess
        data = preprocess_test_file(fpath, config, norm_stats)
        
        # Ensemble prediction with TTA
        all_probs = []
        for model in models:
            tta_probs = apply_tta(model, data, config)
            all_probs.extend(tta_probs)
        
        # Average all predictions
        avg_probs = torch.stack(all_probs).mean(dim=0)  # [1, 3]
        pred_idx = avg_probs.argmax(dim=1).item()
        pred_label = label_map[pred_idx]
        
        results.append({"file_id": fname, "prediction": pred_label})
    
    # Save predictions
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved predictions to {output_csv}")
    print(f"Prediction distribution: {df['prediction'].value_counts().to_dict()}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Predict rail corrugation faults")
    parser.add_argument("--input", type=str, required=True, help="Test data directory")
    parser.add_argument("--output", type=str, required=True, help="Output CSV file")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file")
    parser.add_argument("--models", type=str, nargs="+", help="Model checkpoint paths")
    parser.add_argument("--model_dir", type=str, default="checkpoints", help="Model directory (for auto-discovery)")
    
    args = parser.parse_args()
    
    # Auto-discover model checkpoints if not specified
    if args.models is None:
        model_files = sorted([
            os.path.join(args.model_dir, f) 
            for f in os.listdir(args.model_dir) 
            if f.endswith(".ckpt") and "rail-" in f
        ])
        if not model_files:
            raise FileNotFoundError(f"No model checkpoints found in {args.model_dir}")
        # Use top 5 models
        model_files = model_files[-5:]
    else:
        model_files = args.models
    
    print(f"Using models: {model_files}")
    predict_ensemble(model_files, args.input, args.output, args.config)


if __name__ == "__main__":
    main()