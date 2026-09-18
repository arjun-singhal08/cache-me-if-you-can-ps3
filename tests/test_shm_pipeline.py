import unittest
import numpy as np
import pandas as pd
from pathlib import Path

from src.shm.features import extract_features
from src.shm.predict import predict_single_signal, generate_shm_predictions
from src.shm.data_loader import find_shm_data_dir

class TestSHMPipeline(unittest.TestCase):

    def test_constant_and_zero_variance_signals(self):
        """Test that constant signals do not throw division-by-zero errors or NaNs."""
        # 1. All zeros
        zero_signal = np.zeros(10000, dtype=np.float64)
        feats_zero = extract_features(zero_signal)
        self.assertIsInstance(feats_zero, dict)
        for k, v in feats_zero.items():
            self.assertFalse(np.isnan(v), f"Feature {k} was NaN for zero signal")
            self.assertFalse(np.isinf(v), f"Feature {k} was Inf for zero signal")

        pred_zero = predict_single_signal(zero_signal)
        self.assertTrue(0.0 <= pred_zero <= 1.0, f"Prediction {pred_zero} out of bounds")

        # 2. Constant non-zero flatline
        const_signal = np.full(10000, 42.5, dtype=np.float64)
        feats_const = extract_features(const_signal)
        for k, v in feats_const.items():
            self.assertFalse(np.isnan(v), f"Feature {k} was NaN for const signal")
            self.assertFalse(np.isinf(v), f"Feature {k} was Inf for const signal")

        pred_const = predict_single_signal(const_signal)
        self.assertTrue(0.0 <= pred_const <= 1.0, f"Prediction {pred_const} out of bounds")

    def test_single_file_and_batch_consistency(self):
        """Verify single file inference produces exact same output as batch inference."""
        data_dir = find_shm_data_dir()
        test_file = data_dir / "Test" / "test01.csv"
        self.assertTrue(test_file.exists(), f"Test file not found: {test_file}")

        pred_single = predict_single_signal(test_file)
        self.assertIsInstance(pred_single, float)
        self.assertTrue(0.0 <= pred_single <= 1.0)

    def test_cli_directory_inference(self):
        """Test directory-level inference and output generation."""
        data_dir = find_shm_data_dir()
        test_dir = data_dir / "Test"
        out_csv = Path("predictions/test_cli_output.csv")
        
        df = generate_shm_predictions(input_target=test_dir, output_path=out_csv)
        self.assertEqual(len(df), 16)
        self.assertListEqual(list(df.columns), ['file_id', 'prediction'])
        self.assertTrue(df['file_id'].is_unique)
        self.assertFalse(df['prediction'].isna().any())
        self.assertTrue(((df['prediction'] >= 0.0) & (df['prediction'] <= 1.0)).all())

        if out_csv.exists():
            out_csv.unlink()

if __name__ == '__main__':
    unittest.main()
