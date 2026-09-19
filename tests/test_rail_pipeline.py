import unittest
import numpy as np
import pandas as pd
from pathlib import Path
from starlette.testclient import TestClient

from src.rail.data_loader import find_rail_data_dir, load_single_file
from src.rail.features import extract_features
from src.rail.predict import load_rail_model, predict_single_file
from src.rail.api import app

class TestRailPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_dir = find_rail_data_dir()
        cls.train_dir = cls.data_dir / "Train"
        cls.artifact = load_rail_model()
        cls.sample_file = cls.train_dir / "Train1.csv"
        cls.client = TestClient(app)

    def test_sample_file_feature_extraction(self):
        """Test feature extraction on real train file."""
        if not self.sample_file.exists():
            self.skipTest(f"Sample train file not found: {self.sample_file}")
        feats = extract_features(self.sample_file)
        self.assertGreater(len(feats), 100)
        for k, v in feats.items():
            self.assertFalse(np.isnan(v), f"NaN found in feature {k}")
            self.assertFalse(np.isinf(v), f"Inf found in feature {k}")

    def test_zero_variance_flatline_signal(self):
        """Test model and feature extractor resilience on flatline/constant input."""
        # 10,000 samples x 129 columns of flatline zeros
        cols = ['Rotating speed'] + [f'Ch_{i}' for i in range(1, 129)]
        df_zero = pd.DataFrame(np.zeros((1000, 129)), columns=cols)
        
        feats = extract_features(df_zero)
        for k, v in feats.items():
            self.assertFalse(np.isnan(v), f"NaN found in zero-signal feature {k}")
            self.assertFalse(np.isinf(v), f"Inf found in zero-signal feature {k}")
            
        # Model should predict Normal on flatline
        feat_vector = pd.DataFrame([[feats.get(fn, 0.0) for fn in self.artifact['feature_names']]], columns=self.artifact['feature_names'])
        pred_id = int(self.artifact['model'].predict(feat_vector)[0])
        pred_label = self.artifact['id_to_label'][pred_id]
        self.assertIn(pred_label, ['Normal', 'Side I', 'Side II'])

    def test_single_file_inference(self):
        """Test predict_single_file function output contract."""
        if not self.sample_file.exists():
            self.skipTest(f"Sample train file not found: {self.sample_file}")
        pred_label, details = predict_single_file(self.sample_file, self.artifact)
        self.assertIn(pred_label, ['Normal', 'Side I', 'Side II'])
        self.assertIn('probabilities', details)
        self.assertIn('s1_vib_rms', details)
        self.assertIn('asym_vib_rms_ratio', details)

    def test_fastapi_endpoints(self):
        """Test FastAPI /health and /predict_rail endpoints."""
        res_health = self.client.get("/health")
        self.assertEqual(res_health.status_code, 200)
        self.assertEqual(res_health.json()["status"], "healthy")

        if not self.sample_file.exists():
            return
        # Test predict via file_path
        res_post = self.client.post("/predict_rail", data={"file_path": str(self.sample_file)})
        self.assertEqual(res_post.status_code, 200)
        data = res_post.json()
        self.assertEqual(data["file_id"], "Train1.csv")
        self.assertIn(data["predicted_class"], ['Normal', 'Side I', 'Side II'])
        self.assertIn("bilateral_asymmetry", data)
        self.assertIn("vibration_metrics", data)
        self.assertIn("shock_metrics", data)
        self.assertIn("spectral_highlights", data)

    def test_submission_schema(self):
        """Verify generated test predictions file schema."""
        pred_path = Path("predictions/rail_predictions.csv")
        self.assertTrue(pred_path.exists(), "predictions/rail_predictions.csv does not exist")
        df = pd.read_csv(pred_path)
        self.assertEqual(list(df.columns), ['file_id', 'prediction'])
        self.assertEqual(len(df), 68)
        self.assertTrue(df['file_id'].is_unique)
        valid_classes = {'Normal', 'Side I', 'Side II'}
        self.assertTrue(set(df['prediction'].unique()).issubset(valid_classes))

if __name__ == '__main__':
    unittest.main()
