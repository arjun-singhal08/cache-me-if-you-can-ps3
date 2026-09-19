import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from ACV.predict import (
    load_models,
    parse_car_dataframes,
    build_test_features,
    predict_case,
    predict_acv,
    predict_acv_with_diagnostics
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestACVPipeline(unittest.TestCase):
    """Unit and integration tests for ACV Refrigerant Leak Localisation inference pipeline."""

    def test_load_models(self):
        """Verify load_models loads all 5 models and correct metadata."""
        models, weights, feature_names = load_models()
        self.assertEqual(len(models), 5)
        self.assertIn('lgbm', models)
        self.assertIn('xgb', models)
        self.assertIn('classical', models)
        self.assertIn('unsup', models)
        self.assertIn('siamese', models)
        self.assertEqual(len(weights), 5)
        self.assertEqual(len(feature_names), 1133)

    def test_missing_model_raises_clear_error(self):
        """Verify that a missing model directory or artifact produces a clear FileNotFoundError."""
        with self.assertRaises(FileNotFoundError) as ctx:
            load_models(REPO_ROOT / "non_existent_model_dir")
        self.assertIn("Required ACV model directory not found", str(ctx.exception))

    def test_empty_dataframe_raises_value_error(self):
        """Verify that passing an empty DataFrame raises ValueError."""
        with self.assertRaises(ValueError):
            parse_car_dataframes(pd.DataFrame())

    def test_missing_time_column_raises_value_error(self):
        """Verify that missing Time column raises ValueError."""
        df = pd.DataFrame({
            'Car 01 - Param': [1, 2],
            'Car 02 - Param': [3, 4]
        })
        with self.assertRaises(ValueError) as ctx:
            parse_car_dataframes(df)
        self.assertIn("Time", str(ctx.exception))

    def test_fewer_than_two_cars_raises_value_error(self):
        """Verify that single-car or non-car columns raise ValueError."""
        df = pd.DataFrame({
            'Time': pd.date_range('2026-01-01', periods=5, freq='30s'),
            'Car 01 - Param': [1, 2, 3, 4, 5]
        })
        with self.assertRaises(ValueError) as ctx:
            parse_car_dataframes(df)
        self.assertIn("minimum 2 required", str(ctx.exception))

    def test_live_inference_on_competition_test_case(self):
        """Verify live inference reproduces validated competition ranking without spaces around '|'."""
        candidate_paths = [
            REPO_ROOT.parent / "NebulaX-Hackathon-ProblemStatement" / "PS3" / "02_Datasets" / "ACV" / "Test" / "acv_test_case.xlsx",
            Path.home() / "NebulaX-Hackathon-ProblemStatement" / "PS3" / "02_Datasets" / "ACV" / "Test" / "acv_test_case.xlsx",
            REPO_ROOT / "ACV" / "data" / "Test" / "acv_test_case.xlsx",
        ]
        test_file = next((p for p in candidate_paths if p.exists()), None)
        if test_file is None:
            self.skipTest("acv_test_case.xlsx not found on local filesystem.")

        pred_df, diag = predict_acv_with_diagnostics(test_file)
        self.assertEqual(list(pred_df.columns), ['file_id', 'ranked_cars'])
        self.assertEqual(len(pred_df), 1)

        ranked_str = pred_df['ranked_cars'].iloc[0]
        # Must be strictly '01|04|03|05|07|06|08|02' with no spaces
        self.assertEqual(ranked_str, "01|04|03|05|07|06|08|02")
        self.assertNotIn(" ", ranked_str)

        # Check diagnostics
        first_key = list(diag.keys())[0]
        d = diag[first_key]
        self.assertEqual(d['ranked_cars'], ['01', '04', '03', '05', '07', '06', '08', '02'])
        self.assertIn('01', d['ensemble_scores'])
        self.assertEqual(d['car_diagnostics']['01']['rank'], 1)
        self.assertIn('delta_t_mean', d['car_diagnostics']['01'])


if __name__ == "__main__":
    unittest.main()
