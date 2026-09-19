import os
import sys
import pickle
import unittest
from pathlib import Path
import joblib

# Ensure repository root and subsystem directories are in sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
for sub_dir in [
    REPO_ROOT,
    REPO_ROOT / "ACV",
    REPO_ROOT / "Door",
    REPO_ROOT / "Rail_Corrugation",
    REPO_ROOT / "SHM",
]:
    p = str(sub_dir)
    if p not in sys.path:
        sys.path.insert(0, p)

# Register legacy module namespaces for unpickling artifacts trained within SHM/
try:
    import SHM.src.models as shm_models
    import SHM.src.models.hybrid_model as shm_hybrid
    sys.modules.setdefault('src.models', shm_models)
    sys.modules.setdefault('src.models.hybrid_model', shm_hybrid)
except Exception:
    pass


def _load_artifact(file_path: Path):
    """Load model artifact via joblib or pickle."""
    try:
        return joblib.load(file_path)
    except Exception:
        with open(file_path, "rb") as f:
            return pickle.load(f)


class TestModelSmoke(unittest.TestCase):
    """Smoke tests verifying that all trained model artifacts load into memory without errors."""

    def test_shm_models_load(self):
        """Verify SHM models load into memory."""
        shm_candidates = [
            REPO_ROOT / "models" / "shm_best_model.joblib",
            REPO_ROOT / "models" / "shm_model.joblib",
        ]
        loaded = False
        for p in shm_candidates:
            if p.exists():
                artifact = _load_artifact(p)
                self.assertIsNotNone(artifact)
                loaded = True
        self.assertTrue(loaded, "No SHM root model artifact found.")

    def test_rail_models_load(self):
        """Verify Rail Corrugation models load into memory."""
        rail_candidates = [
            REPO_ROOT / "models" / "rail_best_model.joblib",
            REPO_ROOT / "models" / "rail_model.joblib",
            REPO_ROOT / "Rail_Corrugation" / "models" / "rail_best_model_final.joblib",
        ]
        loaded = False
        for p in rail_candidates:
            if p.exists():
                artifact = _load_artifact(p)
                self.assertIsNotNone(artifact)
                loaded = True
        self.assertTrue(loaded, "No Rail Corrugation model artifact found.")

    def test_door_models_load(self):
        """Verify Door fault diagnosis models load into memory."""
        door_model_dir = REPO_ROOT / "Door" / "model"
        self.assertTrue(door_model_dir.exists(), "Door model directory not found.")
        for fname in ["classifier.pkl", "segmenter.pkl", "feature_names.pkl"]:
            target = door_model_dir / fname
            self.assertTrue(target.exists(), f"Missing Door model file: {fname}")
            artifact = _load_artifact(target)
            self.assertIsNotNone(artifact)

    def test_acv_models_load(self):
        """Verify ACV refrigerant leak models load into memory."""
        acv_model_dir = REPO_ROOT / "ACV" / "models"
        self.assertTrue(acv_model_dir.exists(), "ACV model directory not found.")
        for fname in [
            "ensemble_metadata.pkl",
            "classical_model.pkl",
            "lgbm_model.pkl",
            "xgb_model.pkl",
            "unsup_model.pkl",
            "siamese_model.pkl",
        ]:
            target = acv_model_dir / fname
            self.assertTrue(target.exists(), f"Missing ACV model file: {fname}")
            artifact = _load_artifact(target)
            self.assertIsNotNone(artifact)

    def test_all_discovered_model_artifacts_load(self):
        """Dynamically scan and load every .joblib and .pkl file in the repository."""
        all_models = (
            list(REPO_ROOT.glob("models/*.joblib"))
            + list(REPO_ROOT.glob("models/*.pkl"))
            + list(REPO_ROOT.glob("*/models*/*.joblib"))
            + list(REPO_ROOT.glob("*/models*/*.pkl"))
            + list(REPO_ROOT.glob("Door/model/*.pkl"))
        )

        self.assertGreater(len(all_models), 10, "Expected at least 10 model artifacts across repo.")
        for model_path in all_models:
            with self.subTest(model=model_path.name):
                artifact = _load_artifact(model_path)
                self.assertIsNotNone(artifact, f"Artifact failed to load: {model_path}")


if __name__ == "__main__":
    unittest.main()
