import io
from pathlib import Path
from typing import Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np

from src.rail.predict import load_rail_model, predict_single_file
from src.rail.features import extract_features
from src.rail.data_loader import validate_vibration_data

app = FastAPI(
    title="Rail Corrugation Fault Diagnostic API",
    description="LTA NebulaX Hackathon - Problem Statement 3: Rail Axle-Box Vibration Diagnostic Service",
    version="1.0.0"
)

# Global model artifact cache
_artifact = None

def get_artifact():
    global _artifact
    if _artifact is None:
        _artifact = load_rail_model()
    return _artifact


class RailPredictionRequest(BaseModel):
    file_path: Optional[str] = None


class RailPredictionResponse(BaseModel):
    file_id: str
    predicted_class: str
    probabilities: Dict[str, float]
    bilateral_asymmetry: Dict[str, float]
    vibration_metrics: Dict[str, float]
    shock_metrics: Dict[str, float]
    spectral_highlights: Dict[str, float]
    diagnostic_summary: str


@app.on_event("startup")
def startup_event():
    # Preload model on startup
    get_artifact()


@app.get("/health")
def health():
    return {"status": "healthy", "service": "Rail Corrugation Diagnostic API"}


@app.post("/predict_rail", response_model=RailPredictionResponse)
async def predict_rail(
    file: Optional[UploadFile] = File(None),
    file_path: Optional[str] = Form(None)
):
    """
    Classify axle-box vibration time series for rail corrugation:
    Returns: 'Normal', 'Side I', or 'Side II' corrugation with detailed physics-grounded telemetry.
    Accepts either multipart file upload OR file_path on disk.
    """
    artifact = get_artifact()
    model = artifact['model']
    feature_names = artifact['feature_names']
    id_to_label = artifact['id_to_label']

    df = None
    file_id = "unknown_sample.csv"

    if file is not None and file.filename:
        file_id = file.filename
        content = await file.read()
        try:
            df = pd.read_csv(io.BytesIO(content))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse CSV file: {str(e)}")
    elif file_path:
        p = Path(file_path)
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        file_id = p.name
        try:
            df = pd.read_csv(p)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")
    else:
        raise HTTPException(
            status_code=400,
            detail="Either 'file' upload or 'file_path' parameter must be provided."
        )

    # Validate dataframe
    try:
        df = validate_vibration_data(df)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Validation failed: {str(e)}")

    # Extract features
    feats = extract_features(df)
    feat_vector = pd.DataFrame([[feats.get(fn, 0.0) for fn in feature_names]], columns=feature_names)

    pred_id = int(model.predict(feat_vector)[0])
    pred_class = id_to_label[pred_id]

    probs = {}
    if hasattr(model, 'predict_proba'):
        prob_vals = model.predict_proba(feat_vector)[0]
        for idx, p in enumerate(prob_vals):
            probs[id_to_label[idx]] = float(p)

    # Summary diagnostic note
    if pred_class == "Normal":
        diag = "Both Left (Side I) and Right (Side II) rails exhibit balanced dynamic vibration within normal baseline operating limits."
    elif pred_class == "Side I":
        diag = (
            f"Anomalous high-frequency vibration detected on Left Rail (Side I). "
            f"Bilateral RMS asymmetry ratio is {feats.get('asym_vib_rms_ratio', 1.0):.2f} (elevated Left energy)."
        )
    else:
        diag = (
            f"Anomalous high-frequency vibration detected on Right Rail (Side II). "
            f"Bilateral RMS asymmetry ratio is {feats.get('asym_vib_rms_ratio', 1.0):.2f} (elevated Right energy)."
        )

    return RailPredictionResponse(
        file_id=file_id,
        predicted_class=pred_class,
        probabilities=probs,
        bilateral_asymmetry={
            "asym_vib_rms_ratio": feats.get("asym_vib_rms_ratio", 1.0),
            "asym_vib_energy_ratio": feats.get("asym_vib_energy_ratio", 1.0),
            "norm_asym_vib_index": feats.get("norm_asym_vib_index", 0.0),
            "max_car_asym_ratio": feats.get("max_car_asym_ratio", 1.0),
            "mean_cross_correlation": feats.get("mean_cross_correlation", 0.0)
        },
        vibration_metrics={
            "s1_vib_mean": feats.get("s1_vib_mean", 0.0),
            "s1_vib_rms": feats.get("s1_vib_rms", 0.0),
            "s1_vib_ptp": feats.get("s1_vib_ptp", 0.0),
            "s2_vib_mean": feats.get("s2_vib_mean", 0.0),
            "s2_vib_rms": feats.get("s2_vib_rms", 0.0),
            "s2_vib_ptp": feats.get("s2_vib_ptp", 0.0),
        },
        shock_metrics={
            "s1_shk_rms": feats.get("s1_shk_rms", 0.0),
            "s2_shk_rms": feats.get("s2_shk_rms", 0.0),
            "asym_shk_rms_ratio": feats.get("asym_shk_rms_ratio", 1.0),
            "asym_shk_energy_ratio": feats.get("asym_shk_energy_ratio", 1.0),
        },
        spectral_highlights={
            "s1_dominant_freq_hz": feats.get("s1_dom_freq", 0.0),
            "s2_dominant_freq_hz": feats.get("s2_dom_freq", 0.0),
            "s1_spectral_centroid_hz": feats.get("s1_spec_centroid", 0.0),
            "s2_spectral_centroid_hz": feats.get("s2_spec_centroid", 0.0),
            "pinned_pinned_resonance_imbalance": feats.get("b3_300_800hz_spectral_imbalance", 0.0),
            "corrugation_wear_imbalance": feats.get("b4_800_1500hz_spectral_imbalance", 0.0)
        },
        diagnostic_summary=diag
    )
