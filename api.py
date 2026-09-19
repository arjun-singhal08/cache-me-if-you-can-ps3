from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.shm.service import SHMService


ROOT = Path(__file__).resolve().parent
service = SHMService(model_path=str(ROOT / "models" / "shm_model.joblib"))

app = FastAPI(title="NebulaX PS3 SHM Model API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "subsystem": "SHM", "model_loaded": service.is_model_loaded()}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    result = service.analyze_signal(
        BytesIO(await file.read()),
        filename=file.filename or "uploaded_signal.csv",
    )
    if not result["success"]:
        raise HTTPException(status_code=422, detail=result["error"])

    signal = result["signal"]
    profile = result["profile"]
    return {
        "file_id": file.filename or "uploaded_signal.csv",
        "prediction": result["predicted_damage"],
        "explanation": result["explanation"],
        "recommendation": _recommendation(result["predicted_damage"]),
        "total_samples": len(signal),
        "mean_stress": float(profile["mean_stress"]),
        "max_stress": float(profile["max_stress"]),
        "min_stress": float(profile["min_stress"]),
        "rms": float(profile["rms_stress"]),
        "preview_signal": [float(value) for value in signal[:: max(1, len(signal) // 1600)]],
    }


def _recommendation(damage: float) -> str:
    if damage < 0.20:
        return "No urgent action is indicated. Continue routine monitoring and the normal inspection schedule."
    if damage <= 0.55:
        return "Record this result and ask the maintenance team to check the structure at the next planned service."
    return "Flag this result for the maintenance lead and arrange an engineering inspection before the next service run."