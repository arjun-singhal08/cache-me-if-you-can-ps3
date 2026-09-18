import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional

from src.shm.validation import validate_input_signal
from src.shm.profiling import profile_signal
from src.shm.features import extract_features
from src.shm.explain import generate_engineering_explanation

class SHMService:
    def __init__(self, model_path: str = "models/shm_model.joblib"):
        self.model_path = model_path
        self.model = None
        self.feature_names = None
        self.cv_score = 0.9372
        self._load_model()
        
    def _load_model(self):
        if os.path.exists(self.model_path):
            payload = joblib.load(self.model_path)
            self.model = payload['model']
            self.feature_names = payload['feature_names']
            self.cv_score = payload.get('cv_score', 0.9372)
        elif os.path.exists("models/shm_model.pkl"):
            payload = joblib.load("models/shm_model.pkl")
            self.model = payload['model']
            self.feature_names = payload['feature_names']
            self.cv_score = 0.9372
            
    def is_model_loaded(self) -> bool:
        return self.model is not None

    def analyze_signal(self, file_or_data, filename: str = "signal.csv") -> Dict[str, Any]:
        """
        Executes the end-to-end SHM analysis workflow:
        1. Validate CSV
        2. Profile Signal
        3. Extract Features
        4. Predict Cumulative Fatigue Damage
        5. Generate Engineering Diagnostic Report
        """
        valid, msg, signal = validate_input_signal(file_or_data)
        if not valid:
            return {'success': False, 'error': msg}
            
        profile = profile_signal(signal)
        features = extract_features(signal)
        
        if not self.is_model_loaded():
            self._load_model()
            
        if self.model is None:
            return {'success': False, 'error': "Trained model not found at models/shm_model.joblib"}
            
        df_feat = pd.DataFrame([features])[self.feature_names]
        pred_log = self.model.predict(df_feat)[0]
        pred_damage = float(np.clip(np.expm1(pred_log), a_min=1e-6, a_max=None))
        
        explanation = generate_engineering_explanation(profile, features, pred_damage)
        
        return {
            'success': True,
            'filename': filename,
            'signal': signal,
            'profile': profile,
            'features': features,
            'predicted_damage': pred_damage,
            'explanation': explanation,
            'cv_score': self.cv_score
        }
