from typing import Dict, Any

def generate_engineering_explanation(profile: Dict[str, Any], features: Dict[str, Any], predicted_damage: float) -> str:
    """Return a short maintenance explanation for non-specialist users."""
    ptp = profile.get('peak_to_peak', 0.0)
    rms = profile.get('rms_stress', 0.0)
    total_cycles = features.get('rf_total_cycles', 0.0)
    if predicted_damage < 0.20:
        status = "The structure looks healthy based on this recording."
    elif predicted_damage <= 0.55:
        status = "The recording shows some wear. It should be checked during the next planned service."
    else:
        status = "The recording shows a high level of wear. Arrange an engineering inspection soon."

    return (
        f"{status} The model estimates a fatigue score of {predicted_damage:.3f} "
        f"on a scale where 0 means very little expected wear and 1 means high expected wear. "
        f"The sensor recorded {int(total_cycles):,} repeated stress changes. "
        f"Stress varied by about {ptp:.2f} MPa, with a typical level of {rms:.2f} MPa."
    )
