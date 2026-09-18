from typing import Dict, Any

def generate_engineering_explanation(profile: Dict[str, Any], features: Dict[str, Any], predicted_damage: float) -> str:
    """
    Generates a domain-grounded engineering diagnostic summary based on statistical
    feature extraction, ASTM Rainflow cycle decomposition, and S-N power accumulators.
    
    Adheres strictly to statistical model estimates without unverified operational thresholds.
    """
    ptp = profile.get('peak_to_peak', 0.0)
    rms = profile.get('rms_stress', 0.0)
    max_range = features.get('rf_max_range', ptp)
    total_cycles = features.get('rf_total_cycles', 0.0)
    cycle_count = features.get('rf_cycle_count', 0.0)

    summary = f"""
### 📋 Statistical Diagnostic Summary

* **Estimated Cumulative Fatigue Damage ($D$)**: **`{predicted_damage:.4f}`**
  *(Statistical ExtraTrees regression estimate; uncalibrated against certified operator maintenance thresholds)*

---

#### 🔍 Signal Dynamics & ASTM E1049-85 Rainflow Decomposition
1. **Stress Amplitude Range**: Peak-to-peak variation of **`{ptp:.2f} MPa`** (Maximum Stress: `{profile.get('max_stress', 0.0):.2f} MPa`, Minimum Stress: `{profile.get('min_stress', 0.0):.2f} MPa`) with an RMS stress level of **`{rms:.2f} MPa`**.
2. **Cycle Reversals Counted**: **`{int(total_cycles):,}`** total effective cycles decomposed from **`{int(cycle_count):,}`** discrete turning loops.
3. **Maximum Stress Range Recorded**: **`{max_range:.2f} MPa`**.
4. **Fatigue Damage Accumulator Proxies**: Multi-exponent Basquin sums $\sum n_i (\Delta\sigma)^m$ computed for $m \in [2.5, 5.0]$ to capture non-linear material stress-life degradation.

---

#### ℹ️ Engineering Note
> This diagnostic report provides statistical telemetry analysis and fatigue cycle decomposition proxies based on Miner's linear cumulative damage principles to assist railway maintenance engineering teams.
"""
    return summary.strip()
