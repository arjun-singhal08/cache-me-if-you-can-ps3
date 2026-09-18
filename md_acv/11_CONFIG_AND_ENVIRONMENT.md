# ACV Configuration and Environment

## Core stack

- Python
- pandas
- NumPy
- SciPy
- scikit-learn
- openpyxl
- Streamlit
- Plotly
- PyYAML
- joblib
- pytest

Optional:
- FastAPI
- Uvicorn

## YAML

`config/acv.yaml`

```yaml
app:
  title: "ACV Leak Localisation Explorer"
  local_only: true
  max_upload_mb: 100

data:
  expected_car_count: 8
  sample_interval_seconds: 30
  car_column_pattern: '^Car ([0-9]{2}) - (.+)$'

features:
  use_temperature: true
  use_modes: true
  use_peer_residuals: true
  use_optional_parameters: false
  epsilon: 1.0e-8

ranking:
  method: "robust_peer"
  deterministic_tie_break: "car_id"

validation:
  strategy: "leave_one_case_out"
```

## Secrets

The core ACV pipeline needs no external secret. Keep any optional external-service key in environment variables, never Git.

## Reproducibility

Store random seed, feature version, model version, config snapshot and training case list.

## Configuration semantics

sample_interval_seconds is a nominal validation reference, not a replacement for timestamp differences. expected_car_count validates completeness; it must not generate identifiers. Treat epsilon as a numerical guard, not a validated anomaly threshold. Pin tested dependency versions and capture workbook hashes, parser version and per-fold outputs with the experiment metadata.
