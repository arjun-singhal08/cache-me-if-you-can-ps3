# Configuration and Environment

## Python

Target the current stable Python 3.14.x series if all dependencies provide compatible wheels. Pin the exact tested Python version in Docker/CI once the environment is proven.

Do not use a Python 3.15 release candidate for the hackathon submission.

## Configuration strategy

Use YAML for non-secret application/model settings and environment variables for secrets.

Suggested future files:

```text
config/
├── app.yaml
└── model.yaml
```

Example `app.yaml`:

```yaml
app:
  title: "SHM Engineer Assistant"
  max_upload_mb: 50
  plot_max_points: 10000
  show_training_context: true

paths:
  model: "models/shm_model.joblib"
  metadata: "models/shm_model_metadata.json"
```

Example `model.yaml`:

```yaml
model:
  random_seed: 42
  target_transform: "log"
  cv_folds: 5

features:
  statistical: true
  fatigue_proxy: true
  frequency_domain: false
```

## Secrets

If an optional LLM/API is added, secrets go in environment variables such as:

`LLM_API_KEY`

Never commit `.env` or keys.

## Dependencies

Core recommendation:

- pandas
- numpy
- scipy
- scikit-learn
- joblib
- pyyaml
- plotly
- streamlit

Optional:

- fastapi
- uvicorn
- python-multipart
- pydantic-settings
- rainflow

Avoid adding libraries without a clear scoring, usability, or deployment benefit.

## Reproducibility

Pin tested versions in `requirements.txt` or a lockfile before final submission. Save model metadata including:

- training date;
- code/model version;
- feature schema version;
- training-file count;
- CV strategy;
- CV MAPE summary;
- random seed.
