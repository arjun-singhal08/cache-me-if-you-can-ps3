# System Architecture

## Recommended hackathon architecture

For the current SHM-only scope, use the smallest stack that satisfies the official app requirement reliably.

### Recommended MVP

- **Python:** current stable Python 3.14.x, provided all pinned dependencies install cleanly.
- **Data/ML:** pandas, NumPy, SciPy, scikit-learn, joblib.
- **Visualisation:** Plotly.
- **Frontend/app:** Streamlit.
- **Configuration:** YAML + environment variables for secrets.
- **Deployment:** Docker.
- **LLM:** not required for the core product.

This is preferable to Next.js + FastAPI during the hackathon because Streamlit already provides upload, interactive visualisation, state, and download controls with much less integration risk.

## Optional service architecture

If the team later needs a separate frontend or API:

```text
Next.js or Streamlit UI
        ↓ HTTP
FastAPI
        ↓
SHM application service
        ↓
feature extractor + model artifact
```

Run FastAPI with **Uvicorn** (not “Unicorn”).

Keep this split optional until the model and Streamlit MVP are working.

## Internal Python boundaries

```text
UI
 ↓
Application service
 ├── input validator
 ├── data profiler
 ├── feature extractor
 ├── predictor
 ├── explainer
 └── submission formatter
        ↓
trained model artifact
```

### Key rule

`predict_damage(data)` must live outside Streamlit/FastAPI code. UI code should orchestrate and display, not implement model logic.

## Optional LLM use

An LLM may later provide plain-language descriptions of already-computed statistics. If used:

- make it optional;
- never send sensitive/raw data unless explicitly allowed;
- ground it on computed metrics and project documentation;
- never allow it to alter the numeric model prediction;
- provide deterministic fallback text when the provider is unavailable.

Given the official rubric, LLM work is lower priority than model quality, explainability, and a reliable app.
