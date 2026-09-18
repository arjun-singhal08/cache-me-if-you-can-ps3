# Door System Architecture

## Local-first architecture

```text
Streamlit UI
   -> Door service
       -> CSV loader/validator
       -> timestamp parser
       -> segmenter
       -> feature extractor
       -> classifier
       -> scorer/explainer
       -> submission formatter
```

## Recommended modules

```text
src/door/
  io.py
  schema.py
  timestamps.py
  segment.py
  features.py
  train.py
  classify.py
  score.py
  explain.py
  service.py
  submission.py
```

## Boundary rules

- UI code must not contain segmentation/model logic.
- `service.py` is the single application-facing entry point.
- `segment.py` does not know about Streamlit/FastAPI.
- `classify.py` only receives cycle-level features or segment data.
- `submission.py` generates the exact official CSV columns.
- No code imports from other PS3 subsystem model folders.

## Model artefacts

```text
models/door/
  classifier.joblib
  feature_schema.json
  model_metadata.json
```

Segmentation configuration may be saved in YAML; a learned segmenter, if later created, gets its own Door-only artefact path.
