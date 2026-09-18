# ACV System Architecture

## MVP

```text
Browser
  ↓
Streamlit localhost
  ↓
ACV service
  ├── workbook loader
  ├── schema discovery
  ├── profiler
  ├── feature engine
  ├── peer comparator
  ├── car scorer
  ├── ranker
  ├── explainer
  └── submission exporter
```

## Modules

```text
src/acv/
├── io.py
├── schema.py
├── profiling.py
├── features.py
├── peer.py
├── scoring.py
├── train.py
├── rank.py
├── explain.py
├── service.py
└── submission.py
```

`service.py` should be the canonical inference path used by both UI and scripts.

## Optional backend

FastAPI + Uvicorn is optional. Add it only if a separate backend becomes useful.

Do not duplicate feature/ranking logic in the UI.

## Explanation boundary

Use deterministic, evidence-based text by default. An optional LLM receives only a structured summary of computed values, units, availability and the fixed ranking; it must not recalculate rankings, invent sensor values, claim calibrated probabilities, or prescribe unsupported maintenance actions. Provide deterministic fallback text and keep raw workbooks local unless sending them is explicitly authorised.
