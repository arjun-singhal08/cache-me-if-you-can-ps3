# Target Repository Structure

```text
cache-me-if-you-can-ps3/
├── README.md
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
├── app.py
│
├── md/
│   └── ... project documentation ...
│
├── config/
│   ├── app.yaml
│   └── model.yaml
│
├── src/
│   └── shm/
│       ├── __init__.py
│       ├── io.py
│       ├── validation.py
│       ├── profiling.py
│       ├── features.py
│       ├── model.py
│       ├── service.py
│       ├── explain.py
│       └── submission.py
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── generate_submission.py
│
├── models/
│   ├── .gitkeep
│   └── README.md
│
├── predictions/
│   └── .gitkeep
│
└── tests/
    ├── test_validation.py
    ├── test_features.py
    ├── test_service.py
    └── test_submission.py
```

## Data policy

Do not duplicate the large official raw dataset into Git history.

Preferred local layout:

```text
workspace/
├── cache-me-if-you-can-ps3/
└── NebulaX-Hackathon-ProblemStatement/
```

Use configurable paths to point training scripts at the official repository.

## Module responsibilities

- `io.py` — safe CSV loading.
- `validation.py` — structural/data-quality checks.
- `profiling.py` — engineer-facing data summaries.
- `features.py` — deterministic feature extraction.
- `model.py` — model load/save/train utilities.
- `service.py` — canonical `predict_damage(...)` orchestration.
- `explain.py` — deterministic explanation/context generation.
- `submission.py` — official CSV formatting/validation.

## Rule

Streamlit, FastAPI, and CLI code must call `service.py`; they must not implement separate feature/model pipelines.
