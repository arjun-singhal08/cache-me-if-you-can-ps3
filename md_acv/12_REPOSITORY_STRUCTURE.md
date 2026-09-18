# Recommended ACV Repository Structure

```text
cache-me-if-you-can-ps3/
├── app_acv.py
├── requirements.txt
├── config/
│   └── acv.yaml
├── src/
│   └── acv/
│       ├── io.py
│       ├── schema.py
│       ├── profiling.py
│       ├── features.py
│       ├── peer.py
│       ├── scoring.py
│       ├── train.py
│       ├── rank.py
│       ├── explain.py
│       ├── service.py
│       └── submission.py
├── models/
│   └── acv/
├── predictions/
│   └── acv_predictions.csv
├── scripts/
│   ├── train_acv.py
│   ├── validate_acv.py
│   └── predict_acv.py
├── tests/
│   └── acv/
└── md_acv/
```

Keep organiser workbooks outside the solution repository where practical.

Example:
```text
NebulaX/
├── cache-me-if-you-can-ps3/
└── NebulaX-Hackathon-ProblemStatement/
```
