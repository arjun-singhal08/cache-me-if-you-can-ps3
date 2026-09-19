# Senlytics V2 Frontend

The root `app.py` is the unified operator interface for all four PS3 subsystems.

## Run locally
```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## V2 additions
- Senlytics product identity and simplified hero
- Four complete themes: Operations Dark, Operations Light, Midnight Teal, High Contrast
- Human-readable operator assessment cards grounded in model outputs
- Optional engineering/technical detail expanders
- Genuine Rainflow visualization using the SHM physics implementation
- Genuine live ACV inference using the restored five-model ensemble artifacts
- Rail, SHM, Door and ACV comparison workspaces when multiple inputs are uploaded
- Door anomaly filtering, raw sensor evidence and cycle timeline
- ACV primary-suspect highlighting and thermal telemetry
- Session-only Recent Analyses workspace
- Processing-time feedback, friendlier errors, exact competition CSV downloads
- Optional mouse-wheel chart zoom; Plotly pan/zoom/reset/export remains available
- Python 3.12 Docker runtime aligned with the tested local environment

## Safety / interpretation
Operator guidance intentionally avoids unsupported claims such as remaining-life percentages, mandatory train withdrawal, fixed maintenance deadlines, or declaring lower-ranked ACV cars healthy. Model outputs remain condition-monitoring evidence to support engineering inspection decisions.
