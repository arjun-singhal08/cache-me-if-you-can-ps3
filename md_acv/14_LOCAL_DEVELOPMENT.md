# Local Development

Use localhost first.

## Setup

```bash
python -m venv .venv
```

Activate:

Windows:
```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install:
```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app_acv.py
```

Open:
`http://localhost:8501`

No Streamlit account is required for localhost.

## Build order

`schema discovery → feature extraction → validation → ranking → export → UI`

Useful commands:
```bash
pytest tests/acv
python scripts/validate_acv.py
python scripts/predict_acv.py --input <file.xlsx>
```

## Plan versus implemented commands

The app_acv.py, scripts and tests shown here are target implementation entry points. Verify that they exist before running these examples; this documentation bundle itself does not implement or validate an ACV model. Integrate the ACV service into the team's shared subsystem selector for the final demonstration.
