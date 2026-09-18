# Requirements Traceability — PS3 SHM

## Current scope

We are tackling only **PS3 → Structural Health Monitoring (SHM)** unless the team explicitly expands scope.

## Official SHM task

The SHM task is **regression**: for each dynamic-stress time-series CSV, predict one numeric **cumulative fatigue damage** value.

Official dataset facts:

- Training files: `train01.csv` through `train64.csv`.
- Test files: `test01.csv` through `test16.csv`.
- Training labels: `Train_Labels.csv` with columns `filename,damage`.
- File numbers are random identifiers and must not be treated as time/order/damage information.
- Data covers two rail lines and two load conditions, AW0 and AW4.
- Supplied samples represent healthy operating conditions.
- Reference damage was produced using rainflow counting + Miner's linear cumulative damage rule; teams are not required to reproduce that exact calculation.

## Official scoring

SHM primary score:

`score = max(0, 1 - MAPE)`

where:

`MAPE = mean(abs(y_true - y_pred) / abs(y_true))`

Therefore model selection must optimize validation **MAPE**, not only MAE, RMSE, or R².

## Official SHM prediction file

Filename:

`shm_predictions.csv`

Schema:

```csv
file_id,prediction
test01.csv,0.1234
```

Requirements:

- one row per test file;
- `file_id` includes the source filename extension;
- `prediction` is numeric;
- no accidental pandas index column.

## Current PS3 compulsory deliverables

The current PS3 specification requires, for each attempted subsystem:

1. **A short app demo video, not more than 3 minutes.**
2. **Prediction outputs inside `predictions.zip`.**
3. **A working app for non-technical users**, supporting upload, prediction/result display, and download.

The `*_predictions.csv` files must sit directly at the top level of `predictions.zip`, with no nested prediction folder.

## Important documentation discrepancy

The SHM Info Kit still contains wording referring to a `predict.py` CLI and an older top-level README deliverable description. The current PS3 specification states that its judging rubric is the source of truth and requires the app/video/predictions package above.

Project decision:

- obey the **current PS3 specification** for judged deliverables;
- still implement a reusable prediction function and optional `predict.py` CLI for reproducibility and testing;
- never let the CLI become a separate logic path from the app.

## Overall grading implications

The official PS3 rubric also assesses:

- **Problem Fit** — fit to predictive maintenance, model comparison, explainability, UI, code quality;
- **Technical Execution** — held-out model performance;
- **Ease of Use** — clarity and usefulness for non-technical users.

Our SHM app therefore needs both a sound model and a clear raw-data/prediction experience.
