# Application Workflow

## Recommended MVP flow

```text
Upload SHM CSV
    ↓
File/schema validation
    ↓
Raw Data Overview
    ↓
Signal Explorer
    ↓
Feature Extraction
    ↓
Damage Model
    ↓
Prediction + explanation
    ↓
Official CSV download
```

## Step 1 — Upload

Accept `.csv` only for SHM.

Immediately reject or explain:

- empty files;
- unreadable CSVs;
- no usable numeric signal columns;
- unsupported encoding/format where recovery is unsafe.

Never show a generic stack trace to the user.

## Step 2 — Data Overview

Display a compact summary:

- rows;
- columns;
- numeric channels;
- missing cells;
- file size;
- quality status: `Ready`, `Warning`, or `Cannot predict`.

Quality status is about **input validity**, not structural health.

## Step 3 — Raw Signal Explorer

Allow engineers to:

- choose a numeric channel;
- view an interactive plot;
- inspect descriptive statistics;
- optionally compare two channels;
- read short explanations of each metric.

## Step 4 — Feature Summary

Present only the most useful derived features, grouped into understandable categories:

- amplitude/intensity;
- variability;
- extremes;
- shape/distribution;
- fatigue/cycle proxies.

Do not expose hundreds of model features in the default UI.

## Step 5 — Prediction

Display:

- predicted cumulative damage;
- model/version identifier;
- short explanation of what cumulative damage means;
- where the prediction lies relative to the **training-target distribution** as context only;
- a clear disclaimer that the hackathon model is not a certified maintenance decision system.

## Step 6 — Download

Single-file inference download:

```csv
file_id,prediction
uploaded_file.csv,0.123456
```

Batch mode may generate all test predictions into `shm_predictions.csv`.

## Optional batch workflow

For hackathon submission, support processing the whole official `Test/` folder through a local script or batch endpoint. It must call the same inference code as the UI.
