# Requirements Traceability

## Official task

The ACV minitask is **refrigerant-leak fault localisation**.

Each case contains telemetry for all 8 cars, exactly one of which has a refrigerant leakage fault. The system must rank all cars from most likely to least likely faulty.

This is a ranking/localisation task, not plain top-1 classification.

## Official data

Training:
- `acv_case_01.xlsx`
- `acv_case_02.xlsx`
- `acv_case_03.xlsx`
- `acv_case_04.xlsx`
- `acv_case_05.xlsx`
- `acv_case_06.xlsx`
- `Train_Labels.csv`

Held-out input:
- `acv_test_case.xlsx`

Sampling interval: **30 seconds**.

## Official labels

```csv
filename,faulty_car
acv_case_01.xlsx,01
acv_case_02.xlsx,02
acv_case_03.xlsx,03
acv_case_04.xlsx,01
acv_case_05.xlsx,04
acv_case_06.xlsx,06
```

## Required output

Exact filename: `acv_predictions.csv`

```csv
file_id,ranked_cars
acv_test_case.xlsx,03|01|05|02|04|06|07|08
```

Rules:
- preserve source filename;
- include every car;
- use exact two-digit car identifiers from headers;
- separate IDs with `|`;
- there is no standard `prediction` column.

## Official score

For `n` cars and true faulty car at rank `r`:

`score = (n - (r - 1)) / n`

For 8 cars: rank 1 = 1.000, rank 2 = 0.875, rank 3 = 0.750, ..., rank 8 = 0.125, missing = 0.

## App requirement

The PS3 specification requires a non-technical-user-friendly app supporting upload, processing, clear results and downloadable output.

## Authority and example values

Use the current PS3 specification for compulsory deliverables and the ACV Info Kit for task semantics/scoring. Actual workbook headers define identifiers and available parameters. The example ranking shown in these documents is illustrative, not a prediction or a disclosed test answer.

The 30-second cadence is the documented nominal interval; inspect actual timestamps and report gaps, duplicates, or irregular intervals instead of assuming every adjacent row is 30 seconds apart.

The compulsory package is the working app, a demo video of at most 3 minutes, and predictions.zip. Generate final held-out predictions through the same app that is demonstrated.
