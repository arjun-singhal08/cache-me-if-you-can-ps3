# Testing and Quality Plan

## 1. Input-validation tests

Test:

- empty CSV;
- malformed CSV;
- text-only CSV;
- missing values;
- infinite values;
- constant numeric channel;
- unusual but valid numeric column names;
- large file;
- duplicate rows.

The application should fail gracefully and explain what the user can do.

## 2. Feature tests

For a small deterministic fixture:

- feature names are stable;
- feature order is stable;
- no unintended NaN/inf output;
- same input gives same features;
- visualisation downsampling does not affect model features.

## 3. Model tests

- model artifact loads;
- expected feature schema matches the extractor;
- prediction is a finite number;
- final output is non-negative;
- inference does not use filename-number information.

## 4. Cross-validation quality checks

Record:

- per-fold MAPE;
- mean/median MAPE;
- standard deviation/range;
- MAE/RMSE as diagnostics;
- comparison against simple baselines.

Watch for one fold dominating the score because MAPE heavily penalises errors on low target values.

## 5. App smoke test

From a clean environment:

1. start app;
2. upload valid SHM CSV;
3. inspect profile;
4. open plot;
5. run prediction;
6. download CSV;
7. verify downloaded schema.

## 6. Submission validator

Before final packaging verify `shm_predictions.csv`:

- columns exactly `file_id,prediction`;
- exactly one row for each expected official test file;
- all `test01.csv`–`test16.csv` present;
- no duplicates;
- no missing values;
- all predictions numeric and finite;
- no extra index column;
- source extension preserved;
- file is directly at top level of `predictions.zip`.

## 7. Regression protection

Keep the last known-good model artifact and validation result when experimenting close to the deadline. Do not replace a working submission path without being able to restore it immediately.
