# SHM Modeling Plan

## Goal

Minimise validation **MAPE**, because the official SHM score is `max(0, 1 - MAPE)`.

## Stage 0 — Sanity baselines

Create simple reference models before anything sophisticated:

1. training-target median predictor;
2. training-target geometric-mean predictor, because all provided training targets are positive;
3. simple linear/regularised regression on basic features.

These establish whether later models genuinely improve.

## Stage 1 — File-level feature extraction

Convert each long time-series CSV into one fixed-length feature vector.

### General statistical features

For each usable numeric signal/channel:

- mean;
- standard deviation;
- variance;
- median;
- minimum/maximum;
- peak-to-peak;
- mean absolute value;
- RMS;
- 1/5/10/25/75/90/95/99 percentiles;
- interquartile range;
- skewness;
- kurtosis;
- crest factor where defined;
- mean/maximum absolute first difference;
- zero/mean crossing counts when meaningful.

### Fatigue-aware proxy features

Because the organiser's reference target is based on rainflow/Miner concepts, test explainable fatigue-related features such as:

- turning-point count;
- cycle/range counts from rainflow extraction;
- cycle-range mean/max/percentiles;
- weighted sums of cycle ranges at several powers;
- high-amplitude cycle counts;
- histogram/bin counts of stress ranges.

Do **not** claim to reproduce Miner's damage unless the required S-N constants and units are actually available and used correctly.

### Aggregation

If files contain multiple usable channels, create:

- per-channel features;
- robust across-channel aggregates where appropriate;
- channel-count/schema metadata used only if it represents real input structure rather than filename identity.

## Stage 2 — Candidate models

Start with small-data tabular regressors:

- Ridge/Elastic Net baseline;
- Random Forest Regressor;
- Extra Trees Regressor;
- HistGradientBoostingRegressor or GradientBoostingRegressor.

Only add external boosting libraries if cross-validation shows a real benefit and deployment remains simple.

## Stage 3 — Target transformation

MAPE emphasizes relative error, especially for smaller true values. Compare:

- direct prediction of `damage`;
- prediction of `log(damage)` followed by exponentiation.

All predictions should be constrained to physically/plausibly non-negative values at output.

## Validation

With only 64 labelled files, a single hold-out split is unstable.

Recommended baseline:

- fixed-seed 5-fold cross-validation;
- optionally repeated 5-fold CV for stability;
- primary reported metric: mean MAPE;
- also report fold spread, MAE, RMSE, and R² for diagnostics only.

If reliable line/load-condition grouping is exposed in the data, compare grouped validation to random K-fold to test robustness.

## Leakage rules

Never use:

- train/test filename number as a feature;
- hidden test labels;
- any statistic calculated using validation labels during feature generation;
- test-set target assumptions.

Feature preprocessing that learns parameters must be fitted inside each CV training fold.

## Model selection rule

Select the model by cross-validated MAPE, then retrain it on all 64 labelled training files before predicting the 16 official test files.

Record every serious experiment in `16_EXPERIMENT_LOG.md`.
