# ACV Localisation and Ranking Plan

## Constraint

Only six labelled cases are available. Deep supervised modelling is high-risk.

## Primary baseline: peer-relative anomaly ranking

```text
8-car telemetry
    ↓
per-car features
    ↓
peer-relative normalisation
    ↓
feature-level anomaly evidence
    ↓
combined car score
    ↓
rank all cars
```

## Baseline 1 — robust peer deviation

For each comparable feature:
1. calculate car summary;
2. calculate peer median;
3. calculate MAD/IQR;
4. compute robust deviation;
5. aggregate.

Concept:

`|car_value - peer_median| / (peer_MAD + epsilon)`

## Baseline 2 — temporal residual scoring

For each timestamp and parameter:
- create peer trajectory;
- compute each car's residual;
- summarise magnitude, persistence, peaks and trend.

## Baseline 3 — lightweight supervised car scorer

Build 48 records: 6 cases × 8 cars.

Target:
- 1 = faulty car in the case;
- 0 = other cars.

Candidates:
- regularised Logistic Regression;
- shallow Random Forest;
- shallow Extra Trees.

Validation must hold out whole cases.

## Hybrid

Potential final score:

`w1 * peer_anomaly + w2 * temporal_residual + w3 * supervised_score`

Weights are chosen only from training-case validation.

## Missing-feature handling

- score only features available in the workbook;
- normalise by total valid feature weight;
- do not penalise a parameter absent from the entire file;
- distinguish absent parameter from invalid/missing readings.

## Robustness and evidence limits

Compute peer summaries from the other seven cars when feasible, and compare cars in compatible operating states. A car that is off or heating should not be judged against actively cooling peers without accounting for that difference.

A small epsilon alone does not make zero-MAD scaling reliable: when peers are identical, tiny measurement differences can produce enormous scores. Use a documented scale floor, rank-based fallback, or omit unsupported features; record which fallback was used. Never divide by zero valid feature weight. Insufficient comparable data should produce an explicit failure/warning, not a confident arbitrary ranking.

Within-case scores may be calculated from the whole uploaded case without labels because all eight cars are available at inference. Any learned imputation, scaling, feature selection, score mixing or calibration must be fitted inside the training folds. Do not use car ID, filename, or train identifier as a predictive shortcut.

Relative anomaly scores are not calibrated leak probabilities. Define score normalisation before combining heterogeneous components; validate weights without using the outer held-out case.
