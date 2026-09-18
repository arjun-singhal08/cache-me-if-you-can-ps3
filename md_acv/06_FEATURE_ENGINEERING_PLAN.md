# ACV Feature Engineering Plan

## Temperature features

Where present:
- mean/median/std/min/max;
- slope and rolling variation;
- indoor vs outdoor delta;
- indoor vs cooling-control delta;
- indoor vs heating-control delta.

## Cooling-response relationships

Examples:
- `indoor_minus_outdoor`;
- `indoor_minus_cooling_setpoint`;
- absolute deviation from peer indoor median;
- persistence of temperature error;
- recovery slope.

Avoid assuming one fixed refrigerant-leak signature without validation.

## Operating/control states

For mode/status fields:
- state proportions;
- transition counts;
- longest run;
- entropy;
- disagreement with peer-majority state.

## Temporal peer residuals

For parameter `p` and car `c`:

`residual(t) = value_c(t) - median_other_cars(t)`

Summarise:
- mean residual;
- median absolute residual;
- high-percentile absolute residual;
- fraction of high-deviation samples;
- longest sustained deviation.

## Data-quality features

Track:
- missing fraction;
- invalid-information fraction;
- constant-signal fraction;
- time gaps.

Primarily use these for diagnostics; only use predictively if validation supports it.

## Rich-schema files

Extra telemetry may be explored, but the base model should not depend on parameters unavailable across validation/test cases.

## Normalisation

Prefer within-case transformations:
- robust z-score among cars;
- percentile rank among cars;
- difference or ratio to peer median.

## Comparable and physically interpretable features

Gate cooling-response features on supported cooling-mode semantics and sufficient valid observations. Calculate slopes using elapsed time and state their units; do not bridge long gaps as if observations were contiguous. Align peer comparisons at matching timestamps.

For supervised features, use a fixed training-derived feature schema plus explicit availability indicators. Fit missing-value imputation on training folds only. Do not assign zero to absent temperature measurements as if it were an actual reading. Explanations must distinguish measured values, derived comparisons, and unavailable evidence.
