# Door Classification Plan

Classification happens only after a candidate cycle has been segmented.

## Target

Binary status:

- `Normal`
- `Abnormal resistance`

Training class distribution is 80 Normal / 30 Abnormal resistance.

## Recommended cycle features

### Duration and movement

- cycle duration;
- row count;
- total position travel;
- max/min position;
- position speed statistics where timestamp cadence is reliable;
- monotonicity / reversals in door position.

### Motor current

- mean, median, standard deviation;
- RMS;
- max current;
- high-percentile current;
- peak count;
- current integral/energy proxy;
- time above cycle-relative percentiles;
- current vs position-region features.

### Voltage and EMF

- mean/RMS/max;
- percentiles;
- voltage-current interaction;
- EMF extrema/variation;
- phase-specific values during initial movement, mid-travel, and final movement.

### State behaviour

- unexpected switch toggles;
- lock/open transition timing;
- command-to-movement delay;
- movement-to-final-state delay.

## Baseline models

Start with interpretable/small-data tabular models:

1. Logistic Regression
2. Random Forest
3. Extra Trees
4. Gradient Boosting

Use class weighting or balanced evaluation where appropriate.

## Operation-aware modelling

Opening and closing have different durations/signal dynamics in the labelled data. Compare:

- one combined classifier with operation as an internal feature;
- separate Open and Close classifiers.

Choose based on validation, not intuition.

## Explainability

For the UI, expose human-readable drivers such as unusually high peak current, prolonged high-current period, unusual duration, or irregular position/current relationship. Do not claim physical causation unless the data supports it.
