# UI / UX Specification

## Navigation

Recommended Streamlit sections:

1. **Home**
2. **Upload & Data Check**
3. **Signal Explorer**
4. **Damage Prediction**
5. **How It Works**

Keep navigation shallow enough to demonstrate the full flow inside a 3-minute video.

## Home

Show:

- “Structural Health Monitoring — Cumulative Fatigue Damage Estimation”;
- one-sentence explanation;
- three-step flow: Upload → Inspect → Predict;
- link/expander for methodology.

## Upload & Data Check

After upload, show four summary cards:

- rows;
- columns;
- numeric channels;
- data-quality status.

Below them show:

- first few rows;
- column/data-type table;
- missing/non-finite values;
- warnings.

## Signal Explorer

Controls:

- channel selector;
- optional second-channel comparison;
- plot-range/point limit control.

Visuals:

- interactive line chart;
- histogram or distribution view;
- feature summary cards.

Add short tooltips for RMS, peak-to-peak, kurtosis, cycle range, and cumulative damage.

## Damage Prediction

Primary card:

**Predicted cumulative fatigue damage: `<value>`**

Supporting context:

- model version;
- validation MAPE from development, clearly labelled as validation rather than test performance;
- contextual percentile/range within training targets;
- key feature groups that influenced the prediction, if a stable explainability method is available.

Avoid unsupported claims such as “safe”, “unsafe”, or “replace component now”.

## Download

Provide a button generating exactly:

```csv
file_id,prediction
<original filename>,<numeric value>
```

## Error messages

Use actionable messages:

Bad: `ValueError: could not convert string to float`

Good: `This file has no usable numeric stress channels. Check that you uploaded an SHM CSV from the organiser dataset.`

## Accessibility/usability

- readable labels, not internal variable names;
- no dependence on colour alone;
- concise explanations near unfamiliar metrics;
- sensible number formatting;
- clear loading/progress feedback for large files;
- preserve original filename in download output.
