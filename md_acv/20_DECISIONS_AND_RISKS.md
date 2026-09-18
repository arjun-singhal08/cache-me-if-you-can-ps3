# ACV Decisions and Risks

## Decisions

### Local-first UI
Use Streamlit on localhost for MVP.

### Dynamic parsing
Never assume a universal parameter set.

### Peer-relative baseline
Compare each car against other cars in the same case.

### Leave-One-Case-Out validation
Prevents leakage from correlated rows.

### Avoid large model first
Six labelled cases are too few to justify complex modelling without evidence.

## Risks

### Small dataset
Mitigate with simple models, robust features and per-case validation.

### Parameter mismatch
Mitigate with header parsing, common-core features and graceful optional features.

### One unusually rich training workbook
Do not make baseline depend on parameters unavailable elsewhere.

### Leading-zero IDs
Keep car IDs as strings.

### UI/script divergence
Use one canonical service path.

### Overclaiming
Present ranking/evidence, not unsupported maintenance or safety conclusions.

### Validation-selection optimism

Six cases cannot support extensive tuning with a reliable untouched estimate. Pre-specify simple baselines, use case-wise inner selection where practical, and distinguish exploratory from independent performance.

### Anomaly is not automatically refrigerant leakage

Operating modes, invalid sensors and differing control settings can create peer anomalies. Condition comparisons on supported state semantics and report uncertainty; do not turn a ranking into maintenance clearance.

### Degenerate peer statistics

Zero dispersion or sparse telemetry can inflate scores. Require finite scores, documented scale fallbacks and sufficient comparable evidence before ranking.
