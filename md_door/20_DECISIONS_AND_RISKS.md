# Door Decisions and Risks

## Decisions

### D1 — local-first Streamlit

Reason: fastest path to the official non-technical-user app requirement without mandatory hosting/signup.

### D2 — two-stage pipeline

Segmentation and classification remain explicit separate stages so failures are diagnosable.

### D3 — hybrid deterministic segmentation first

Reason: only 110 labelled cycles are available; state/position signals provide strong domain structure and a complex temporal neural model may overfit.

### D4 — classical cycle classifier first

Reason: 110 segments, with 30 abnormal examples, favour feature-based small-data models before deep learning.

### D5 — blocked temporal validation

Reason: random splitting of adjacent cycles can overestimate generalisation on a continuous stream.

## Risks

### R1 — documentation/header mismatch

The header reference lists optional metadata not present in the inspected current 17-column test header.

Mitigation: schema-by-name and optional-column tolerance.

### R2 — boundary error dominates score

A correct class with sloppy timing loses IoU credit.

Mitigation: track boundary IoU separately during debugging.

### R3 — over-segmentation

Extra fragments reduce soft precision.

Mitigation: debounce/hysteresis and one-cycle state-machine logic.

### R4 — under-segmentation

Missed cycles reduce soft recall.

Mitigation: use multiple start signals and inspect missed ground-truth examples.

### R5 — class imbalance

80 normal vs 30 abnormal.

Mitigation: class-aware validation/weights and report abnormal recall/precision alongside the official end-to-end score.

### R6 — operation-specific distributions

Open and Close cycles differ structurally.

Mitigation: include operation internally or compare separate classifiers.
