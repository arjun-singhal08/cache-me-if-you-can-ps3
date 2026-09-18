# Decisions and Risks

## Current decisions

### D1 — SHM only for the current build

Reason: complete one strong end-to-end subsystem before adding breadth.

### D2 — Streamlit first

Reason: directly satisfies the non-technical upload/predict/download requirement with minimal integration risk.

### D3 — FastAPI is optional

Reason: useful for a decoupled Next.js frontend or external API, but not needed for the MVP.

### D4 — No required LLM

Reason: the official score is based on numeric prediction accuracy and app usability. Deterministic explanations are cheaper, faster, and more reliable.

### D5 — YAML for non-secret configuration

Reason: clear, editable app/model configuration without hard-coded paths/settings.

### D6 — Docker for reproducible deployment

Reason: reduces environment mismatch and gives judges/teammates a repeatable runtime.

## Key risks

### R1 — Small labelled dataset

Only 64 labelled SHM files. Risk of unstable validation and overfitting.

Mitigation: simple models, repeated/robust CV, strong baselines, limited hyperparameter search.

### R2 — MAPE sensitivity to small targets

Training damage values include relatively small positive numbers, so percentage errors can become large.

Mitigation: optimize MAPE directly and test log-target modelling.

### R3 — Undocumented raw schema assumptions

Hard-coded channel positions/units can break the app or mislead users.

Mitigation: dynamic schema inspection and explicit “unknown” states.

### R4 — UI/model logic divergence

Separate code paths can produce different final CSVs.

Mitigation: one canonical prediction service.

### R5 — Over-engineering

Next.js + FastAPI + LLM + multiple containers can consume hackathon time without improving judging outcomes.

Mitigation: Streamlit MVP first; add architecture only when a concrete requirement justifies it.

### R6 — Unsupported safety claims

A prediction could be presented as a certified maintenance decision.

Mitigation: show cumulative-damage estimate and training context, not invented operational thresholds; explain limitations clearly.

## Open questions to resolve through data inspection

- actual CSV column names and data types;
- whether sampling frequency/time information is included;
- whether rail-line/AW0/AW4 metadata is recoverable per file;
- whether one or multiple stress channels are present in each file;
- which fatigue-aware features are stable across all files.

Resolve these from the actual files before hard-coding model assumptions.
