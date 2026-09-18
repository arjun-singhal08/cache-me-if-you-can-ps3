# Door Segmentation Plan

Segmentation is the first major modelling problem. Classification quality cannot recover a cycle that was never detected or whose boundary is badly placed.

## Baseline approach: deterministic state machine

Use command/state transitions to identify candidate cycles.

Candidate start evidence may include:

- `Open command` rising/active;
- `Close command` rising/active;
- `Door is opening` becoming active;
- `Door is closing` becoming active;
- meaningful `Door leaf position` movement.

Candidate end evidence may include:

- movement state becoming inactive;
- `Door Opened` confirmation for opening;
- `Door Locked` / close/lock switches confirming closing;
- door position becoming stable;
- a short persistence/hysteresis requirement to avoid noisy toggles.

## Why not use one hard threshold

The official Door Info Kit explicitly warns that distributions differ among doors and that fixed uniform thresholds can cause false alarms/misses. The Info Kit also warns not to assume any single state column is always the most robust boundary signal.

Therefore segmentation should combine multiple signals.

## Recommended progression

### V0 — command/state baseline

Simple rule-based start/end detection.

### V1 — state + position confirmation

Require movement/position evidence to confirm state transitions.

### V2 — boundary refinement

After finding a coarse segment, search a small window around its edges for the first/last meaningful movement/current change.

### V3 — learned boundary model only if needed

Only attempt a learned temporal model if V1/V2 cannot reach adequate local IoU. The small amount of labelled data makes a complex sequence model risky.

## Guardrails

- Minimum/maximum plausible duration must come from training data statistics, not arbitrary values.
- Merge duplicate candidates referring to one physical cycle.
- Avoid generating tiny fragments from switch bounce.
- Preserve timestamps exactly.
- Never tune on `Test.csv` answers; they are unavailable by design.
