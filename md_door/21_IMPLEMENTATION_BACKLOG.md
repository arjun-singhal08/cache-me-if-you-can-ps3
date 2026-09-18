# Door Implementation Backlog

Build vertically in this order.

## P0 — understand and validate data

- [ ] Build robust native timestamp parser.
- [ ] Load `Train.csv` / `Test.csv` by column names.
- [ ] Load `Train_Segments_Answer.csv`.
- [ ] Build raw stream Plotly explorer.
- [ ] Verify observed sampling-interval distribution.

## P1 — build the official local scorer

- [ ] IoU function.
- [ ] same-label candidate matching.
- [ ] greedy highest-IoU one-to-one matching.
- [ ] soft precision/recall/F1.
- [ ] unit tests against hand-worked cases.

Do this before optimising the model.

## P2 — segmentation baseline

- [ ] Detect Open candidates.
- [ ] Detect Close candidates.
- [ ] Add movement/position confirmation.
- [ ] Add debounce/hysteresis.
- [ ] Compare candidate segments against ground truth.
- [ ] Add boundary refinement.

## P3 — cycle feature dataset

- [ ] Cut ground-truth training cycles.
- [ ] Generate time/current/voltage/EMF/position/state features.
- [ ] Add operation as internal feature.
- [ ] Validate feature schema and missing values.

## P4 — classifier baseline

- [ ] Logistic Regression.
- [ ] Random Forest.
- [ ] Extra Trees.
- [ ] Gradient Boosting.
- [ ] Compare combined vs separate Open/Close models.

## P5 — true end-to-end validation

- [ ] Define contiguous blocked folds.
- [ ] Run segmentation on raw held-out interval.
- [ ] Classify predicted segments.
- [ ] Score using official IoU-weighted F1.
- [ ] Perform failure analysis from the timeline.

## P6 — application

- [ ] Upload + schema validation.
- [ ] stream overview.
- [ ] signal explorer.
- [ ] predicted segment overlay.
- [ ] cycle-detail panel.
- [ ] downloadable official CSV.

## P7 — finalisation

- [ ] Freeze model/config.
- [ ] Run full `Test.csv`.
- [ ] Validate `door_predictions.csv`.
- [ ] Put CSV at top level of `predictions.zip`.
- [ ] Test app locally from a clean environment.
- [ ] Record demo video.

## Stretch work only after P0-P7 work

- probabilistic/learned boundary detector;
- more advanced cycle-shape features;
- optional FastAPI service;
- Docker packaging/hosted deployment;
- natural-language explanation layer.
