# ACV Implementation Backlog

## P0 — official contract
- [ ] Rank-decay scorer
- [ ] Submission validator
- [ ] Unit tests ranks 1–8
- [ ] Lock output schema

## P1 — workbook ingestion
- [ ] Load `.xlsx`
- [ ] Detect worksheet/time
- [ ] Parse `Car <NN> - <parameter>`
- [ ] Preserve string IDs
- [ ] Produce data-quality report

## P2 — engineer explorer
- [ ] Workbook summary
- [ ] Car/parameter inventory
- [ ] Raw parameter plot
- [ ] Peer overlay
- [ ] Data-quality summary

## P3 — baseline peer score
- [ ] Robust per-car summaries
- [ ] Peer median/MAD
- [ ] Anomaly score
- [ ] Deterministic ranking
- [ ] LOCO evaluation

## P4 — temporal residuals
- [ ] Align cars by time
- [ ] Peer trajectory
- [ ] Residual magnitude
- [ ] Persistence features
- [ ] Compare validation

## P5 — lightweight supervised score
- [ ] Build 48 car-case records
- [ ] Strong regularisation
- [ ] Train only inside folds
- [ ] Compare/hybridise

## P6 — freeze final inference
- [ ] Select strongest stable official score
- [ ] Freeze features/config
- [ ] Save inference artefacts

## P7 — app
- [ ] Upload
- [ ] Inspect
- [ ] Rank
- [ ] Display all cars
- [ ] Explain evidence
- [ ] Download CSV

## P8 — held-out file
- [ ] Run only after pipeline frozen
- [ ] Generate `acv_predictions.csv`
- [ ] Validate output
- [ ] Do not seek/invent hidden label

## P9 — packaging
- [ ] Clean-machine test
- [ ] Docker test
- [ ] Put CSV in `predictions.zip`
- [ ] Demo video
- [ ] Final checklist

## Acceptance criteria across milestones

Use a label-independent baseline before supervised tuning; document random-ranking expected score. Add failure-path tests and a single canonical inference route before UI integration. Verify all six labels against source data, save outer-fold rankings and preserve case-wise isolation. The final demo must use the same app-generated CSV placed in predictions.zip.
