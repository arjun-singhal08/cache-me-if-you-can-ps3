# Door Submission Checklist

## Prediction pipeline

- [ ] `Test.csv` loads successfully.
- [ ] Cycles are detected from the continuous stream.
- [ ] Every predicted cycle receives one allowed status.
- [ ] End-to-end local IoU-weighted F1 evaluator works.
- [ ] Final model/config is frozen.

## `door_predictions.csv`

- [ ] Filename is exactly `door_predictions.csv`.
- [ ] Columns are exactly `start_time,end_time,prediction` for the submission copy.
- [ ] One row per predicted segment.
- [ ] Only `Normal` / `Abnormal resistance` labels.
- [ ] No DataFrame index column.
- [ ] No missing timestamps.
- [ ] Every start is before its end.
- [ ] Timestamps are parseable under official rules.
- [ ] File opens successfully after saving.

## App

- [ ] Upload works for Door CSV.
- [ ] Validation messages are understandable.
- [ ] Timeline renders.
- [ ] Detected cycles are visible.
- [ ] Prediction table renders.
- [ ] CSV download works.
- [ ] Runs locally from clean instructions.

## Hackathon package

- [ ] `door_predictions.csv` sits directly inside `predictions.zip` with any other attempted subsystem prediction files.
- [ ] No nested prediction folder.
- [ ] Final held-out inputs are run through the app to generate the submitted prediction CSVs.
- [ ] Top-level submission folder is named exactly after the registered team.
- [ ] App source/deployment is packaged in `app/` under that team folder.
- [ ] Optional write-up/code/model materials, if included, follow the official `Optional_Items/` layout.
- [ ] Demo video demonstrates upload -> prediction -> download and is <= 3 minutes.
- [ ] Raw organiser dataset is not included in the final submission package.
