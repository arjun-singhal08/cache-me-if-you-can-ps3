# SHM Submission Checklist

## Official SHM deliverables

- [ ] Working app usable without touching code.
- [ ] Demo video is not more than 3 minutes.
- [ ] `shm_predictions.csv` generated from official held-out test input files.
- [ ] `predictions.zip` created.
- [ ] `shm_predictions.csv` is directly inside `predictions.zip`, not in a nested folder.

## Prediction CSV

- [ ] Filename is exactly `shm_predictions.csv`.
- [ ] Columns are exactly `file_id,prediction`.
- [ ] Every expected test file appears once.
- [ ] File extensions are included.
- [ ] Prediction values are numeric and finite.
- [ ] No pandas index column.
- [ ] No missing/duplicate rows.

## App demo flow

The video should show, end-to-end:

1. open app;
2. explain SHM in one sentence;
3. upload/drag in an SHM CSV;
4. show raw-data overview;
5. inspect one signal/chart;
6. run damage prediction;
7. show result and explanation;
8. download prediction CSV.

## Optional strengthening material

- [ ] README with setup/run instructions.
- [ ] Short write-up explaining features, model selection, CV, MAPE, assumptions, and limitations.
- [ ] Development code and trained model included/linked appropriately.
- [ ] Validation result table available for judges/team.

## Packaging rule

Do **not** include organiser raw datasets or the example-submission files in the final submission package.
