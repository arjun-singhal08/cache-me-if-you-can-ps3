# ACV Submission Checklist

## Pipeline
- [ ] Reads `.xlsx`
- [ ] Discovers car IDs dynamically
- [ ] Handles varying parameters
- [ ] Preserves leading-zero IDs
- [ ] Processes all cars
- [ ] Reports invalid inputs clearly

## Ranking
- [ ] Every car exactly once
- [ ] Most likely faulty car first (not a safety-risk verdict)
- [ ] Deterministic ties
- [ ] Leave-one-case-out validation done
- [ ] Official score recorded
- [ ] Method/config frozen

## App
- [ ] Runs locally
- [ ] Upload works
- [ ] Overview clear
- [ ] Peer visualisation works
- [ ] Ranking clear
- [ ] Download works
- [ ] No signup required locally

## Required file

```csv
file_id,ranked_cars
acv_test_case.xlsx,03|01|05|02|04|06|07|08
```

- [ ] filename exactly `acv_predictions.csv`
- [ ] correct headers
- [ ] source filename preserved
- [ ] all cars present
- [ ] identifiers exact
- [ ] pipe separator
- [ ] no extra index

## Packaging
- [ ] `acv_predictions.csv` directly inside `predictions.zip`
- [ ] no nested folder
- [ ] app source/deployment in `app/` beneath the registered team-name folder
- [ ] final predictions generated through the demonstrated app
- [ ] optional material follows official `Optional_Items/` layout
- [ ] raw organiser workbooks excluded
- [ ] demo video shows selection/upload/result/download and is at most 3 minutes
- [ ] final package tested
