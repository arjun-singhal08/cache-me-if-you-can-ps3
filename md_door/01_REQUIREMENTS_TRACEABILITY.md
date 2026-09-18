# Door Requirements Traceability

## Official task

Given a single continuous `Test.csv` stream:

1. detect each door opening/closing cycle;
2. predict each detected cycle as `Normal` or `Abnormal resistance`;
3. output one row per predicted cycle.

This is **temporal segment detection plus binary classification**. It is not file-level classification.

## Required prediction file

Filename:

`door_predictions.csv`

Required columns:

```csv
start_time,end_time,prediction
```

Allowed prediction values:

- `Normal`
- `Abnormal resistance`

Do not include a `file_id` column. `Open`/`Close` operation does not need to be predicted.

The official Info Kit accepts native timestamps or standard ISO-parseable timestamps. This project preserves native timestamps as its default. An extra `confidence` column is officially allowed but ignored for scoring; the project deliberately exports only the three required columns.

## Official metric

Door is scored by **IoU-weighted F1**. A prediction only matches a true segment if:

- labels are identical;
- time ranges overlap;
- the true/predicted pair survives one-to-one greedy highest-IoU matching.

Therefore the application must optimise three things together:

- boundary quality;
- missed/spurious segment rate;
- status classification quality.

## Mandatory app behaviour

The PS3 specification requires a non-technical user to be able to:

- upload the input data;
- run the pipeline;
- view predictions/results;
- download prediction output.

The demo video must be no longer than 3 minutes.

## Source precedence

1. Current `PS3/01_Problem_Statement_3_Specifications.md` for overall deliverables/scoring.
2. `Door_Subsystem_Info_Kit.md` for Door task and metric.
3. Actual `Train.csv` / `Test.csv` headers for runtime parsing.
4. `Door Data Headers.md` for semantic explanations.
5. `04_Example_Submission/door_predictions.csv` for exact output shape.
