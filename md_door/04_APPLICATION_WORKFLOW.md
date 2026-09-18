# Door Application Workflow

## User flow

```text
Upload Door CSV
   -> validate schema/timestamps
   -> show stream summary
   -> visualise key signals
   -> detect candidate cycles
   -> review detected boundaries
   -> extract cycle features
   -> classify Normal / Abnormal resistance
   -> show timeline + cycle table
   -> download door_predictions.csv
```

## Inference flow

### Stage A — input validation

Check:

- required columns exist;
- timestamps parse;
- timestamps are monotonic or can be safely sorted;
- no duplicate timestamps that break segmentation;
- analogue signals are numeric;
- binary/state columns contain expected values or are normalised safely.

### Stage B — segmentation

Detect cycle start/end boundaries using a state-machine or hybrid method. Do not classify yet.

### Stage C — cycle profiling

For every candidate cycle compute:

- operation guess (`Open`/`Close`) for internal use;
- duration;
- row count;
- position change;
- current/voltage/EMF features;
- command/state-transition features.

### Stage D — classification

Predict:

- `Normal`
- `Abnormal resistance`

### Stage E — output

Return a table with at least:

- start time;
- end time;
- prediction.

The app may additionally display operation, confidence, duration, and explanatory features, but only the official three columns should be required for submission.
