# Door Testing and Quality

## Unit tests

### Timestamp parser

Test native non-zero-padded timestamps, milliseconds, minute/hour rollover, malformed values.

### Schema

Test missing required columns, optional metadata columns, numeric conversion, reordered columns.

### Segmentation

Synthetic cases:

- one clean Open cycle;
- one clean Close cycle;
- adjacent cycles separated by a gap;
- switch bounce;
- missing state flag with position movement;
- no cycle;
- partial cycle at stream edge.

### Scoring

Reproduce the official IoU matching logic with cases for:

- perfect match;
- wrong label;
- partial overlap;
- duplicate predictions;
- missed cycle;
- extra cycle.

### Submission

Require exactly:

- `start_time`
- `end_time`
- `prediction`

Validate allowed labels and `start_time < end_time`.

## End-to-end tests

Run a held-out contiguous interval from `Train.csv` through the same public service used by the app and compare with the corresponding ground-truth segments.

## Quality gates before submission

- no crash on `Test.csv`;
- no NaN boundaries/labels;
- timestamps parse and preserve order;
- no overlapping duplicate predictions unless intentionally justified;
- official local scorer runs;
- downloaded CSV re-opens correctly.
