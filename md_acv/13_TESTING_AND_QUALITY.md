# ACV Testing and Quality

## Schema tests

Test:
- normal `Car 03 - ...` header;
- all car IDs;
- unknown parameters;
- extra metadata;
- varied parameter counts.

## Feature tests

Test:
- missing values;
- constant series;
- absent optional parameters;
- categorical modes;
- peer residuals;
- zero-safe robust scaling.

## Score tests

Expected rank scores:
`1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125`
and missing = `0`.

## Submission tests

Verify:
- exact filename;
- exact headers `file_id,ranked_cars`;
- every car once;
- no duplicates;
- no missing cars;
- leading zeros preserved;
- pipe separator;
- no extra dataframe index.

## Integration test

`load → inspect → features → score → rank → export`

must run without manual intervention.

## UI smoke test

Start app, upload workbook, inspect chart, run ranking, download CSV, verify downloaded schema.

Malformed input should produce an actionable error, not a raw stack trace.

## Additional failure and leakage checks

- Empty/corrupt workbook, ambiguous sheets and duplicate headers.
- Unknown/missing/duplicate car IDs and label joins preserving leading zeros.
- Invalid flags, unsupported mode encodings, irregular timestamps and large gaps.
- All-NaN parameters, zero peer MAD, no comparable peers and zero total feature weight.
- Infinities and NaNs in intermediate features or final scores.
- Column/car-order permutations leave the semantic ranking unchanged.
- Tied scores use the documented stable identifier tie-break.
- Outer validation case labels cannot influence fitted preprocessing, feature selection or weights.
- UI, script and exported CSV agree for the same workbook and frozen configuration.
- Exported results belong to the current upload, not a previously saved case.
