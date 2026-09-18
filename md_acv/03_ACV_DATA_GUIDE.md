# ACV Data Guide

## Layout

```text
PS3/02_Datasets/ACV/
├── Train/
│   ├── acv_case_01.xlsx
│   ├── acv_case_02.xlsx
│   ├── acv_case_03.xlsx
│   ├── acv_case_04.xlsx
│   ├── acv_case_05.xlsx
│   └── acv_case_06.xlsx
├── Test/
│   └── acv_test_case.xlsx
└── Train_Labels.csv
```

## Case structure

Each workbook contains continuous multivariate telemetry sampled every 30 seconds.

The official documentation describes:
- 3 identifying columns: car model, train number, time;
- ACV telemetry for 8 cars.

Per-car headers follow:

`Car <NN> - <parameter>`

Example:

`Car 03 - ACV Running Mode`

The final ranking must preserve `03`, not convert it to `3`.

## Critical schema rule

The parameter set differs between case files.

Most files record about 8 parameters per car:
- setting mode;
- running mode;
- cooling control temperature;
- heating control temperature;
- indoor average temperature;
- outdoor average temperature;
- load-halved status;
- information-valid status.

One training file contains more than 60 parameters per car.

Therefore:
- never hard-code column positions;
- discover car and parameter names from headers;
- distinguish absent parameters from missing values;
- keep a common-core feature path that works across cases.

## Label structure

There are only 6 labelled cases, one faulty car per case.

Do not treat each timestamp row as an independent labelled training example. Rows inside one case share a case-level fault identity and are correlated.

## Data checks

For every workbook:
- open successfully;
- identify worksheet/time column;
- discover car IDs;
- inspect unique cars;
- check duplicate columns;
- measure missing values;
- identify constant parameters;
- inspect information-valid flags;
- confirm time ordering;
- estimate sample interval.

## Modelling implication

Prefer features expressing how each car differs from the other seven cars in the same case.

## Parser and identifier safeguards

Read Train_Labels.csv with faulty_car as a string to preserve leading zeros. Join labels by full filename, never row order. Reject duplicate labels or labels whose car is absent from the workbook.

Inspect sheet names before choosing a sheet; report ambiguous telemetry sheets rather than silently selecting an unrelated first sheet. Reject duplicate car/parameter headers and non-parseable timestamps. Treat parameter names as schema, not positional offsets.

Use the documented eight-car count as a validation expectation. Never fabricate missing cars or silently export a partial ranking. Explain incomplete telemetry and stop final export until the case satisfies the input contract.

Decode categorical modes and information-valid flags only when their meanings are established. Do not assume a flag polarity or treat category numbers as continuous temperature-like measurements.
