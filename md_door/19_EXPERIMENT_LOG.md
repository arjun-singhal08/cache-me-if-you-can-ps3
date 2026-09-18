# Door Experiment Log

Copy this section for every meaningful experiment.

## Experiment ID

`DOOR-YYYYMMDD-NN`

## Goal

What is being tested?

## Data split

Which contiguous train interval(s) are training vs validation?

## Segmentation configuration

- start rule:
- end rule:
- position confirmation:
- hysteresis/debounce:
- boundary refinement:

## Classifier

- model:
- features:
- class weighting:
- hyperparameters:

## Results

- IoU-weighted F1:
- soft precision:
- soft recall:
- matched segments:
- missed segments:
- extra segments:
- mean matched IoU:
- abnormal classification observations:

## Failure analysis

Which cycles failed and why?

- start too early/late;
- end too early/late;
- duplicate/fragment;
- missed cycle;
- wrong status;
- unusual signal/state pattern.

## Decision

Keep / reject / retest.
