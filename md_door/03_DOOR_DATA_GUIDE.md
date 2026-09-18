# Door Data Guide

## Files

The official Door dataset contains:

- `Train.csv` — continuous labelled-development stream.
- `Train_Segments_Answer.csv` — ground-truth cycle boundaries/status for `Train.csv`.
- `Test.csv` — continuous unlabelled held-out stream.

## Training segment summary

`Train_Segments_Answer.csv` currently contains **110 true cycles**:

- 80 `Normal`
- 30 `Abnormal resistance`
- 55 `Open`
- 55 `Close`
- 40 Normal Close
- 40 Normal Open
- 15 Abnormal Close
- 15 Abnormal Open

Reported segment row counts range from **137 to 190**, averaging about **164 rows**.

Operation is useful for analysis and modelling, but it is not part of the required submission output.

## Current runtime columns

The actual current `Test.csv` header is:

```text
Datetime
Motor current(mA)
Motor Voltage(10mV)
Motor electrodynamic force
Door opening time(.1s)
Door closing time(.1s)
Close command
Open command
DCSR
DCSL
DLSR
DLSL
Door Opened
Door Locked
Door is opening
Door is closing
Door leaf position
```

The sample timestamps advance by 20 ms in the inspected first rows, but code should derive cadence from timestamps instead of hard-coding 20 ms.

## Timestamp format

Native format example:

`2023-7-5-0-0-3-760`

Meaning:

`Year-Month-Day-Hour-Minute-Second-Millisecond`

Values are not necessarily zero-padded.

Runtime code must parse this explicitly and preserve the original timestamps for output.

## Important documentation inconsistency

`Door Data Headers.md` lists `Car Type`, `Car Number`, and `Door Number`, while the current official runtime CSV header inspected from `Test.csv` contains 17 fields and does not include those three metadata columns.

Implementation rule:

- trust the actual uploaded CSV header;
- identify required signal columns by name;
- tolerate optional metadata fields if present;
- do not use fixed numeric column positions.

## Important switches

- `Close command`: value 1 triggers closing.
- `Open command`: value 1 triggers opening.
- `DCSR`, `DCSL`: close switch states.
- `DLSR`, `DLSL`: locked switch states.
- `Door Opened`, `Door Locked`: state indicators.
- `Door is opening`, `Door is closing`: movement state indicators.

## Core analogue signals

- motor current;
- motor voltage;
- motor electrodynamic/back-EMF signal;
- door leaf position.

These should drive cycle understanding and abnormal-resistance features after segmentation.
