# ACV UI / UX Specification

## Goal

A new engineer should understand the case without reading code.

## Case Overview

Show:
- filename;
- row count;
- time range;
- sample interval;
- car identifiers;
- shared/optional parameters;
- warnings.

## Telemetry Explorer

Controls:
- parameter;
- one/all cars;
- raw vs peer-relative view.

Charts:
- traces over time;
- peer median;
- selected-car residual;
- categorical state timeline where useful.

## Car Comparison

Show an eight-row comparison table:
`Car | relative score | temperature evidence | mode evidence | data quality`

## Fault Localisation

Show the full ordered ranking with concise evidence.

If internal scores are displayed, call them **relative fault-likelihood scores**, not probabilities unless actually calibrated.

## Explanation

State:
- what was compared;
- strongest evidence;
- limitations.

## Export

Provide a clear button:
**Download acv_predictions.csv**

No local sign-in should be required.
