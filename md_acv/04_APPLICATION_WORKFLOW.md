# ACV Application Workflow

```text
Upload ACV .xlsx
    ↓
Validate workbook
    ↓
Discover cars and parameters
    ↓
Show raw-data overview
    ↓
Explore telemetry and peer comparisons
    ↓
Extract car-level features
    ↓
Calculate relative fault scores
    ↓
Rank every car
    ↓
Explain top-ranked evidence
    ↓
Download acv_predictions.csv
```

## Input

Primary input is one `.xlsx` case. Preserve the original filename.

## Schema discovery

Automatically identify:
- time;
- metadata;
- car IDs;
- available parameters for each car.

## Overview

Show:
- filename;
- rows;
- time range;
- sample interval;
- detected cars;
- parameters/car;
- shared and optional parameters;
- missing or invalid readings.

## Exploration

Useful views:
- parameter selector;
- all-car overlay;
- individual-car trace;
- peer-median comparison;
- residual/deviation plot;
- feature comparison.

## Ranking

Return all cars from highest to lowest relative fault likelihood.

Tie-breaking must be deterministic.

## Explanation

Describe measured evidence without claiming unsupported physical causation.

## Export

```csv
file_id,ranked_cars
acv_test_case.xlsx,03|01|05|02|04|06|07|08
```
