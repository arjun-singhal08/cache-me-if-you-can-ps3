# SHM Data Guide

## What the data represents

The organiser describes the SHM files as dynamic-stress data collected from rail-vehicle load-bearing structures such as carbodies and bogie frames. The monitoring system starts when a vehicle is powered on, continuously acquires dynamic stress, and periodically stores independent files containing monitoring-point data.

Each SHM CSV represents an equal-length time segment from a measurement point/monitoring context.

## Official train/test layout

```text
PS3/02_Datasets/SHM/
├── Train/
│   ├── train01.csv
│   ├── ...
│   └── train64.csv
├── Test/
│   ├── test01.csv
│   ├── ...
│   └── test16.csv
└── Train_Labels.csv
```

`Train_Labels.csv`:

```csv
filename,damage
train01.csv,0.103662995
...
```

## Critical rule: filenames are identifiers only

The organiser explicitly states that train/test file numbers are randomly assigned and do not encode recording order or cumulative damage. Never use the numeric part of the filename as a model feature.

## Operating conditions

The SHM dataset was collected across:

- two rail lines;
- AW0 and AW4 load conditions;
- healthy operating samples.

If reliable line/load metadata becomes available in the raw files, consider it when designing grouped validation. Do not infer those categories from filenames.

## What the target means

The target is cumulative fatigue damage. The supplied reference values were produced using fatigue-analysis concepts including:

- stress cycles;
- rainflow counting;
- an S-N curve;
- Miner's linear cumulative damage rule.

The challenge does **not** require the model to reproduce that exact physical calculation.

## Raw-data experience in our app

Because production-quality code should not depend on undocumented column positions, the upload pipeline should inspect the CSV dynamically.

For every uploaded file display:

- filename and file size;
- row count and column count;
- column names;
- inferred data types;
- numeric vs non-numeric columns;
- missing-value counts;
- infinite/non-finite values;
- duplicate-row count;
- constant/near-constant columns;
- basic numeric statistics;
- suspiciously extreme values;
- plot-ready numeric channels.

## Signal explorer

For selected numeric channels show:

- raw line plot;
- zoom/range selection;
- mean and standard deviation;
- min/max;
- RMS;
- peak-to-peak;
- selected percentiles;
- optional histogram;
- optional frequency-domain view only when sampling information is known or safely derivable.

For performance, downsample only for **visualisation**. Feature extraction and inference should use the intended full-resolution processing pipeline unless the model explicitly defines otherwise.

## Do not silently assume

Do not hard-code or invent:

- sampling frequency;
- stress units;
- timestamp units;
- fixed sensor/channel names;
- maintenance thresholds.

If a property is not explicitly available in the file or organiser documentation, label it as unknown.
