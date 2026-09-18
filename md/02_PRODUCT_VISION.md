# Product Vision — SHM Engineer Assistant

## One-sentence product

A simple SHM web application that helps a new rail engineer **inspect a raw dynamic-stress file, understand its signal characteristics, estimate cumulative fatigue damage, and export the official prediction format without touching Python code**.

## Primary user

A new or occasional engineer who:

- has a CSV from an SHM monitoring system;
- may not know the dataset schema in advance;
- wants to understand data quality and signal behaviour before trusting a prediction;
- needs an explainable result rather than a black-box number.

## User goals

The app should answer, in order:

1. **What did I upload?**
2. **Is the file structurally usable?**
3. **What do the raw signals look like?**
4. **What signal characteristics matter to the model?**
5. **What cumulative damage does the model estimate?**
6. **How should I interpret this estimate in context?**
7. **Can I download the official-format prediction?**

## Product principles

### Evidence before prediction

Do not immediately show a single model number. First show file metadata and data-quality checks so the engineer sees what the model received.

### Plain language first

Every technical term should have a one-line explanation or tooltip: RMS, peak-to-peak, kurtosis, rainflow cycle, MAPE, cumulative damage.

### No invented maintenance thresholds

The app must not invent red/amber/green maintenance decisions. The Info Kit explains that Miner's rule traditionally treats `D >= 1` as fatigue failure, but the hackathon model predicts dataset-specific cumulative damage for supplied segments. Present that theory as background, not as a certified maintenance decision.

### One model path

The app, batch prediction script, tests, and final submission must all call the same prediction service/functions.

### Explainability without an LLM dependency

Core explanations should be deterministic and generated from calculated statistics/features. An LLM is optional and must never create or alter the numerical damage prediction.

## MVP success criteria

The MVP is complete when a judge can:

- upload an SHM CSV;
- see the detected schema and quality checks;
- inspect one or more signals visually;
- run the trained model;
- see the predicted damage and a concise explanation;
- download a valid `file_id,prediction` CSV.
