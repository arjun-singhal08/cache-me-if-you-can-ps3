# Codex Working Guide

## Scope

This project currently targets only the NebulaX PS3 **SHM** subsystem.

## Before coding

Read:

1. `md/01_REQUIREMENTS_TRACEABILITY.md`
2. `md/03_SHM_DATA_GUIDE.md`
3. `md/05_MODELING_PLAN.md`
4. the official `PS3/03_References/SHM/SHM_Info_Kit.md`

## Non-negotiable rules

- Do not invent undocumented raw-data units, sampling frequencies, or channel names.
- Never use the numeric portion of `trainXX.csv` or `testXX.csv` as a predictive feature.
- Optimize model selection against validation MAPE.
- Keep feature extraction deterministic.
- Use one canonical inference path for app, API, CLI, and submission generation.
- Preserve the exact official output schema: `file_id,prediction`.
- Do not commit secrets or large organiser datasets.
- Do not add an LLM dependency to core inference.
- Do not change the output contract without checking the official repository.

## Preferred implementation sequence

1. input loader + validator;
2. raw data profiler;
3. feature extractor;
4. simple baselines;
5. cross-validation harness;
6. selected model + artifact metadata;
7. canonical prediction service;
8. Streamlit UI;
9. submission generator + validator;
10. Docker/deployment.

## Change discipline

For each meaningful change:

- state what requirement it serves;
- add/update tests;
- run affected tests;
- keep backwards compatibility with the current prediction service where possible;
- update docs when assumptions change.

Prefer simple, measured improvements over framework expansion.
