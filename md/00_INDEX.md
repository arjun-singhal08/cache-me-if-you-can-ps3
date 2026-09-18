# SHM Application Documentation Index

Project: **Chache Me If You Can — NebulaX 2026 PS3 / SHM**  
Solution repo: `arjun-singhal08/cache-me-if-you-can-ps3`  
Official problem repo: `aochinwen/NebulaX-Hackathon-ProblemStatement`

## Purpose

This folder is the project source of truth for the SHM application. It is intentionally focused on the **Structural Health Monitoring (SHM)** minitask of PS3.

## Read in this order

1. `01_REQUIREMENTS_TRACEABILITY.md` — what the official challenge requires and what is optional.
2. `02_PRODUCT_VISION.md` — who the app serves and what experience we are building.
3. `03_SHM_DATA_GUIDE.md` — what the official SHM data represents and how not to misuse it.
4. `04_APPLICATION_WORKFLOW.md` — end-to-end user and system flow.
5. `05_MODELING_PLAN.md` — baseline, feature engineering, validation, and model-selection plan.
6. `06_SYSTEM_ARCHITECTURE.md` — recommended technology and component boundaries.
7. `07_UI_UX_SPEC.md` — screens and interactions for new engineers.
8. `08_API_CONTRACT.md` — optional FastAPI contract if the UI is decoupled from Python.
9. `09_CONFIG_AND_ENVIRONMENT.md` — YAML/configuration/environment conventions.
10. `10_REPOSITORY_STRUCTURE.md` — target repository layout and ownership rules.
11. `11_TESTING_AND_QUALITY.md` — tests, data validation, model checks, and submission validation.
12. `12_DEPLOYMENT_DOCKER.md` — Docker and deployment plan.
13. `13_SUBMISSION_CHECKLIST.md` — exact SHM packaging and demo checks.
14. `14_TEAM_GIT_WORKFLOW.md` — safe team Git workflow.
15. `15_CODEX_WORKING_GUIDE.md` — instructions for Codex/AI-assisted development.
16. `16_EXPERIMENT_LOG.md` — template for model experiments.
17. `17_DECISIONS_AND_RISKS.md` — architecture decisions, open risks, and non-goals.

## Official references

- PS3 specification: https://github.com/aochinwen/NebulaX-Hackathon-ProblemStatement/blob/main/PS3/01_Problem_Statement_3_Specifications.md
- SHM Info Kit: https://github.com/aochinwen/NebulaX-Hackathon-ProblemStatement/blob/main/PS3/03_References/SHM/SHM_Info_Kit.md
- SHM labels: https://github.com/aochinwen/NebulaX-Hackathon-ProblemStatement/blob/main/PS3/02_Datasets/SHM/Train_Labels.csv
- Example SHM submission: https://github.com/aochinwen/NebulaX-Hackathon-ProblemStatement/blob/main/PS3/04_Example_Submission/shm_predictions.csv

## Documentation rule

If these docs conflict with the **current official PS3 specification**, follow the official specification and update these docs. Do not silently invent dataset columns, units, thresholds, or labels that are not documented by the organisers.
