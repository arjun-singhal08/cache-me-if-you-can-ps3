# ACV Documentation Index

Read these files in order.

| # | Document | Purpose |
|---|---|---|
| 1 | `01_REQUIREMENTS_TRACEABILITY.md` | Official requirements mapped to implementation |
| 2 | `02_PRODUCT_VISION.md` | Engineer-facing product goal |
| 3 | `03_ACV_DATA_GUIDE.md` | Dataset, labels, schema caveats |
| 4 | `04_APPLICATION_WORKFLOW.md` | End-to-end user and inference flow |
| 5 | `05_LOCALISATION_RANKING_PLAN.md` | Core ranking strategy |
| 6 | `06_FEATURE_ENGINEERING_PLAN.md` | Per-car and peer-relative features |
| 7 | `07_SCORING_AND_VALIDATION.md` | Official score and validation |
| 8 | `08_SYSTEM_ARCHITECTURE.md` | Application architecture |
| 9 | `09_UI_UX_SPEC.md` | Engineer-friendly UI |
| 10 | `10_API_CONTRACT.md` | Optional FastAPI interface |
| 11 | `11_CONFIG_AND_ENVIRONMENT.md` | YAML/environment configuration |
| 12 | `12_REPOSITORY_STRUCTURE.md` | Recommended project layout |
| 13 | `13_TESTING_AND_QUALITY.md` | Tests and quality gates |
| 14 | `14_LOCAL_DEVELOPMENT.md` | Local-first workflow |
| 15 | `15_DEPLOYMENT_DOCKER.md` | Docker plan |
| 16 | `16_SUBMISSION_CHECKLIST.md` | Final ACV checks |
| 17 | `17_TEAM_GIT_WORKFLOW.md` | Beginner Git workflow |
| 18 | `18_CODEX_WORKING_GUIDE.md` | Codex guardrails |
| 19 | `19_EXPERIMENT_LOG.md` | Experiment template |
| 20 | `20_DECISIONS_AND_RISKS.md` | Decisions and risks |
| 21 | `21_IMPLEMENTATION_BACKLOG.md` | Prioritised build sequence |

## Golden rules

1. Parse each workbook from its actual column headers.
2. Never assume all cases have the same parameter set.
3. Preserve two-digit car IDs such as `03`.
4. Rank every car in the case.
5. Validate by complete case, never by random rows.
6. Prefer simple case-relative methods because there are only six labelled training cases.
7. Export exactly `acv_predictions.csv`.
