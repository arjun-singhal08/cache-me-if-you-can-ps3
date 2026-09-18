# Door Documentation Index

Read in this order:

1. `01_REQUIREMENTS_TRACEABILITY.md` — official requirements and source-of-truth decisions.
2. `02_PRODUCT_VISION.md` — what the Door application should achieve for engineers.
3. `03_DOOR_DATA_GUIDE.md` — dataset, labels, columns, timestamps, and known inconsistencies.
4. `04_APPLICATION_WORKFLOW.md` — end-to-end user and inference flow.
5. `05_SEGMENTATION_PLAN.md` — how to find door cycles in the continuous stream.
6. `06_CLASSIFICATION_PLAN.md` — how to classify each detected cycle.
7. `07_SCORING_AND_VALIDATION.md` — IoU-weighted F1 and local validation.
8. `08_SYSTEM_ARCHITECTURE.md` — code architecture and module boundaries.
9. `09_UI_UX_SPEC.md` — engineer-friendly interface.
10. `10_API_CONTRACT.md` — optional FastAPI interface.
11. `11_CONFIG_AND_ENVIRONMENT.md` — YAML and environment conventions.
12. `12_REPOSITORY_STRUCTURE.md` — target file/folder structure.
13. `13_TESTING_AND_QUALITY.md` — tests and quality gates.
14. `14_LOCAL_DEVELOPMENT.md` — localhost workflow; no cloud sign-up required.
15. `15_DEPLOYMENT_DOCKER.md` — Docker packaging.
16. `16_SUBMISSION_CHECKLIST.md` — final official output checks.
17. `17_TEAM_GIT_WORKFLOW.md` — simple Git workflow.
18. `18_CODEX_WORKING_GUIDE.md` — rules for AI-assisted development.
19. `19_EXPERIMENT_LOG.md` — model/segmentation experiment template.
20. `20_DECISIONS_AND_RISKS.md` — design decisions and known risks.
21. `21_IMPLEMENTATION_BACKLOG.md` — practical build order.

Core principle: **segment first, classify second, score the complete end-to-end pipeline.**
