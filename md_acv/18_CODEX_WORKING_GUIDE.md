# Codex Working Guide — ACV

Read first:
1. `01_REQUIREMENTS_TRACEABILITY.md`
2. `03_ACV_DATA_GUIDE.md`
3. `05_LOCALISATION_RANKING_PLAN.md`
4. `07_SCORING_AND_VALIDATION.md`
5. `12_REPOSITORY_STRUCTURE.md`

## Rules

- Output contract is `file_id,ranked_cars`.
- Preserve `03` as `03`.
- Parse workbook structure from headers.
- Never hard-code one universal workbook width.
- Never randomly split rows from the same case.
- Always rank every detected car.
- Prefer simple robust methods before complex neural approaches.
- Never create hidden test labels.
- Keep inference deterministic.
- Run scorer, validation, submission and UI tests before completion.
