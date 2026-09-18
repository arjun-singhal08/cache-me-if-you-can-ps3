# Codex Working Guide — Door

Before modifying Door code:

1. Read `md_door/00_INDEX.md`.
2. Read `md_door/01_REQUIREMENTS_TRACEABILITY.md`.
3. Read official `Door_Subsystem_Info_Kit.md` and `Door Data Headers.md`.
4. Inspect the actual CSV header before assuming column positions.
5. Preserve the exact official output contract.

## Rules

- Treat Door as a standalone subsystem.
- Do not import modelling logic, features, models, or configuration from other PS3 subsystem implementations.
- Segment before classifying.
- Evaluate the complete held-out stream interval, not only pre-cut cycles.
- Use the official IoU-weighted F1 implementation as the main local metric.
- Prefer simple measurable changes over architecture rewrites.
- Maintain one canonical `DoorService`/service-layer inference route used by UI, CLI, and optional API.
- Never manually adjust `Test.csv` predictions by looking for hidden answers.
- Record every meaningful threshold/model change in `19_EXPERIMENT_LOG.md`.

## Definition of done for a change

- tests pass;
- app still runs;
- local evaluator runs;
- official CSV contract remains valid;
- experiment notes are updated if behaviour/score changed.
