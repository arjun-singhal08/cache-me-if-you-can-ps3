# ACV Documentation Package

This folder is the development source of truth for the **PS3 ACV refrigerant-leak localisation minitask**.

Official references:
- `PS3/01_Problem_Statement_3_Specifications.md`
- `PS3/03_References/ACV/ACV_Subsystem_Info_Kit.md`
- `PS3/02_Datasets/ACV/`
- `PS3/04_Example_Submission/acv_predictions.csv`

Official repository:
`https://github.com/aochinwen/NebulaX-Hackathon-ProblemStatement`

Team repository:
`https://github.com/arjun-singhal08/cache-me-if-you-can-ps3`

## Scope

The application must accept an ACV `.xlsx` case, discover the cars and available parameters dynamically, help a new engineer understand the telemetry, rank every car from most to least likely to have a refrigerant leak, and export the official `acv_predictions.csv`.

The ranking pipeline must remain deterministic and reproducible. Optional explanation features must never silently change the ranking.

Start with `00_INDEX.md`.
