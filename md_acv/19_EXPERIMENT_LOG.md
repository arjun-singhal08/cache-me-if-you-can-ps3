# ACV Experiment Log

## Experiment ID

`ACV-EXP-XXX`

### Date
YYYY-MM-DD

### Objective
State the hypothesis.

### Validation
Leave-One-Case-Out

### Features
List exact feature families/version.

### Ranking method
Robust peer / temporal residual / supervised / hybrid.

### Hyperparameters
Record exact values.

### Results

| Held-out case | True car | Predicted rank | Official score |
|---|---|---:|---:|
| 01 | 01 | | |
| 02 | 02 | | |
| 03 | 03 | | |
| 04 | 01 | | |
| 05 | 04 | | |
| 06 | 06 | | |

Summary:
- Mean rank:
- Top-1:
- Top-2:
- Top-3:
- Mean official score:

Decision:
- [ ] keep
- [ ] reject
- [ ] investigate

### Reproducibility and evidence

- Organiser source commit and workbook hashes:
- Parser/schema/feature version:
- Dependency versions and random seeds:
- Per-fold preprocessing and inner selection procedure:
- Missing-feature/zero-MAD fallbacks:
- Baseline comparison (random expected score 0.5625 for eight cars):
- All six predicted rankings and explanation evidence:
- Number of variants tried and model-selection limitations:

The true-car entries above must be verified against the actual Train_Labels.csv before use; never infer labels from filenames.
