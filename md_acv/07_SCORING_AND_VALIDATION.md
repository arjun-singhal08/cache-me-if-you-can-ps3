# ACV Scoring and Validation

## Official score

`score = (n - (r - 1)) / n`

For 8 cars:
- 1st: 1.000
- 2nd: 0.875
- 3rd: 0.750
- 4th: 0.625
- 5th: 0.500
- 6th: 0.375
- 7th: 0.250
- 8th: 0.125
- missing: 0

## Local scorer

Implement:

`rank_decay_score(true_car, ranked_cars)`

Unit-test every rank and the missing case.

## Validation strategy

Use **Leave-One-Case-Out** validation.

For each fold:
1. hold out one complete case;
2. learn transformations/model only from the other five;
3. rank all cars in the held-out case;
4. record true-car rank and official score.

Primary development metric:
`mean official score across 6 folds`

## Never split rows randomly

Rows from the same case are temporally correlated and share the same fault label.

## Secondary diagnostics

Track:
- mean/median true-car rank;
- top-1, top-2, top-3 rates;
- per-case ranking;
- variance across folds.

Choose the simplest stable method with the strongest official validation score.

## Baselines and honest model selection

For a uniformly random complete eight-car ranking, expected score is 0.5625 and expected top-1 accuracy is 12.5%. Report these alongside a deterministic label-independent baseline. A score above 0.5 alone is not evidence of useful localisation.

Six cases are six independent evaluation units, not 48 independent car samples. Record all six fold rankings rather than only an average. Avoid confident generalisation claims from six outcomes.

If choosing hyperparameters, features or hybrid weights, use only the five outer-training cases (for example inner case-held-out validation), or pre-specify a simple baseline. Selecting the best variant on all six outer outcomes and quoting that same score introduces selection optimism; disclose exploratory tuning.

Validate the ranking before scoring: exact discovered IDs, each once, no invented or missing cars. Use the expected case car count as n after validating the permutation. The missing-true-car test verifies the official zero-credit rule; it does not make an incomplete submission valid.

Current-case peer transformations require no labels and are allowed during inference. They are distinct from population transformations fitted across cases, which must be trained inside folds.
