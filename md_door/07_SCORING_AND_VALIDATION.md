# Door Scoring and Validation

## Official metric: IoU-weighted F1

For a true segment and predicted segment with the same label:

```text
intersection = max(0, min(true_end, pred_end) - max(true_start, pred_start))
union = true_duration + pred_duration - intersection
IoU = intersection / union if union > 0 else 0
```

Pairs require IoU > 0 and identical labels.

Matching is one-to-one, greedily assigning the highest-IoU valid pairs first.

Then:

```text
soft_recall = sum(matched IoU) / number_of_true_segments
soft_precision = sum(matched IoU) / number_of_predicted_segments
score = harmonic_mean(soft_recall, soft_precision)
```

The official formula defines the final score as 0 when soft precision and soft recall are both 0. Guard division by zero explicitly: use soft precision = 0 when there are no predictions. For local intervals with no true segments, use soft recall = 0 and report the empty interval separately; do not claim this local edge-case convention is a separately specified official empty-ground-truth rule.

## Local validator requirement

Implement the official formula locally before serious tuning. The validator should return:

- final IoU-weighted F1;
- soft precision;
- soft recall;
- matched segment count;
- missed true segments;
- unmatched predictions;
- mean/median matched IoU;
- confusion counts by label for matched/nearby cycles.

## Validation split

Because the labelled examples come from a single continuous stream, avoid relying only on a random segment split.

Preferred validation:

- create contiguous time blocks;
- train on some blocks and validate on held-out blocks;
- ensure abnormal examples exist in each validation fold where possible.

A secondary stratified segment split may be used for classifier debugging, but the final reported local score should run the **full segmentation + classification pipeline** on held-out continuous intervals.

## Tune in this order

1. segment recall/precision;
2. boundary IoU;
3. classification errors;
4. complete official score.

Never report classifier accuracy alone as the Door system result.
