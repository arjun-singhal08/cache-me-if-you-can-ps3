# Door Configuration and Environment

## Python

Use the team's chosen current stable Python version only after confirming library compatibility. Consistency across teammates matters more than using a prerelease/interpreter unsupported by dependencies.

## Suggested dependencies

- pandas
- numpy
- scipy
- scikit-learn
- joblib
- streamlit
- plotly
- pyyaml
- fastapi (optional)
- uvicorn (optional)

## YAML

Suggested:

`config/door.yaml`

```yaml
app:
  title: Door Condition Monitoring
  max_upload_mb: 50
  plot_max_points: 12000

segmentation:
  mode: hybrid_state_position
  boundary_refinement: true
  derive_sampling_interval: true

classification:
  random_seed: 42
  model: extra_trees
  use_operation_feature: true

validation:
  blocked_folds: 5
  metric: iou_weighted_f1
```

Values that control cycle-duration ranges, boundary windows, or thresholds must be learned/justified from training data and recorded in experiments.

## Secrets

The core Door solution requires no API keys. Keep optional service keys in environment variables, never YAML or Git.
