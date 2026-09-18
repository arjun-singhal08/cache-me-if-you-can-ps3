# Door Repository Structure

Target structure inside the team repository:

```text
cache-me-if-you-can-ps3/
  app.py
  requirements.txt
  Dockerfile
  config/
    door.yaml
  md_door/
    ...documentation...
  src/
    door/
      __init__.py
      io.py
      schema.py
      timestamps.py
      segment.py
      features.py
      train.py
      classify.py
      score.py
      explain.py
      service.py
      submission.py
  models/
    door/
      classifier.joblib
      feature_schema.json
      model_metadata.json
  scripts/
    train_door.py
    evaluate_door.py
    predict_door.py
    validate_door_submission.py
  predictions/
    door_predictions.csv
  tests/
    door/
      test_timestamps.py
      test_schema.py
      test_segmentation.py
      test_scoring.py
      test_submission.py
```

## Raw data policy

Do not duplicate organiser raw datasets into Git unless the team intentionally chooses to and repository limits permit it. Prefer a documented local data path pointing to the official dataset clone.

## Isolation rule

Door modules/config/models/tests must be independently runnable and must not import another subsystem's modelling package.
