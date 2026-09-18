# Door Docker Deployment

Docker comes after the local pipeline works.

## Minimal container goal

The container should run the application without downloading organiser data or retraining models.

Include:

- application code;
- Door model artefacts;
- configuration defaults;
- requirements.

Do not include raw organiser training/test datasets in the final image unless specifically needed and permitted.

## Example commands

```bash
docker build -t door-condition-monitor .
docker run --rm -p 8501:8501 door-condition-monitor
```

Open `http://localhost:8501`.

## Container requirements

- Streamlit binds to `0.0.0.0`.
- Prediction output is downloadable through the browser.
- The model path is deterministic.
- No external API/service should be required for the core prediction flow.
