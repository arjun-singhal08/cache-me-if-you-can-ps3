# Optional FastAPI Contract

## Status

**Optional.** Build this only after the Streamlit/model pipeline works, or if the team chooses Next.js as the frontend.

## Server

Framework: FastAPI  
ASGI server: Uvicorn

## Endpoints

### `GET /health`

Response:

```json
{"status":"ok","model_loaded":true,"model_version":"..."}
```

### `POST /api/v1/shm/profile`

Multipart upload: one CSV.

Returns a compact schema/data-quality profile:

```json
{
  "file_id":"sample.csv",
  "rows":1000,
  "columns":4,
  "numeric_columns":["..."],
  "warnings":[],
  "status":"ready"
}
```

### `POST /api/v1/shm/predict`

Multipart upload: one CSV.

Response:

```json
{
  "file_id":"sample.csv",
  "prediction":0.123456,
  "model_version":"...",
  "warnings":[]
}
```

### `POST /api/v1/shm/predict-csv`

Same input, returns downloadable CSV with:

```csv
file_id,prediction
sample.csv,0.123456
```

## API rules

- use the exact same service function as Streamlit/batch inference;
- set upload-size limits;
- validate content before model execution;
- never expose stack traces to normal clients;
- use structured error responses;
- do not persist uploaded files unless explicitly required;
- never return hidden/internal model paths or secrets.
