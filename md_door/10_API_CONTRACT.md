# Door API Contract (Optional)

FastAPI is optional. Do not introduce it until the local Streamlit pipeline works end to end.

Run with Uvicorn if used.

## Suggested endpoints

### `GET /health`

Returns service/model status.

### `POST /door/analyse`

Input: multipart CSV.

Returns:

```json
{
  "stream": {"rows": 0, "start_time": "", "end_time": ""},
  "segments": [
    {
      "start_time": "...",
      "end_time": "...",
      "operation": "Open",
      "prediction": "Normal",
      "confidence": 0.0
    }
  ]
}
```

`operation` and `confidence` are application metadata, not official submission fields.

### `POST /door/predictions.csv`

Returns official-format CSV bytes:

```csv
start_time,end_time,prediction
```

## Contract rule

FastAPI and Streamlit must call the same `src/door/service.py` functions so results cannot diverge.
