# Optional ACV API Contract

FastAPI is optional.

## Health

`GET /health`

## Inspect

`POST /acv/inspect`

Returns filename, detected cars, row count, parameters and warnings.

## Rank

`POST /acv/rank`

Example:

```json
{
  "file_id": "acv_test_case.xlsx",
  "ranking": ["03","01","05","02","04","06","07","08"],
  "scores": {"03": 2.41, "01": 1.77}
}
```

## Prediction CSV

`POST /acv/prediction-csv`

Returns:

```csv
file_id,ranked_cars
acv_test_case.xlsx,03|01|05|02|04|06|07|08
```

## Guardrails

- validate workbook;
- preserve filename;
- preserve leading zeros;
- never silently omit a car;
- deterministic ordering.

## Response completeness

The scores object above is abbreviated for illustration. A real successful response must include finite scores for every ranked car or explicitly omit scores entirely. Return schema/data-quality errors instead of a partial success. All endpoints must call the same ACV service used by Streamlit.
