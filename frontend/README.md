# SHM Fatigue Monitor - HTML/CSS/JS Frontend

Frontend for LTA NebulaX PS3 - Team Cache Me If You Can.

## Run locally
Because the frontend uses `fetch()`, serve the folder over HTTP rather than double-clicking `index.html`.

Examples:
- VS Code: install/use Live Server and open `index.html`.
- Any static host: GitHub Pages, Netlify, Vercel, etc.

## Connect the Python backend
The frontend is configured in `app.js` with:

```js
const API_BASE_URL = "http://127.0.0.1:8000";
```

The current frontend expects:

### POST /predict
Request: `multipart/form-data` with the CSV under field name `file`.

Accepted response shapes include:

```json
{ "prediction": 0.37 }
```

or

```json
{
  "prediction": 0.37,
  "explanation": "Optional explanation from the backend/LLM.",
  "recommendation": "Optional recommended action."
}
```

`damage` or `D` can be used instead of `prediction`; the frontend accepts all three.

## CORS
If frontend and Python server are on different origins/ports, the Python server must allow CORS for the frontend origin.

## Batch behavior
The frontend currently calls `/predict` once per uploaded file. This means the Python teammate does NOT need a separate batch API. The `API.batch` placeholder is reserved for a future optimized batch endpoint.

## Important
The browser computes only preview statistics and chart data. The actual fatigue prediction is obtained from the Python backend.
