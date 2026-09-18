# ACV Docker Deployment

Docker comes after local reliability.

## Concept

```dockerfile
FROM python:3-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app_acv.py", "--server.address=0.0.0.0", "--server.port=8501"]
```

Pin a tested Python image before final submission.

Build:
```bash
docker build -t nebula-acv .
```

Run:
```bash
docker run --rm -p 8501:8501 nebula-acv
```

Quality gate:
- upload works;
- `openpyxl` present;
- configs/models included;
- inference does not require organiser training data;
- download works;
- no login required.

## Build context

Add a .dockerignore excluding raw datasets, personal paths, secrets, virtual environments and Git metadata. COPY . . otherwise includes those files in the build context/image. Include frozen ACV artifacts and a pinned environment; do not retrain during container startup.
