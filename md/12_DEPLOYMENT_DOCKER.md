# Deployment and Docker Plan

## Goal

A judge or teammate should be able to launch the app predictably without reproducing a local developer environment.

## Recommended deployment unit

For the MVP, one Docker image containing:

- Python runtime;
- pinned dependencies;
- Streamlit app;
- trained SHM model artifact;
- non-secret YAML configuration.

## Container behaviour

The container should:

1. start the web app automatically;
2. expose one documented port;
3. load the model at startup or first use;
4. avoid writing uploaded raw data permanently by default;
5. provide a health check if practical.

## If FastAPI is added

Prefer either:

- separate UI and API containers under Docker Compose; or
- one API container plus separately deployed Next.js frontend.

Do not add multi-container complexity merely for architecture aesthetics.

## Docker checklist

- use an exact stable Python base version;
- copy dependency file before application code for layer caching;
- run as a non-root user if time permits;
- set a sensible upload limit;
- include `.dockerignore`;
- exclude raw organiser datasets, `.git`, caches, secrets, and local environments;
- verify model file path inside the image;
- test the final image on a second machine if possible.

## Deployment priority

1. local app works;
2. Docker image works;
3. hosted deployment works;
4. optional FastAPI/Next.js split only if needed.
