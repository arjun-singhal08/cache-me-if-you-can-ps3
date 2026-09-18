# Door Local Development

Cloud hosting is not required for initial development.

## Recommended workflow

```bash
python -m venv .venv
```

Activate the environment, then:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open:

`http://localhost:8501`

No Streamlit account is required for localhost use.

## Dataset layout outside the solution repo

Example:

```text
NebulaX/
  cache-me-if-you-can-ps3/
  NebulaX-Hackathon-ProblemStatement/
    PS3/02_Datasets/Door/
      Train.csv
      Train_Segments_Answer.csv
      Test.csv
```

Use YAML/environment configuration for the local dataset path rather than hard-coding a teammate's absolute path.

## Development sequence

1. parse timestamps;
2. visualise stream;
3. implement official scorer;
4. implement baseline segmentation;
5. evaluate boundaries;
6. extract cycle features;
7. train classifier;
8. evaluate complete pipeline;
9. connect Streamlit;
10. export official CSV.
