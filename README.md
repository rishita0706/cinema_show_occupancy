# Cinema Show Occupancy (Patna) - MLOps mini project

| | |
|---|---|
| **Group id** | G215 |
| **Problem id** | MP152 |
| **Course / faculty** | SCSE3040 MLOps, Upasna Joshi (Batch 10) |

## Members

| Enrolment no. | Name |
|---|---|
| S24CSEU0987 | Yash Vardhan Patel |
| S24CSEU0992 | Rishita Sinha |
| S24CSEU0989 | Atharv Krishna |
| S24CSEU0973 | Rajana Bharadwaj |

## What this project does

Predicts `occupancy_pct` (how full a cinema show is) from the show's price, release
age, rating, time, screen size, distance to the metro, language and location.
The metric is MAE. The repo holds the full pipeline: data checks, cleaning, DVC,
MLflow tracking and registry, a FastAPI service, Docker, CI, Kubernetes files,
JSON logging, drift monitoring, retraining and the holdout predictions.

## Folder map

```
src/cinema/config.py       reads params.yaml
src/cinema/data_checks.py  finds the 7 kinds of data problem in train.csv
src/cinema/cleaning.py     fixes them (duplicates, unit x100, 9999 blanks, language spellings)
src/cinema/train.py        v1: 4 MLflow runs, best one registered, alias @champion
src/cinema/monitor.py      PSI, MAE per location, retrain v2, promotion rule
src/cinema/predict.py      holdout predictions with the @champion model
app/main.py                FastAPI: POST /predict, GET /health, JSON logs
tests/                     pytest tests (one group uses TestClient)
params.yaml                paths and settings
data/raw/*.dvc             DVC pointers (the big CSVs are not in git)
models/model.joblib        the champion model the service loads
mlflow.db                  MLflow database (runs + registry)
reports/findings.yaml      the filled form;  reports/*.json = full reports
predictions/holdout_predictions.csv
Dockerfile, .dockerignore, .github/workflows/ci.yml, k8s/*.yaml
```

## How to run (in this order)

```
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
dvc pull                          # only if data/raw/train.csv and later.csv are missing

python -m src.cinema.data_checks  # 1. the seven data checks
python -m src.cinema.cleaning     # 2. cleaning summary
python -m src.cinema.train        # 3. v1 (run once)
python -m src.cinema.monitor      # 4. PSI, per-location MAE, v2, decision (run once)
python -m src.cinema.predict      # 5. holdout predictions
pytest                            # 6. tests
```

Run the API:

```
uvicorn app.main:app --port 8000     # then open http://127.0.0.1:8000/docs
```

Docker and Kubernetes:

```
docker build -t mp152-api:1.0 .
docker run -p 8000:8000 mp152-api:1.0
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml
```

## Client rules enforced by the API (422 if broken)

- `distance_to_metro_km` above 8.5 is rejected.
- `ticket_price_rs` below 50 is rejected.

## Experiment tracking

MLflow with `sqlite:///mlflow.db`. Experiment `cinema_show_occupancy`, model
`cinema_occupancy_model`. Version 1 is the best of four runs. Version 2 is
trained on train.csv plus the first 70% of later.csv (by date); the champion
alias moves to version 2 only if its MAE on the last 30% is at least 10% lower.
Every run logs `data_sha256` of the original `train.csv`.