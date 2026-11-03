# ⚡ VoltCast

**Household Electricity Consumption Prediction System** · MLOps-1 (25CSH-320) mini project, Chandigarh University

VoltCast forecasts a household's hourly electricity consumption (kW) for the next 1–48 hours and raises an alert
for peak-consumption hours. It is built as a complete MLOps workflow: versioned data and pipeline, experiment
tracking, tuned models, a Flask REST API in Docker, CI/CD, and monitoring with Prometheus and Grafana.

- **Live demo:** https://voltcast-forecast.vercel.app (React web app + Flask API, auto-deployed from this repo by Vercel)
- **Dataset:** [UCI Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption): 2,075,259 one-minute readings from one house near Paris, Dec 2006 – Nov 2010. Stored as `data/household_power_consumption.csv` (downloaded automatically, versioned with DVC).
- **Forecaster:** Random Forest (a syllabus algorithm), test MAE **0.295 kW**, RMSE **0.434 kW**, R² **0.620**, accuracy (1 − WAPE) **67.9%** (22% lower MAE than the persistence baseline)
- **Peak-hour classifier:** decision tree tuned with GridSearchCV, test accuracy **80.3%**, recall **77.9%**, ROC-AUC **0.873**

## Syllabus coverage (25CSH-320)

| Syllabus topic / experiment | Where it is done |
|---|---|
| Exp 1 · Data preprocessing: cleaning, missing values, outliers, normalisation, train/val/test split | `src/data_ingestion.py`, `src/preprocess.py` (gap filling, IQR outlier report), `StandardScaler` for KNN/SVM, chronological split in `src/train.py` |
| Exp 2 · Linear regression evaluated with MSE | `linear_regression` in `src/train.py`; MSE, RMSE, MAE, R² in `reports/metrics.json` |
| Unit 1–2 algorithms: linear regression, decision tree, KNN, SVM, random forest | all trained and compared in `src/train.py`; only these may be deployed (`selection.deployable` in `params.yaml`), so Random Forest is deployed. Gradient boosting and two baselines are kept as reference points |
| Exp 3 · Decision tree classifier: accuracy, confusion matrix, precision–recall | `src/classify.py` (peak-hour classifier), `reports/figures/fig_classification.png` |
| Exp 4 · GridSearchCV tuning, tuned vs default | `tune()` in `src/train.py` and `src/classify.py`, time-series cross-validation, `fig_tuning.png` |
| Overfitting, underfitting, regularisation | train vs validation error (`fig_train_vs_val.png`), depth validation curve (`fig_validation_curve.png`), ridge `alpha` search |
| Exp 5 · Version-controlled repository, ML in a DevOps pipeline | Git + GitHub, `dvc.yaml` pipeline |
| Exp 6 · Docker | `Dockerfile` (multi-stage: React build + Python), `docker-compose.yml` |
| Exp 7 · Flask REST API + Prometheus metrics | `serving/main.py`, `app.py`, `/api/metrics` |
| Exp 8 · CI/CD with GitHub Actions | `.github/workflows/ci.yml` (dvc repro, quality gates, tests, Docker build); Vercel deploys every push |
| Exp 9 · DVC versioning of data, models and experiments | `dvc.yaml`, `dvc.lock`, `dvc metrics show`; MLflow model registry |
| Exp 10 · Monitoring and logging with Prometheus and Grafana | `monitoring/` (Prometheus scrape config, provisioned Grafana dashboard), `logs/predictions.jsonl`, PSI drift in `src/monitor.py` |
| Cloud and on-premise deployment, microservices, Kubernetes | Vercel (cloud), Docker Compose (on-premise, 5 services), `k8s/deployment.yaml` |
| Ethics: fairness, bias, transparency, privacy, security | `MODEL_CARD.md`, bias slice analysis (`reports/fairness.json`, `fig_fairness.png`), input validation and limits in the API |

## Project structure

```
configs/params.yaml        all settings: paths, split dates, features, model grids, thresholds
src/data_ingestion.py      download the UCI archive, convert to CSV, validate schema
src/preprocess.py          parse, fill gaps, resample hourly, IQR outlier report
src/features.py            31 leakage-free lag / rolling / calendar / sub-meter features
src/train.py               9 regressors, GridSearchCV tuning, MLflow tracking and registry
src/classify.py            peak-hour classifier (decision tree, GridSearchCV), classification metrics
src/monitor.py             PSI data drift + error degradation -> retraining decision
src/evaluate.py            EDA, model diagnostics, validation curve, classification and bias-slice figures
src/analysis.py            every model on the test set, accuracy-style metrics, horizon/learning curves, CV folds
src/predict.py             recursive multi-step forecasting with 95% intervals and peak alerts
src/export_web.py          serving data + dashboard JSON for the web app
serving/main.py            Flask REST API with Pydantic validation, Prometheus metrics, prediction log
app.py                     web entrypoint: API under /api, React site at /
web/                       React + Vite web dashboard (deployed on Vercel)
dashboard/app.py           Streamlit dashboard (local / Docker)
monitoring/                Prometheus config and provisioned Grafana dashboard
k8s/deployment.yaml        Kubernetes Deployment, Service and autoscaler
tests/test_pipeline.py     unit, leakage, classification, API and Prometheus tests
dvc.yaml, dvc.lock         reproducible pipeline and the exact versions of every output
MODEL_CARD.md              intended use, performance, fairness, privacy, security
scripts/                   show_metrics.py (compact `dvc metrics show`), list_registry.py (MLflow registry)
report/                    project report (.docx / .pdf, 134 pages) and its generators
reports/                   metrics, monitoring, figures, analysis/, terminal/ (captured command output), screenshots
```

## Quick start

```bash
pip install -r requirements-dev.txt
dvc repro                      # or: python run_pipeline.py   (about 4 minutes, includes GridSearchCV)
dvc metrics show --json | python scripts/show_metrics.py
pytest -q                      # 11 tests
```

Run the web app and API locally:

```bash
cd web && npm install && npm run build && cd ..
python app.py                  # http://localhost:8000  (API under /api)
```

Run the full stack in Docker: API + web app, Streamlit, MLflow, Prometheus and Grafana:

```bash
docker compose up --build
```

| Service | URL |
|---|---|
| Web app + Flask API | http://localhost:8000 |
| Streamlit dashboard | http://localhost:8501 |
| MLflow UI | http://localhost:5000 |
| Prometheus | http://localhost:9090 |
| Grafana (dashboard "VoltCast - Model Monitoring") | http://localhost:3000 |

### REST API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | liveness and loaded model |
| GET | `/api/model-info` | model, hyper-parameters, training window, test metrics |
| GET | `/api/forecast?origin=2010-11-20T08:00&horizon=6` | forecast from a point in the test period, with actual values |
| POST | `/api/predict` | forecast from your own 169+ hourly readings |
| GET | `/api/sample-request` | ready-made body for `/api/predict` |
| GET | `/api/metrics` | Prometheus metrics |

```bash
curl -s localhost:8000/api/sample-request > req.json
curl -s -X POST localhost:8000/api/predict -H "Content-Type: application/json" -d @req.json
```

## Results

Validation period Jan–May 2010 (default hyper-parameters):

| Model | Train MAE | Val MAE | Val RMSE | Val R² |
|---|---|---|---|---|
| Persistence baseline | 0.449 | 0.441 | 0.669 | 0.364 |
| Seasonal naive (24 h) | 0.663 | 0.636 | 0.912 | -0.184 |
| Linear regression | 0.409 | 0.398 | 0.557 | 0.559 |
| Ridge regression | 0.409 | 0.398 | 0.557 | 0.559 |
| Decision tree | 0.000 | 0.506 | 0.776 | 0.142 |
| K-nearest neighbours | 0.347 | 0.395 | 0.566 | 0.544 |
| SVM (SVR) | 0.296 | 0.355 | 0.531 | 0.598 |
| **Random forest (deployed)** | 0.220 | **0.345** | **0.509** | **0.632** |
| Gradient boosting (reference, not deployable) | 0.289 | 0.341 | 0.498 | 0.647 |

The default decision tree over-fits (zero training error). GridSearchCV picks `max_depth=10, min_samples_leaf=100`,
which cuts its validation MAE from 0.506 to 0.380. The default Random Forest is the best syllabus algorithm and is deployed;
the gradient-boosting reference is better by only 0.001 kW test MAE.

Monitoring flags seasonal drift (PSI 0.73 on the water-heater/AC sub-meter between the winter-heavy training data
and the summer test period) and recommends retraining. Bias analysis shows the largest errors in evening and
high-use hours; see `MODEL_CARD.md`.

## Deployment

Vercel is connected to this repository: every push to `main` redeploys. It builds the React app (`vercel.json`),
installs only `requirements.txt` (the slim serving dependencies) and runs the Flask app in `app.py` as a Python function.
`.vercelignore` keeps the dataset, training code and reports out of the bundle.

## Regenerating the report

```bash
python report/make_diagrams.py && python report/make_diagrams_more.py
python report/capture_terminal.py      # re-runs the key commands and saves their output (Docker Desktop running)
python report/build_report.py
```
