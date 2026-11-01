# ⚡ Household Electricity Consumption Prediction System

An end-to-end **MLOps** project that forecasts a household's hourly electricity consumption (kW) for the next 1–48 hours.

- **Dataset:** [UCI Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) — 2,075,259 one-minute readings, Dec 2006 – Nov 2010 (downloaded automatically).
- **Best model:** Histogram Gradient Boosting — test MAE **0.293 kW**, RMSE **0.432 kW**, R² **0.623** (22% lower MAE than the persistence baseline).
- **Web app:** React (Vite) + Plotly, calling the FastAPI model service
- **Live demo:** https://household-power-forecast.vercel.app (web dashboard; API docs at [/api/docs](https://household-power-forecast.vercel.app/api/docs), auto-deployed from this repo by Vercel)
- **MLOps stack:** config-driven pipeline (`params.yaml`, `dvc.yaml`), MLflow tracking + model registry, FastAPI + Docker serving, Streamlit dashboard, GitHub Actions CI/CT, PSI drift monitoring, pytest.

## Project structure

```
configs/params.yaml        all settings (paths, split dates, features, hyper-parameters)
src/data_ingestion.py      download + schema validation
src/preprocess.py          cleaning, hourly resampling, gap filling
src/features.py            31 leakage-free lag / rolling / calendar / sub-meter features
src/train.py               5 models compared, MLflow logging, model registry
src/monitor.py             PSI data drift + error degradation -> retraining decision
src/evaluate.py            EDA + diagnostic figures, permutation importance
src/predict.py             recursive multi-step forecasting with 95% intervals
serving/main.py            FastAPI service (/predict, /forecast, /health, /model-info, /metrics, /sample-request)
app.py                     web entrypoint: API under /api, React site (built into public/) at /
web/                       React + Vite web dashboard (deployed on Vercel)
src/export_web.py          exports serving data + dashboard JSON for the web app
dashboard/app.py           Streamlit dashboard (5 tabs, local / Docker)
tests/test_pipeline.py     unit, leakage, API and quality-gate tests
run_pipeline.py            runs every stage end to end
dvc.yaml                   same pipeline as a DVC DAG (`dvc repro`)
Dockerfile, docker-compose.yml, .github/workflows/ci.yml
report/                    report generator + final report (.docx / .pdf)
reports/                   metrics, monitoring output, figures, screenshots
```

## Quick start

```bash
pip install -r requirements-dev.txt
python run_pipeline.py          # ingest -> preprocess -> features -> train -> monitor -> evaluate -> export (~1 min)
pytest -q                       # 8 tests
```

Then, in separate terminals:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db        # http://localhost:5000
cd web && npm install && npm run build && cd ..          # build the React site into public/
uvicorn app:app --port 8000                              # web dashboard http://localhost:8000, API docs /api/docs
streamlit run dashboard/app.py                           # http://localhost:8501
```

Or everything at once with Docker: `docker compose up --build`.

### Calling the API

```bash
curl "localhost:8000/api/forecast?origin=2010-11-20T08:00&horizon=6"
curl -s localhost:8000/api/sample-request > req.json   # one week of real readings, horizon 24
curl -s -X POST localhost:8000/api/predict -H "Content-Type: application/json" -d @req.json
```

## Results

| Model | Val MAE | Val RMSE | Val R² |
|---|---|---|---|
| Persistence baseline | 0.441 | 0.669 | 0.364 |
| Seasonal naive (24 h) | 0.636 | 0.912 | -0.185 |
| Ridge regression | 0.398 | 0.557 | 0.559 |
| Random Forest | 0.345 | 0.509 | 0.632 |
| **Hist Gradient Boosting** | **0.342** | **0.498** | **0.647** |

The monitoring stage flags seasonal drift (PSI 0.73 on the water-heater/AC sub-meter between the winter-heavy training data and the summer test period) and recommends retraining.

## Deployment

The Vercel project is connected to this GitHub repository: every push to `main` redeploys automatically.
Vercel builds the React app (`vercel.json`: `cd web && npm ci && npm run build`, output `public/`), installs only
`requirements.txt` (the slim serving dependencies), runs `app.py` as a Python function and serves `public/` from its CDN. `.vercelignore` keeps raw data, training code and reports out of the bundle.

## Regenerating the report

```bash
python report/make_diagrams.py
python report/capture_screenshots.py "<path to msedge.exe or chrome.exe>"   # needs the dashboard running
python report/build_report.py
```
