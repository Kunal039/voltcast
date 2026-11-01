"""Flask REST API serving the VoltCast models, with Prometheus metrics and prediction logging.

Endpoints (mounted under /api by app.py):
  GET  /health          liveness + loaded model
  GET  /model-info      model, training window and test metrics
  POST /predict         forecast from user-supplied hourly readings (JSON)
  GET  /forecast        forecast from a point in the stored test period, with actual values
  GET  /sample-request  ready-made body for POST /predict
  GET  /metrics         Prometheus metrics (scraped by Prometheus, shown in Grafana)
"""
import json
import logging
import os
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import pandas as pd
from flask import Blueprint, Response, g, jsonify, request
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field, ValidationError

from src.predict import MIN_HISTORY_HOURS, forecast, load_classifier, load_model
from src.utils import ROOT, load_params

api = Blueprint("api", __name__)

# ---------- Prometheus metrics ----------
REQUESTS = Counter("voltcast_requests_total", "HTTP requests", ["endpoint", "method", "status"])
LATENCY = Histogram("voltcast_request_latency_seconds", "Request latency", ["endpoint"],
                    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5))
PREDICTIONS = Counter("voltcast_predictions_total", "Hourly values forecast")
PREDICTED_KW = Histogram("voltcast_predicted_kw", "Distribution of forecast consumption (kW)",
                         buckets=(0.25, 0.5, 1, 1.5, 2, 3, 4, 6))
PEAK_ALERTS = Counter("voltcast_peak_alerts_total", "Forecast hours flagged as peak hours")
ERRORS = Counter("voltcast_prediction_errors_total", "Rejected or failed prediction requests", ["reason"])
MODEL_INFO = Gauge("voltcast_model_info", "Loaded model (value is always 1)", ["model", "trained_at"])
TEST_MAE = Gauge("voltcast_model_test_mae_kw", "Test-set MAE of the loaded model")


# ---------- prediction log (one JSON line per request) ----------
def _prediction_logger() -> logging.Logger:
    logger = logging.getLogger("voltcast.predictions")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger.addHandler(logging.StreamHandler())
        # Serverless hosts (Vercel) have a read-only disk: log to stdout only there.
        if not os.environ.get("VERCEL"):
            log_dir = Path(os.environ.get("VOLTCAST_LOG_DIR", ROOT / "logs"))
            log_dir.mkdir(parents=True, exist_ok=True)
            logger.addHandler(logging.FileHandler(log_dir / "predictions.jsonl", encoding="utf-8"))
    return logger


PRED_LOG = _prediction_logger()


def log_predictions(source: str, fc: pd.DataFrame, extra: dict | None = None) -> None:
    PREDICTIONS.inc(len(fc))
    PEAK_ALERTS.inc(int(fc["peak_alert"].sum()))
    for v in fc["predicted_kw"]:
        PREDICTED_KW.observe(v)
    _, meta = load_model()
    PRED_LOG.info(json.dumps({
        "time": datetime.now().isoformat(timespec="seconds"), "source": source, "model": meta["model_name"],
        "horizon": len(fc), "first_hour": str(fc["timestamp"].iloc[0]),
        "predicted_kw": [round(v, 3) for v in fc["predicted_kw"]],
        "peak_hours": int(fc["peak_alert"].sum()), **(extra or {}),
    }))


@api.before_app_request
def _start_timer():
    g.t0 = time.perf_counter()


@api.after_app_request
def _record(response):
    endpoint = request.url_rule.rule if request.url_rule else "unmatched"
    if endpoint.endswith("/metrics"):
        return response
    REQUESTS.labels(endpoint, request.method, response.status_code).inc()
    LATENCY.labels(endpoint).observe(time.perf_counter() - g.get("t0", time.perf_counter()))
    return response


# ---------- input validation ----------
class Reading(BaseModel):
    timestamp: datetime
    Global_active_power: float = Field(ge=0, description="Hourly mean active power, kW")
    Global_reactive_power: float = Field(ge=0)
    Voltage: float = Field(gt=0)
    Global_intensity: float = Field(ge=0)
    Sub_metering_1: float = Field(ge=0, description="Kitchen, Wh in the hour")
    Sub_metering_2: float = Field(ge=0, description="Laundry, Wh in the hour")
    Sub_metering_3: float = Field(ge=0, description="Water heater & AC, Wh in the hour")


class ForecastRequest(BaseModel):
    history: list[Reading]
    horizon: int = Field(default=1, ge=1, le=48)


def error(status: int, message: str, reason: str):
    ERRORS.labels(reason).inc()
    return jsonify({"error": message}), status


def to_records(fc: pd.DataFrame) -> list[dict]:
    # NaN is not valid JSON: missing values (e.g. no classifier probability) become null.
    clean = fc.astype(object).where(fc.notna(), None)
    return [{**r, "timestamp": r["timestamp"].isoformat()} for r in clean.to_dict(orient="records")]


@lru_cache(maxsize=1)
def serving_history() -> pd.DataFrame:
    """Recent hourly readings shipped with the service (test period + two weeks of context)."""
    params = load_params()
    return pd.read_csv(ROOT / params["data"]["serving"], index_col="datetime", parse_dates=True)


# ---------- endpoints ----------
@api.get("/health")
def health():
    _, meta = load_model()
    return jsonify(status="ok", model=meta["model_name"], trained_at=meta["trained_at"])


@api.get("/model-info")
def model_info():
    _, meta = load_model()
    _, peak_kw = load_classifier()
    keys = ["model_name", "hyperparameters", "tuned", "trained_at", "mlflow_run_id", "train_period",
            "test_period", "test_metrics", "baseline_test_metrics"]
    return jsonify({k: meta[k] for k in keys} | {"n_features": len(meta["features"]), "peak_threshold_kw": peak_kw})


@api.post("/predict")
def predict():
    try:
        req = ForecastRequest.model_validate(request.get_json(force=True, silent=False))
    except ValidationError as e:
        return error(422, json.loads(e.json(include_url=False)), "validation")
    except Exception:
        return error(400, "body must be JSON", "bad_json")
    if len(req.history) < MIN_HISTORY_HOURS:
        return error(422, f"history must contain at least {MIN_HISTORY_HOURS} hourly readings", "short_history")
    hist = pd.DataFrame([r.model_dump() for r in req.history]).set_index("timestamp").sort_index()
    hist.index = pd.DatetimeIndex(hist.index).tz_localize(None)
    if hist.index.has_duplicates:
        return error(422, "history contains duplicate timestamps", "duplicates")
    hist = hist.asfreq("1h").interpolate(limit_direction="both")
    hist["Sub_metering_other"] = (hist["Global_active_power"] * 1000
                                  - hist[["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]].sum(axis=1)).clip(lower=0)
    fc = forecast(hist, req.horizon)
    log_predictions("predict", fc)
    _, meta = load_model()
    return jsonify(model=meta["model_name"], horizon=req.horizon, forecast=to_records(fc),
                   predicted_energy_kwh=round(float(fc["predicted_kw"].sum()), 3))


@api.get("/forecast")
def forecast_at():
    """Forecast from a point in the stored test period and compare with what actually happened."""
    try:
        origin = pd.Timestamp(request.args["origin"]).tz_localize(None).floor("h")
        horizon = int(request.args.get("horizon", 24))
    except (KeyError, ValueError):
        return error(422, "query needs origin=YYYY-MM-DDTHH:MM and an integer horizon", "validation")
    if not 1 <= horizon <= 48:
        return error(422, "horizon must be between 1 and 48", "validation")
    hourly = serving_history()
    history = hourly[hourly.index < origin]
    if len(history) < MIN_HISTORY_HOURS:
        return error(422, f"origin must be at least {MIN_HISTORY_HOURS} h after {hourly.index[0]}", "short_history")
    fc = forecast(history.iloc[-MIN_HISTORY_HOURS * 2:], horizon)
    actual = hourly["Global_active_power"].reindex(fc["timestamp"])
    past = hourly["Global_active_power"].loc[origin - pd.Timedelta(hours=72): origin - pd.Timedelta(hours=1)]
    log_predictions("forecast", fc, {"origin": str(origin)})
    return jsonify(
        forecast=[{**r, "actual_kw": None if pd.isna(a) else float(a)} for r, a in zip(to_records(fc), actual.values)],
        history=[{"timestamp": t.isoformat(), "kw": float(v)} for t, v in past.items()],
        predicted_energy_kwh=round(float(fc["predicted_kw"].sum()), 3),
        range=[str(hourly.index[MIN_HISTORY_HOURS]), str(hourly.index[-1])],
    )


@api.get("/sample-request")
def sample_request():
    """Returns a ready-to-use request body built from the last hours of the dataset."""
    hourly = serving_history().iloc[-MIN_HISTORY_HOURS:]
    cols = list(Reading.model_fields)[1:]
    rows = hourly[cols].round(3).reset_index().rename(columns={"datetime": "timestamp"})
    rows["timestamp"] = rows["timestamp"].astype(str)
    return jsonify(history=rows.to_dict(orient="records"), horizon=24)


@api.get("/metrics")
def metrics():
    _, meta = load_model()
    MODEL_INFO.labels(meta["model_name"], meta["trained_at"]).set(1)
    TEST_MAE.set(meta["test_metrics"]["mae"])
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)
