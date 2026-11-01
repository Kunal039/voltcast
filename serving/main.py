"""FastAPI model-serving service for household electricity consumption forecasts."""
import time
from datetime import datetime
from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field

from src.predict import MIN_HISTORY_HOURS, forecast, load_model
from src.utils import ROOT, load_params

app = FastAPI(
    title="Household Electricity Consumption Prediction API",
    description="Forecasts hourly household active power (kW) from recent smart-meter readings.",
    version="1.0.0",
)
STATS = {"requests": 0, "predictions": 0, "total_latency_ms": 0.0}


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
    history: list[Reading] = Field(description=f"At least {MIN_HISTORY_HOURS} consecutive hourly readings")
    horizon: int = Field(default=1, ge=1, le=48, description="Hours ahead to forecast")


class ForecastPoint(BaseModel):
    timestamp: datetime
    predicted_kw: float
    lower_kw: float
    upper_kw: float


class ForecastResponse(BaseModel):
    model: str
    horizon: int
    forecast: list[ForecastPoint]
    predicted_energy_kwh: float


@app.middleware("http")
async def track_latency(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    STATS["requests"] += 1
    STATS["total_latency_ms"] += (time.perf_counter() - t0) * 1000
    return response


@app.get("/health")
def health():
    _, meta = load_model()
    return {"status": "ok", "model": meta["model_name"], "trained_at": meta["trained_at"]}


@app.get("/model-info")
def model_info():
    _, meta = load_model()
    return {k: meta[k] for k in ["model_name", "trained_at", "mlflow_run_id", "train_period",
                                  "test_period", "test_metrics", "baseline_test_metrics"]} | {
        "n_features": len(meta["features"])}


@app.get("/metrics")
def service_metrics():
    n = max(STATS["requests"], 1)
    return {**STATS, "avg_latency_ms": round(STATS["total_latency_ms"] / n, 2)}


@app.post("/predict", response_model=ForecastResponse)
def predict(req: ForecastRequest):
    if len(req.history) < MIN_HISTORY_HOURS:
        raise HTTPException(422, f"history must contain at least {MIN_HISTORY_HOURS} hourly readings")
    hist = pd.DataFrame([r.model_dump() for r in req.history]).set_index("timestamp").sort_index()
    hist.index = pd.DatetimeIndex(hist.index).tz_localize(None)
    if hist.index.has_duplicates:
        raise HTTPException(422, "history contains duplicate timestamps")
    hist = hist.asfreq("1h").interpolate(limit_direction="both")
    hist["Sub_metering_other"] = (hist["Global_active_power"] * 1000
                                  - hist[["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]].sum(axis=1)).clip(lower=0)
    fc = forecast(hist, req.horizon)
    STATS["predictions"] += len(fc)
    _, meta = load_model()
    return ForecastResponse(
        model=meta["model_name"], horizon=req.horizon,
        forecast=fc.to_dict(orient="records"),
        predicted_energy_kwh=round(float(fc["predicted_kw"].sum()), 3),
    )


@lru_cache(maxsize=1)
def serving_history() -> pd.DataFrame:
    """Recent hourly readings shipped with the service (test period + one week of context)."""
    params = load_params()
    return pd.read_csv(ROOT / params["data"]["serving"], index_col="datetime", parse_dates=True)


@app.get("/sample-request")
def sample_request(hours: int = MIN_HISTORY_HOURS):
    """Returns a ready-to-use request body built from the last hours of the dataset."""
    hourly = serving_history().iloc[-hours:]
    cols = list(Reading.model_fields)[1:]
    rows = hourly[cols].round(3).reset_index().rename(columns={"datetime": "timestamp"})
    rows["timestamp"] = rows["timestamp"].astype(str)
    return {"history": rows.to_dict(orient="records"), "horizon": 24}


@app.get("/forecast")
def forecast_at(origin: datetime, horizon: int = Query(24, ge=1, le=48)):
    """Forecast from a point in the stored test period and compare with what actually happened."""
    hourly = serving_history()
    origin = pd.Timestamp(origin).tz_localize(None).floor("h")
    history = hourly[hourly.index < origin]
    if len(history) < MIN_HISTORY_HOURS:
        raise HTTPException(422, f"origin must be at least {MIN_HISTORY_HOURS} h after {hourly.index[0]}")
    fc = forecast(history.iloc[-MIN_HISTORY_HOURS * 2:], horizon)
    actual = hourly["Global_active_power"].reindex(fc["timestamp"])
    past = hourly["Global_active_power"].loc[origin - pd.Timedelta(hours=72): origin - pd.Timedelta(hours=1)]
    STATS["predictions"] += len(fc)
    return {
        "forecast": [{**r, "timestamp": r["timestamp"].isoformat(), "actual_kw": None if pd.isna(a) else float(a)}
                     for r, a in zip(fc.to_dict(orient="records"), actual.values)],
        "history": [{"timestamp": t.isoformat(), "kw": float(v)} for t, v in past.items()],
        "predicted_energy_kwh": round(float(fc["predicted_kw"].sum()), 3),
        "range": [str(hourly.index[MIN_HISTORY_HOURS]), str(hourly.index[-1])],
    }
