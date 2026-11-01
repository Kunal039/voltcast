import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from serving.main import app
from src.features import build_features, feature_columns
from src.monitor import psi
from src.predict import MIN_HISTORY_HOURS
from src.preprocess import clean_and_resample
from src.train import metrics
from src.utils import load_params

PARAMS = load_params()
client = TestClient(app)


def synthetic_hourly(hours=400):
    idx = pd.date_range("2010-01-01", periods=hours, freq="h")
    rng = np.random.default_rng(0)
    gap = 1 + 0.8 * np.sin(2 * np.pi * idx.hour / 24) + rng.normal(0, 0.1, hours)
    return pd.DataFrame({
        "Global_active_power": np.abs(gap), "Global_reactive_power": 0.1, "Voltage": 240.0,
        "Global_intensity": np.abs(gap) * 4.2, "Sub_metering_1": 1.0, "Sub_metering_2": 2.0,
        "Sub_metering_3": 5.0, "Sub_metering_other": 10.0,
    }, index=idx)


def test_resample_fills_gaps():
    idx = pd.date_range("2010-01-01", periods=180, freq="min")
    raw = pd.DataFrame({c: 1.0 for c in ["Global_active_power", "Global_reactive_power", "Voltage",
                                         "Global_intensity", "Sub_metering_1", "Sub_metering_2",
                                         "Sub_metering_3"]}, index=idx)
    raw.iloc[60:120] = np.nan  # one fully missing hour
    hourly = clean_and_resample(raw, "1h", 6)
    assert len(hourly) == 3
    assert not hourly.isna().any().any()


def test_features_have_no_leakage():
    h = synthetic_hourly()
    f = build_features(h, "Global_active_power", PARAMS["features"]["lags"], PARAMS["features"]["rolling_windows"])
    assert list(f.columns[:-1]) == feature_columns(PARAMS)
    # lag_1 at time t must equal the target at t-1, never the target at t.
    t = f.index[10]
    assert f.loc[t, "lag_1"] == pytest.approx(h.loc[t - pd.Timedelta(hours=1), "Global_active_power"])
    assert f.loc[t, "roll_mean_3"] == pytest.approx(
        h["Global_active_power"].loc[t - pd.Timedelta(hours=3): t - pd.Timedelta(hours=1)].mean())


def test_metrics_perfect_prediction():
    m = metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert m["mae"] == 0 and m["rmse"] == 0 and m["r2"] == 1


def test_psi_detects_shift():
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 1, 5000)
    assert psi(ref, rng.normal(0, 1, 5000)) < 0.1
    assert psi(ref, rng.normal(1.5, 1, 5000)) > 0.2


def test_health_endpoint():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_endpoint_returns_horizon():
    body = client.get("/sample-request").json()
    body["horizon"] = 6
    r = client.post("/predict", json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    assert len(out["forecast"]) == 6
    assert all(p["lower_kw"] <= p["predicted_kw"] <= p["upper_kw"] for p in out["forecast"])


def test_predict_rejects_short_history():
    body = client.get("/sample-request").json()
    body["history"] = body["history"][: MIN_HISTORY_HOURS - 10]
    assert client.post("/predict", json=body).status_code == 422


def test_model_beats_baseline():
    import json
    from src.utils import ROOT
    meta = json.loads((ROOT / "models" / "metadata.json").read_text())
    assert meta["test_metrics"]["mae"] < meta["baseline_test_metrics"]["mae"]
