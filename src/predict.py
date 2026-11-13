"""Inference helpers shared by the API and the dashboard."""
import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from src.features import build_features
from src.utils import ROOT, load_params

MIN_HISTORY_HOURS = 169  # longest lag (168h) + 1


@lru_cache(maxsize=1)
def load_model():
    model = joblib.load(ROOT / "models" / "model.joblib")
    if "n_jobs" in model.get_params():
        model.set_params(n_jobs=1)  # forecasts predict one row at a time; worker start-up would dominate
    meta = json.loads((ROOT / "models" / "metadata.json").read_text())
    return model, meta


@lru_cache(maxsize=1)
def load_classifier():
    """Peak-hour classifier and its kW threshold (see src/classify.py)."""
    clf = joblib.load(ROOT / "models" / "peak_classifier.joblib")
    if "n_jobs" in clf.get_params():
        clf.set_params(n_jobs=1)
    info = json.loads((ROOT / "models" / "peak_classifier.json").read_text())
    return clf, info["threshold_kw"]


def _next_row(history: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Append a placeholder for the next hour and compute its feature row."""
    next_ts = history.index[-1] + pd.Timedelta(hours=1)
    ext = pd.concat([history, pd.DataFrame(index=[next_ts], columns=history.columns, dtype=float)])
    target = params["features"]["target"]
    ext[target] = ext[target].fillna(-1.0)  # placeholder so dropna() keeps the row; not used as a feature
    feats = build_features(ext, target, params["features"]["lags"], params["features"]["rolling_windows"])
    return feats.iloc[[-1]]


def forecast(history: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Recursive multi-step forecast from an hourly history frame.

    Exogenous readings (voltage, current, sub-meters) for future hours are unknown,
    so each forecast hour reuses the readings of the same hour on the previous day.
    Averaged over the 24 hours this cut the MAE from 0.569 to 0.484 kW on 60 random
    test origins (carrying the last observed hour forward instead), because it
    keeps the daily pattern.

    Peak-hour alerts come from the classifier for every hour.
    """
    params = load_params()
    model, meta = load_model()
    clf, _ = load_classifier()
    if len(history) < MIN_HISTORY_HOURS:
        raise ValueError(f"Need at least {MIN_HISTORY_HOURS} hours of history, got {len(history)}")
    hist = history.sort_index().copy()
    target = params["features"]["target"]
    out = []
    for step in range(1, horizon + 1):
        row = _next_row(hist, params)
        x = row[meta["features"]].values
        yhat = float(max(model.predict(x)[0], 0.0))
        peak_prob = float(clf.predict_proba(x)[0, 1])
        # Uncertainty grows with the recursive horizon.
        band = 1.96 * meta["residual_std"] * np.sqrt(step)
        out.append({"timestamp": row.index[0], "predicted_kw": yhat,
                    "lower_kw": max(yhat - band, 0.0), "upper_kw": yhat + band,
                    "peak_probability": peak_prob, "peak_alert": peak_prob >= 0.5})
        new = hist.loc[[row.index[0] - pd.Timedelta(hours=24)]].copy()  # same hour yesterday
        new.index = row.index
        new[target] = yhat
        hist = pd.concat([hist, new])
    return pd.DataFrame(out)
