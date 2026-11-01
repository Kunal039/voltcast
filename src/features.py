"""Stage 3 - Feature engineering for next-hour consumption forecasting.

Every feature at time t uses only information available up to hour t-1,
so the model forecasts consumption for hour t without leakage.
"""
import numpy as np
import pandas as pd

from src.utils import get_logger, load_params, resolve

log = get_logger("features")

# Public holidays in France (dataset location: Sceaux, near Paris), fixed-date ones.
FIXED_HOLIDAYS = {(1, 1), (5, 1), (5, 8), (7, 14), (8, 15), (11, 1), (11, 11), (12, 25)}


def calendar_features(idx: pd.DatetimeIndex) -> pd.DataFrame:
    f = pd.DataFrame(index=idx)
    f["hour"] = idx.hour
    f["dayofweek"] = idx.dayofweek
    f["month"] = idx.month
    f["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    f["is_holiday"] = [int((d.month, d.day) in FIXED_HOLIDAYS) for d in idx]
    # Cyclical encodings so 23:00 and 00:00 are close in feature space.
    f["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    f["month_sin"] = np.sin(2 * np.pi * (idx.month - 1) / 12)
    f["month_cos"] = np.cos(2 * np.pi * (idx.month - 1) / 12)
    return f


def build_features(hourly: pd.DataFrame, target: str, lags: list[int], windows: list[int]) -> pd.DataFrame:
    y = hourly[target]
    f = calendar_features(hourly.index)
    for lag in lags:
        f[f"lag_{lag}"] = y.shift(lag)
    past = y.shift(1)
    for w in windows:
        f[f"roll_mean_{w}"] = past.rolling(w).mean()
        f[f"roll_std_{w}"] = past.rolling(w).std()
    for c in ["Global_reactive_power", "Voltage", "Global_intensity",
              "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]:
        f[f"{c}_lag1"] = hourly[c].shift(1)
    f["target"] = y
    return f.dropna()


def feature_columns(params: dict) -> list[str]:
    """Deterministic feature list used by training, the API and the dashboard."""
    fp = params["features"]
    cols = ["hour", "dayofweek", "month", "is_weekend", "is_holiday",
            "hour_sin", "hour_cos", "month_sin", "month_cos"]
    cols += [f"lag_{l}" for l in fp["lags"]]
    for w in fp["rolling_windows"]:
        cols += [f"roll_mean_{w}", f"roll_std_{w}"]
    cols += [f"{c}_lag1" for c in ["Global_reactive_power", "Voltage", "Global_intensity",
                                    "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]]
    return cols


def main() -> None:
    params = load_params()
    hourly = pd.read_parquet(resolve(params["data"]["processed"]))
    feats = build_features(hourly, params["features"]["target"],
                           params["features"]["lags"], params["features"]["rolling_windows"])
    assert list(feats.columns[:-1]) == feature_columns(params)
    feats.to_parquet(resolve(params["data"]["features"]))
    log.info("Built %d rows x %d features", len(feats), feats.shape[1] - 1)


if __name__ == "__main__":
    main()
