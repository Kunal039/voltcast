"""Stage 2 - Preprocessing: clean minute-level readings and resample to hourly."""
import json

import numpy as np
import pandas as pd

from src.utils import ROOT, get_logger, load_params, resolve

log = get_logger("preprocess")

NUMERIC = [
    "Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity",
    "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
]


def load_raw(path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", na_values=["?"], low_memory=False)
    df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
    df = df.drop(columns=["Date", "Time"]).set_index("datetime").sort_index()
    df[NUMERIC] = df[NUMERIC].astype("float32")
    return df


def clean_and_resample(df: pd.DataFrame, rule: str, max_gap: int) -> pd.DataFrame:
    # Energy not captured by the three sub-meters (Wh per minute).
    df["Sub_metering_other"] = (
        df["Global_active_power"] * 1000 / 60
        - df["Sub_metering_1"] - df["Sub_metering_2"] - df["Sub_metering_3"]
    ).clip(lower=0)
    agg = {c: "mean" for c in ["Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity"]}
    agg.update({c: "sum" for c in ["Sub_metering_1", "Sub_metering_2", "Sub_metering_3", "Sub_metering_other"]})
    # min_count=1 keeps all-missing hours as NaN instead of 0 for summed columns.
    hourly = df.resample(rule).agg(agg)
    counts = df["Global_active_power"].resample(rule).count()
    for c in ["Sub_metering_1", "Sub_metering_2", "Sub_metering_3", "Sub_metering_other"]:
        hourly.loc[counts == 0, c] = np.nan
    # Fill short outages by time interpolation; longer gaps use same-hour-last-week values.
    hourly = hourly.interpolate(method="time", limit=max_gap, limit_area="inside")
    hourly = hourly.fillna(hourly.shift(168)).ffill().bfill()
    return hourly


def main() -> None:
    params = load_params()
    raw = load_raw(resolve(params["data"]["raw_txt"]))
    missing_pct = float(raw["Global_active_power"].isna().mean() * 100)
    log.info("Loaded %d minute rows, %.2f%% missing", len(raw), missing_pct)
    hourly = clean_and_resample(raw, params["data"]["resample_rule"], params["data"]["max_interp_gap_hours"])
    out = resolve(params["data"]["processed"])
    hourly.to_parquet(out)
    stats = {
        "raw_rows": int(len(raw)),
        "raw_missing_pct": round(missing_pct, 3),
        "hourly_rows": int(len(hourly)),
        "start": str(hourly.index.min()),
        "end": str(hourly.index.max()),
    }
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "data_stats.json").write_text(json.dumps(stats, indent=2))
    log.info("Saved %d hourly rows to %s", len(hourly), out)


if __name__ == "__main__":
    main()
