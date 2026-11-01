"""Stage 7 - Export: lightweight artifacts for the hosted web app (React site in web/).

Writes the recent hourly history the API needs (CSV, so serving does not need
pyarrow) and a JSON bundle of precomputed charts for the static dashboard page.
"""
import json
import shutil

import pandas as pd

from src.predict import MIN_HISTORY_HOURS
from src.utils import ROOT, get_logger, load_params, resolve

log = get_logger("export_web")


def main() -> None:
    params = load_params()
    hourly = pd.read_parquet(resolve(params["data"]["processed"]))
    meta = json.loads((ROOT / "models" / "metadata.json").read_text())

    test_start = pd.Timestamp(meta["test_period"][0])
    recent = hourly[hourly.index >= test_start - pd.Timedelta(hours=MIN_HISTORY_HOURS * 2)]
    recent.round(4).to_csv(resolve(params["data"]["serving"]), index_label="datetime")

    y = hourly["Global_active_power"]
    daily = y.resample("D").mean().round(3)
    heat = y.groupby([y.index.dayofweek, y.index.hour]).mean().unstack().round(3)
    profile = y.groupby([y.index.dayofweek >= 5, y.index.hour]).mean().unstack(0).round(3)
    sub = (hourly[["Sub_metering_1", "Sub_metering_2", "Sub_metering_3", "Sub_metering_other"]].sum() / 1000).round(0)
    preds = pd.read_parquet(ROOT / "reports" / "test_predictions.parquet").iloc[-24 * 14:].round(3)
    imp = pd.read_csv(ROOT / "reports" / "feature_importance.csv", index_col=0)["importance"].sort_values()
    bundle = {
        "stats": json.loads((ROOT / "reports" / "data_stats.json").read_text()),
        "metrics": json.loads((ROOT / "reports" / "metrics.json").read_text()),
        "monitoring": json.loads((ROOT / "reports" / "monitoring.json").read_text()),
        "model": {k: meta[k] for k in ["model_name", "trained_at", "test_period", "test_metrics",
                                        "baseline_test_metrics"]} | {"n_features": len(meta["features"])},
        "daily": {"x": [d.strftime("%Y-%m-%d") for d in daily.index], "y": daily.tolist()},
        "heatmap": heat.values.tolist(),
        "profile": {"weekday": profile[False].tolist(), "weekend": profile[True].tolist()},
        "submeter": dict(zip(["Kitchen", "Laundry", "Water heater / AC", "Other"], sub.tolist())),
        "test_window": {"x": [t.isoformat() for t in preds.index], "actual": preds["actual"].tolist(),
                        "predicted": preds["predicted"].tolist()},
        "importance": {"names": imp.index.tolist(), "values": imp.round(4).tolist()},
        "forecast_range": [str(recent.index[MIN_HISTORY_HOURS]), str(recent.index[-1])],
        "classification": json.loads((ROOT / "reports" / "classification.json").read_text()),
        "fairness": json.loads((ROOT / "reports" / "fairness.json").read_text()),
    }
    out = resolve(params["data"]["web_data"])
    out.write_text(json.dumps(bundle))
    img = ROOT / "web" / "public" / "img"
    img.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "report" / "diagrams" / "fig_architecture.png", img / "architecture.png")
    log.info("Wrote %d serving rows and dashboard bundle (%.0f KB)", len(recent), out.stat().st_size / 1024)


if __name__ == "__main__":
    main()
