"""Stage 6 - Monitoring: data drift (PSI) and performance checks that decide on retraining."""
import json

import numpy as np
import pandas as pd

from src.train import metrics, split
from src.utils import ROOT, get_logger, load_params, resolve

log = get_logger("monitor")

MONITORED = ["lag_1", "roll_mean_24", "Voltage_lag1", "Global_intensity_lag1",
             "Sub_metering_3_lag1", "hour", "target"]


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index: <0.1 stable, 0.1-0.2 moderate, >0.2 significant drift."""
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref = np.histogram(reference, edges)[0] / len(reference)
    cur = np.histogram(current, edges)[0] / len(current)
    ref, cur = np.clip(ref, 1e-6, None), np.clip(cur, 1e-6, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def main() -> None:
    params = load_params()
    df = pd.read_parquet(resolve(params["data"]["features"]))
    train, _, test = split(df, params)
    drift = {c: round(psi(train[c].values, test[c].values), 4) for c in MONITORED}
    drifted = [c for c, v in drift.items() if v > params["monitoring"]["psi_threshold"]]

    meta = json.loads((ROOT / "models" / "metadata.json").read_text())
    preds = pd.read_parquet(ROOT / "reports" / "test_predictions.parquet")
    # Compare recent production-like window (last 30 days) with the overall test MAE.
    recent = preds.iloc[-24 * 30:]
    recent_mae = metrics(recent["actual"], recent["predicted"])["mae"]
    degradation = (recent_mae / meta["test_metrics"]["mae"] - 1) * 100
    retrain = bool(drifted) or degradation > params["monitoring"]["mae_degradation_pct"]

    report = {
        "psi": drift,
        "drifted_features": drifted,
        "recent_30d_mae": round(recent_mae, 4),
        "reference_test_mae": round(meta["test_metrics"]["mae"], 4),
        "mae_degradation_pct": round(degradation, 2),
        "retraining_recommended": retrain,
    }
    (ROOT / "reports" / "monitoring.json").write_text(json.dumps(report, indent=2))
    log.info("PSI: %s", drift)
    log.info("Recent MAE %.4f (%.1f%% vs reference) -> retrain=%s", recent_mae, degradation, retrain)


if __name__ == "__main__":
    main()
