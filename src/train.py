"""Stage 4 - Training: compare candidate models with MLflow tracking, register the best.

Selection is done on a chronological validation window; the winner is then
refit on train+validation and scored once on the untouched test window.
"""
import json
import time
from datetime import datetime, timezone

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features import feature_columns
from src.utils import ROOT, get_logger, load_params, resolve

log = get_logger("train")


def split(df: pd.DataFrame, params: dict):
    train_end = pd.Timestamp(params["split"]["train_end"])
    val_end = pd.Timestamp(params["split"]["val_end"])
    return df[df.index <= train_end], df[(df.index > train_end) & (df.index <= val_end)], df[df.index > val_end]


def metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 0.1, None))) * 100),
    }


class NaiveForecaster:
    """Baseline that copies one lag column (persistence or seasonal naive)."""

    def __init__(self, column: str, columns: list[str]):
        self.idx = columns.index(column)

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.asarray(X)[:, self.idx]


def candidates(params: dict, cols: list[str]) -> dict:
    mp = params["models"]
    return {
        "persistence_baseline": NaiveForecaster("lag_1", cols),
        "seasonal_naive_24h": NaiveForecaster("lag_24", cols),
        "ridge_regression": make_pipeline(StandardScaler(), Ridge(**mp["ridge"])),
        "random_forest": RandomForestRegressor(**mp["random_forest"]),
        "hist_gradient_boosting": HistGradientBoostingRegressor(**mp["hist_gradient_boosting"]),
    }


def main() -> None:
    params = load_params()
    cols = feature_columns(params)
    df = pd.read_parquet(resolve(params["data"]["features"]))
    train, val, test = split(df, params)
    log.info("Train %d | Val %d | Test %d rows", len(train), len(val), len(test))

    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment"])

    results = {}
    for name, model in candidates(params, cols).items():
        with mlflow.start_run(run_name=name):
            t0 = time.perf_counter()
            model.fit(train[cols].values, train["target"].values)
            fit_s = time.perf_counter() - t0
            m = metrics(val["target"], model.predict(val[cols].values))
            mlflow.log_param("model", name)
            if name in params["models"]:
                mlflow.log_params(params["models"][name])
            mlflow.log_param("n_features", len(cols))
            mlflow.log_metrics({f"val_{k}": v for k, v in m.items()})
            mlflow.log_metric("fit_seconds", fit_s)
            results[name] = {**m, "fit_seconds": fit_s}
            log.info("%-24s val MAE=%.4f RMSE=%.4f R2=%.4f (%.1fs)", name, m["mae"], m["rmse"], m["r2"], fit_s)

    ml_models = [n for n in results if not isinstance(candidates(params, cols)[n], NaiveForecaster)]
    best = min(ml_models, key=lambda n: results[n]["rmse"])
    log.info("Best model on validation: %s", best)

    # Refit winner on train+val and evaluate once on the held-out test window.
    full = pd.concat([train, val])
    final = candidates(params, cols)[best]
    final.fit(full[cols].values, full["target"].values)
    test_pred = final.predict(test[cols].values)
    test_m = metrics(test["target"], test_pred)
    baseline_m = metrics(test["target"], test["lag_1"])
    log.info("TEST %s: MAE=%.4f RMSE=%.4f R2=%.4f | persistence MAE=%.4f",
             best, test_m["mae"], test_m["rmse"], test_m["r2"], baseline_m["mae"])

    with mlflow.start_run(run_name=f"{best}_final") as run:
        mlflow.log_param("model", best)
        mlflow.log_param("trained_on", "train+val")
        mlflow.log_metrics({f"test_{k}": v for k, v in test_m.items()})
        signature = infer_signature(test[cols].iloc[:5], test_pred[:5])
        mlflow.sklearn.log_model(
            final, name="model", signature=signature,
            registered_model_name=params["mlflow"]["registered_model"],
            # We produced this model ourselves in this run, so its tree type is trusted.
            skops_trusted_types=["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"],
        )
        run_id = run.info.run_id

    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(final, models_dir / "model.joblib")
    metadata = {
        "model_name": best,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mlflow_run_id": run_id,
        "features": cols,
        "train_period": [str(full.index.min()), str(full.index.max())],
        "test_period": [str(test.index.min()), str(test.index.max())],
        "test_metrics": test_m,
        "baseline_test_metrics": baseline_m,
        # Training residual spread for simple prediction intervals in the API.
        "residual_std": float(np.std(test["target"].values - test_pred)),
    }
    (models_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(
        {"validation": results, "test": {best: test_m, "persistence_baseline": baseline_m}}, indent=2))
    pd.DataFrame({"actual": test["target"], "predicted": test_pred}).to_parquet(
        ROOT / "reports" / "test_predictions.parquet")
    log.info("Saved model to %s", models_dir / "model.joblib")


if __name__ == "__main__":
    main()
