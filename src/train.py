"""Stage 4 - Training: compare candidate models with MLflow tracking, tune them, register the best.

1. Every syllabus algorithm (linear regression, decision tree, KNN, SVM, random forest)
   plus ridge, gradient boosting and two naive baselines is trained with default settings.
   Train and validation errors are both recorded to expose over- and under-fitting.
2. Models with a grid in params.yaml are tuned with GridSearchCV using time-series
   cross-validation on the training period (each fold only trains on the past).
3. The best variant on the chronological validation window is refit on train+validation
   and scored once on the untouched test window.
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
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from src.features import feature_columns
from src.utils import ROOT, get_logger, load_params, resolve

log = get_logger("train")
# Tree node storage of the models this pipeline itself trains (forests, single trees, boosting).
TRUSTED_TYPES = ["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor", "sklearn.tree._tree.Tree"]


def split(df: pd.DataFrame, params: dict):
    train_end = pd.Timestamp(params["split"]["train_end"])
    val_end = pd.Timestamp(params["split"]["val_end"])
    return df[df.index <= train_end], df[(df.index > train_end) & (df.index <= val_end)], df[df.index > val_end]


def metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
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


def build(name: str, params: dict, cols: list[str], overrides: dict | None = None):
    """Create a fresh, unfitted model. Distance- and margin-based models get feature scaling."""
    hp = {**params["models"].get(name, {}), **(overrides or {})}
    makers = {
        "persistence_baseline": lambda: NaiveForecaster("lag_1", cols),
        "seasonal_naive_24h": lambda: NaiveForecaster("lag_24", cols),
        "linear_regression": lambda: make_pipeline(StandardScaler(), LinearRegression(**hp)),
        "ridge_regression": lambda: make_pipeline(StandardScaler(), Ridge(**hp)),
        "decision_tree": lambda: DecisionTreeRegressor(**hp),
        "knn": lambda: make_pipeline(StandardScaler(), KNeighborsRegressor(**hp)),
        "svm": lambda: make_pipeline(StandardScaler(), SVR(**hp)),
        "random_forest": lambda: RandomForestRegressor(**hp),
        "hist_gradient_boosting": lambda: HistGradientBoostingRegressor(**hp),
    }
    return makers[name]()


CANDIDATES = ["persistence_baseline", "seasonal_naive_24h", "linear_regression", "ridge_regression",
              "decision_tree", "knn", "svm", "random_forest", "hist_gradient_boosting"]


def tune(name: str, params: dict, cols: list[str], X, y) -> GridSearchCV:
    model = build(name, params, cols)
    grid = params["tuning"]["grids"][name]
    # Inside a pipeline, parameters are addressed as <step>__<param>.
    prefix = f"{model.steps[-1][0]}__" if isinstance(model, Pipeline) else ""
    search = GridSearchCV(
        model, {prefix + k: v for k, v in grid.items()},
        cv=TimeSeriesSplit(n_splits=params["tuning"]["cv_splits"]),
        scoring=params["tuning"]["scoring"],
        n_jobs=1 if name == "random_forest" else -1,  # the forest already uses every core
        return_train_score=True,
    )
    search.fit(X, y)
    search.best_params_unprefixed = {k.removeprefix(prefix): v for k, v in search.best_params_.items()}
    return search


def main() -> None:
    params = load_params()
    cols = feature_columns(params)
    df = pd.read_parquet(resolve(params["data"]["features"]))
    train, val, test = split(df, params)
    X_tr, y_tr, X_val, y_val = train[cols].values, train["target"].values, val[cols].values, val["target"].values
    log.info("Train %d | Val %d | Test %d rows", len(train), len(val), len(test))

    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment"])

    # 1. Default models: train vs validation error.
    results = {}
    for name in CANDIDATES:
        model = build(name, params, cols)
        with mlflow.start_run(run_name=name):
            t0 = time.perf_counter()
            model.fit(X_tr, y_tr)
            fit_s = time.perf_counter() - t0
            m = metrics(y_val, model.predict(X_val))
            train_m = metrics(y_tr, model.predict(X_tr))
            mlflow.log_params({"model": name, "variant": "default", "n_features": len(cols), **params["models"].get(name, {})})
            mlflow.log_metrics({**{f"val_{k}": v for k, v in m.items()}, "train_mae": train_m["mae"],
                                "train_r2": train_m["r2"], "fit_seconds": fit_s})
            results[name] = {**m, "train_mae": train_m["mae"], "train_r2": train_m["r2"], "fit_seconds": fit_s}
            log.info("%-24s train MAE=%.4f  val MAE=%.4f RMSE=%.4f R2=%.4f (%.1fs)",
                     name, train_m["mae"], m["mae"], m["rmse"], m["r2"], fit_s)

    # 2. Hyper-parameter tuning with time-series cross-validation.
    tuning = {}
    for name in params["tuning"]["grids"]:
        t0 = time.perf_counter()
        search = tune(name, params, cols, X_tr, y_tr)
        tuned_val = metrics(y_val, search.best_estimator_.predict(X_val))
        tuning[name] = {
            "best_params": search.best_params_unprefixed,
            "cv_mae": float(-search.best_score_),
            "candidates_tried": len(search.cv_results_["params"]),
            "default_val": {k: results[name][k] for k in ["mae", "rmse", "r2"]},
            "tuned_val": {k: tuned_val[k] for k in ["mae", "rmse", "r2"]},
            "search_seconds": time.perf_counter() - t0,
        }
        with mlflow.start_run(run_name=f"{name}_tuned"):
            mlflow.log_params({"model": name, "variant": "tuned", **search.best_params_unprefixed})
            mlflow.log_metrics({"cv_mae": tuning[name]["cv_mae"], **{f"val_{k}": v for k, v in tuned_val.items()}})
        log.info("tuned %-18s %s  CV MAE=%.4f  val MAE %.4f -> %.4f", name, search.best_params_unprefixed,
                 tuning[name]["cv_mae"], results[name]["mae"], tuned_val["mae"])

    # 3. Pick the best variant (default or tuned) on validation RMSE among the syllabus algorithms;
    #    the best model overall is recorded for comparison only.
    variants = [(n, None, results[n]["rmse"]) for n in CANDIDATES if n not in ("persistence_baseline", "seasonal_naive_24h")]
    variants += [(n, t["best_params"], t["tuned_val"]["rmse"]) for n, t in tuning.items()]
    overall = min(variants, key=lambda v: v[2])
    allowed = params["selection"]["deployable"]
    best, best_params, _ = min((v for v in variants if v[0] in allowed), key=lambda v: v[2])
    log.info("Best model on validation: %s (%s); best overall incl. reference models: %s", best,
             "tuned " + str(best_params) if best_params else "default", overall[0])

    full = pd.concat([train, val])
    final = build(best, params, cols, best_params)
    final.fit(full[cols].values, full["target"].values)
    test_pred = final.predict(test[cols].values)
    test_m = metrics(test["target"], test_pred)
    baseline_m = metrics(test["target"], test["lag_1"])
    log.info("TEST %s: MAE=%.4f RMSE=%.4f R2=%.4f | persistence MAE=%.4f",
             best, test_m["mae"], test_m["rmse"], test_m["r2"], baseline_m["mae"])

    with mlflow.start_run(run_name=f"{best}_final") as run:
        mlflow.log_params({"model": best, "trained_on": "train+val", **(best_params or {})})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_m.items()})
        mlflow.sklearn.log_model(
            final, name="model", signature=infer_signature(test[cols].iloc[:5], test_pred[:5]),
            registered_model_name=params["mlflow"]["registered_model"],
            # We produced this model ourselves in this run, so its tree type is trusted.
            skops_trusted_types=TRUSTED_TYPES,
        )
        run_id = run.info.run_id

    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(final, models_dir / "model.joblib", compress=3)  # a 200-tree forest is ~65 MB uncompressed
    metadata = {
        "model_name": best,
        "hyperparameters": {**params["models"].get(best, {}), **(best_params or {})},
        "tuned": best_params is not None,
        "selection": {"deployable": allowed, "best_overall": overall[0], "best_overall_val_rmse": overall[2]},
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mlflow_run_id": run_id,
        "features": cols,
        "train_period": [str(full.index.min()), str(full.index.max())],
        "test_period": [str(test.index.min()), str(test.index.max())],
        "test_metrics": test_m,
        "baseline_test_metrics": baseline_m,
        # Residual spread on the test window, used for simple prediction intervals in the API.
        "residual_std": float(np.std(test["target"].values - test_pred)),
    }
    (models_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(
        {"validation": results, "tuning": tuning, "test": {best: test_m, "persistence_baseline": baseline_m}}, indent=2))
    pd.DataFrame({"actual": test["target"], "predicted": test_pred}).to_parquet(
        ROOT / "reports" / "test_predictions.parquet")
    log.info("Saved model to %s", models_dir / "model.joblib")


if __name__ == "__main__":
    main()
