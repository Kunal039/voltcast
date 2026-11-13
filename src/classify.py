"""Stage 5 - Peak-hour classification: will the next hour be a high-consumption hour?

Turns the forecasting problem into a binary alert (useful for demand response):
label = 1 when consumption exceeds the 75th percentile of the training period.
A decision tree is trained with default settings and tuned with GridSearchCV
(time-series folds, F1 scoring), and compared with a random forest classifier.
Evaluated with accuracy, precision, recall, F1, ROC-AUC and the confusion matrix.
"""
import json

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.tree import DecisionTreeClassifier

from src.features import feature_columns
from src.train import split
from src.utils import ROOT, get_logger, load_params, resolve

log = get_logger("classify")


def scores(y_true, proba, threshold: float = 0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),  # [[TN, FP], [FN, TP]]
    }


def main() -> None:
    params = load_params()
    cp = params["classification"]
    cols = feature_columns(params)
    df = pd.read_parquet(resolve(params["data"]["features"]))
    train, val, test = split(df, params)
    threshold_kw = float(train["target"].quantile(cp["threshold_quantile"]))
    label = lambda d: (d["target"] > threshold_kw).astype(int).values
    X_tr, y_tr, X_val, y_val = train[cols].values, label(train), val[cols].values, label(val)
    log.info("Peak threshold %.3f kW | positive rate train %.1f%% val %.1f%%",
             threshold_kw, y_tr.mean() * 100, y_val.mean() * 100)

    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment"] + "-peak-classifier")

    search = GridSearchCV(DecisionTreeClassifier(**cp["decision_tree"]), cp["grid"],
                          cv=TimeSeriesSplit(n_splits=params["tuning"]["cv_splits"]), scoring="f1", n_jobs=-1)
    search.fit(X_tr, y_tr)
    models = {
        "decision_tree_default": DecisionTreeClassifier(**cp["decision_tree"]).fit(X_tr, y_tr),
        "decision_tree_tuned": search.best_estimator_,
        "random_forest": RandomForestClassifier(**cp["random_forest"]).fit(X_tr, y_tr),
    }
    val_scores = {}
    for name, model in models.items():
        s = scores(y_val, model.predict_proba(X_val)[:, 1])
        s["train_f1"] = float(f1_score(y_tr, model.predict(X_tr)))
        val_scores[name] = s
        with mlflow.start_run(run_name=name):
            mlflow.log_metrics({f"val_{k}": v for k, v in s.items() if k != "confusion_matrix"})
        log.info("%-22s val acc=%.3f prec=%.3f rec=%.3f F1=%.3f AUC=%.3f (train F1 %.3f)", name, s["accuracy"],
                 s["precision"], s["recall"], s["f1"], s["roc_auc"], s["train_f1"])

    best = max(val_scores, key=lambda n: val_scores[n]["f1"])
    full = pd.concat([train, val])
    final = {"decision_tree_default": lambda: DecisionTreeClassifier(**cp["decision_tree"]),
             "decision_tree_tuned": lambda: DecisionTreeClassifier(**cp["decision_tree"], **search.best_params_),
             "random_forest": lambda: RandomForestClassifier(**cp["random_forest"])}[best]()
    final.fit(full[cols].values, label(full))
    proba = final.predict_proba(test[cols].values)[:, 1]
    y_test = label(test)
    test_scores = scores(y_test, proba)
    fpr, tpr, _ = roc_curve(y_test, proba)
    prec, rec, _ = precision_recall_curve(y_test, proba)
    log.info("TEST %s: acc=%.3f prec=%.3f rec=%.3f F1=%.3f AUC=%.3f", best, test_scores["accuracy"],
             test_scores["precision"], test_scores["recall"], test_scores["f1"], test_scores["roc_auc"])
    with mlflow.start_run(run_name=f"{best}_final"):
        mlflow.log_params({"model": best, "threshold_kw": threshold_kw, **search.best_params_})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_scores.items() if k != "confusion_matrix"})

    # Reference: flag a peak whenever the regression forecast exceeds the threshold.
    reg = pd.read_parquet(ROOT / "reports" / "test_predictions.parquet").reindex(test.index)
    rule = (reg["predicted"].values > threshold_kw).astype(float)
    rule_scores = {k: v for k, v in scores(y_test, rule).items() if k != "roc_auc"}
    log.info("TEST forecast>threshold rule: acc=%.3f prec=%.3f rec=%.3f F1=%.3f", rule_scores["accuracy"],
             rule_scores["precision"], rule_scores["recall"], rule_scores["f1"])

    joblib.dump(final, ROOT / "models" / "peak_classifier.joblib", compress=3)
    # Small sidecar the API needs at run time (reports/ is not shipped to the server).
    (ROOT / "models" / "peak_classifier.json").write_text(json.dumps(
        {"model": best, "threshold_kw": threshold_kw, "hyperparameters": final.get_params(),
         "tuned_tree_params": search.best_params_}, indent=2, default=str))
    (ROOT / "reports" / "classification.json").write_text(json.dumps({
        "threshold_kw": threshold_kw,
        "threshold_quantile": cp["threshold_quantile"],
        "best_params": {k: v for k, v in search.best_params_.items()},
        "cv_f1": float(search.best_score_),
        "validation": val_scores,
        "selected": best,
        "test": test_scores,
        "forecast_threshold_rule_test": rule_scores,
        "positive_rate_test": float(y_test.mean()),
        # Down-sampled curves for plotting.
        "roc": {"fpr": fpr[:: max(1, len(fpr) // 200)].round(4).tolist(), "tpr": tpr[:: max(1, len(tpr) // 200)].round(4).tolist()},
        "pr": {"precision": prec[:: max(1, len(prec) // 200)].round(4).tolist(), "recall": rec[:: max(1, len(rec) // 200)].round(4).tolist()},
    }, indent=2))
    log.info("Saved peak classifier (%s)", best)


if __name__ == "__main__":
    main()
