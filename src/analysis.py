"""Stage 8 - Extended analysis for the report: every model on the test set, extra metrics, curves.

Produces reports/analysis.json and the figures in reports/analysis/:
  * test-set scores of all nine regressors (default and tuned), with accuracy-style metrics
  * per-model forecast, scatter and residual plots
  * error by forecast horizon (1-24 h ahead) for the deployed recursive forecaster
  * learning curves, cross-validation fold scores and hyper-parameter response curves
  * confusion matrices, ROC/PR overlays, classification report and threshold sweep
  * additional EDA (missing data, distributions, autocorrelation, profiles) and monthly drift
"""
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (classification_report, confusion_matrix, explained_variance_score, max_error,
                             median_absolute_error, precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_validate, learning_curve
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src.features import feature_columns
from src.monitor import MONITORED, psi
from src.predict import MIN_HISTORY_HOURS, forecast
from src.train import CANDIDATES, build, metrics, split
from src.utils import ROOT, get_logger, load_params, resolve

log = get_logger("analysis")
OUT = ROOT / "reports" / "analysis"
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "serif", "font.size": 9})
BLUE, ORANGE, GREY, GREEN, RED = "#1f5fa8", "#e07b24", "#888888", "#2f8f5b", "#b42318"
LABELS = {
    "persistence_baseline": "Persistence baseline", "seasonal_naive_24h": "Seasonal naive (24 h)",
    "linear_regression": "Linear regression", "ridge_regression": "Ridge regression",
    "decision_tree": "Decision tree", "knn": "K-nearest neighbours", "svm": "SVM (SVR)",
    "random_forest": "Random forest", "hist_gradient_boosting": "Gradient boosting",
}


def save(name: str) -> None:
    plt.tight_layout()
    plt.savefig(OUT / name, bbox_inches="tight")
    plt.close()


def extended_metrics(y, p) -> dict:
    """Standard regression metrics plus accuracy-style measures that are easy to explain."""
    y, p = np.asarray(y), np.asarray(p)
    err = np.abs(y - p)
    return {
        **metrics(y, p),
        "median_ae": float(median_absolute_error(y, p)),
        "max_error": float(max_error(y, p)),
        "explained_variance": float(explained_variance_score(y, p)),
        # 1 - WAPE: share of total consumption predicted correctly (common in energy forecasting).
        "accuracy_1_minus_wape": float(1 - err.sum() / np.abs(y).sum()),
        "within_0_25kw_pct": float((err <= 0.25).mean() * 100),
        "within_0_5kw_pct": float((err <= 0.5).mean() * 100),
    }


# ---------------------------------------------------------------- regression on the test set
def all_models_on_test(params, cols, train, val, test, tuning):
    full = pd.concat([train, val])
    X, y, Xt, yt = full[cols].values, full["target"].values, test[cols].values, test["target"].values
    results, preds = {}, {}
    variants = [(n, None) for n in CANDIDATES] + [(n, t["best_params"]) for n, t in tuning.items()]
    for name, hp in variants:
        key = name + ("_tuned" if hp else "")
        t0 = time.perf_counter()
        model = build(name, params, cols, hp).fit(X, y)
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        p = model.predict(Xt)
        pred_ms = (time.perf_counter() - t0) / len(Xt) * 1000
        results[key] = {**extended_metrics(yt, p), "fit_seconds": fit_s, "predict_ms_per_row": pred_ms}
        if not hp:
            preds[name] = p
        log.info("test %-30s MAE=%.4f R2=%.4f acc(1-WAPE)=%.3f", key, results[key]["mae"], results[key]["r2"],
                 results[key]["accuracy_1_minus_wape"])
    return results, preds


def per_model_figures(test, preds):
    names = list(preds)
    week = slice(-24 * 7, None)
    fig, axes = plt.subplots(3, 3, figsize=(10, 7.5), sharey=True)
    for ax, n in zip(axes.flat, names):
        ax.plot(test.index[week], test["target"].values[week], color=GREY, lw=0.9, label="Actual")
        ax.plot(test.index[week], preds[n][week], color=ORANGE, lw=0.9, label="Predicted")
        ax.set_title(LABELS[n], fontsize=9)
        ax.tick_params(axis="x", labelsize=6, rotation=30)
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle("Next-hour forecasts of every model, last 7 days of the test set", fontsize=10)
    save("per_model_forecasts.png")

    fig, axes = plt.subplots(3, 3, figsize=(10, 9))
    lim = [0, float(test["target"].max())]
    for ax, n in zip(axes.flat, names):
        ax.scatter(test["target"], preds[n], s=2, alpha=0.25, color=BLUE)
        ax.plot(lim, lim, color="black", lw=0.7, ls="--")
        ax.set(title=LABELS[n], xlabel="Actual kW", ylabel="Predicted kW")
    fig.suptitle("Predicted vs actual, test set (dashed line = perfect prediction)", fontsize=10)
    save("per_model_scatter.png")

    fig, axes = plt.subplots(3, 3, figsize=(10, 7.5), sharex=True)
    for ax, n in zip(axes.flat, names):
        r = test["target"].values - preds[n]
        ax.hist(r, bins=60, range=(-3, 3), color=BLUE)
        ax.axvline(0, color="black", lw=0.7)
        ax.set_title(f"{LABELS[n]}  (mean {r.mean():+.3f})", fontsize=9)
    fig.suptitle("Residual distribution of every model, test set (actual - predicted, kW)", fontsize=10)
    save("per_model_residuals.png")


def test_metric_bars(results):
    names = [n for n in CANDIDATES]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, (k, title) in zip(axes.flat, [("mae", "MAE (kW)"), ("rmse", "RMSE (kW)"), ("r2", "R²"),
                                          ("accuracy_1_minus_wape", "Accuracy (1 - WAPE)")]):
        vals = [results[n][k] for n in names]
        cols = [GREY if "baseline" in n or "naive" in n else BLUE for n in names]
        cols[names.index("hist_gradient_boosting")] = ORANGE
        ax.barh([LABELS[n] for n in names], vals, color=cols)
        ax.invert_yaxis()
        ax.set_title(f"Test {title}")
        for i, v in enumerate(vals):
            ax.text(v, i, f" {v:.3f}", va="center", fontsize=7)
    save("test_metrics_all_models.png")


# ---------------------------------------------------------------- forecasting horizon
def horizon_curve(params):
    hourly = pd.read_csv(ROOT / params["data"]["serving"], index_col="datetime", parse_dates=True)
    rng = np.random.default_rng(7)
    lo, hi = MIN_HISTORY_HOURS * 2, len(hourly) - 25
    origins = hourly.index[np.sort(rng.choice(np.arange(lo, hi), 60, replace=False))]
    err = np.zeros((len(origins), 24))
    naive = np.zeros((len(origins), 24))
    for i, o in enumerate(origins):
        hist = hourly[hourly.index < o].iloc[-MIN_HISTORY_HOURS * 2:]
        fc = forecast(hist, 24)
        actual = hourly["Global_active_power"].reindex(fc["timestamp"]).values
        err[i] = np.abs(actual - fc["predicted_kw"].values)
        naive[i] = np.abs(actual - hourly["Global_active_power"].reindex(fc["timestamp"] - pd.Timedelta(hours=24)).values)
    mae, mae_naive = err.mean(axis=0), naive.mean(axis=0)
    fig, ax = plt.subplots(figsize=(9, 3.4))
    h = np.arange(1, 25)
    ax.plot(h, mae, "o-", color=ORANGE, label="VoltCast recursive forecast")
    ax.plot(h, mae_naive, "s--", color=GREY, label="Same hour yesterday (seasonal naive)")
    ax.set(title=f"Forecast error by horizon ({len(origins)} random origins in the test period)",
           xlabel="Hours ahead", ylabel="MAE (kW)", xticks=h)
    ax.legend(frameon=False)
    save("horizon_curve.png")
    return {"origins": len(origins), "mae_by_hour": mae.round(4).tolist(), "naive_mae_by_hour": mae_naive.round(4).tolist()}


# ---------------------------------------------------------------- learning curves and CV
def learning_curves(params, cols, train):
    X, y = train[cols].values, train["target"].values
    cv = TimeSeriesSplit(n_splits=5)
    picks = [("linear_regression", None), ("decision_tree", None), ("random_forest", {"n_estimators": 100}),
             ("hist_gradient_boosting", None)]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.2), sharey=True)
    out = {}
    for ax, (n, hp) in zip(axes, picks):
        sizes, tr, va = learning_curve(build(n, params, cols, hp), X, y, cv=cv, n_jobs=-1,
                                       train_sizes=np.linspace(0.2, 1.0, 5), scoring="neg_mean_absolute_error")
        ax.plot(sizes, -tr.mean(axis=1), "o-", color=GREY, label="Training")
        ax.plot(sizes, -va.mean(axis=1), "o-", color=BLUE, label="Cross-validation")
        ax.fill_between(sizes, -va.mean(axis=1) - va.std(axis=1), -va.mean(axis=1) + va.std(axis=1), color=BLUE, alpha=0.12)
        ax.set(title=LABELS[n], xlabel="Training rows")
        out[n] = {"sizes": sizes.tolist(), "train_mae": (-tr.mean(axis=1)).round(4).tolist(),
                  "cv_mae": (-va.mean(axis=1)).round(4).tolist()}
    axes[0].set_ylabel("MAE (kW)")
    axes[0].legend(frameon=False)
    fig.suptitle("Learning curves (time-series cross-validation)", fontsize=10)
    save("learning_curves.png")
    return out


def cv_fold_scores(params, cols, train, tuning):
    X, y = train[cols].values, train["target"].values
    out = {}
    for n, t in tuning.items():
        r = cross_validate(build(n, params, cols, t["best_params"]), X, y, cv=TimeSeriesSplit(n_splits=5),
                           scoring="neg_mean_absolute_error", n_jobs=1 if n == "random_forest" else -1)
        out[n] = (-r["test_score"]).round(4).tolist()
    fig, ax = plt.subplots(figsize=(9, 3.4))
    for i, (n, s) in enumerate(out.items()):
        ax.plot(range(1, 6), s, "o-", label=LABELS[n])
    ax.set(title="Mean absolute error in each time-series CV fold (tuned models)", xlabel="Fold (later folds train on more history)",
           ylabel="MAE (kW)", xticks=range(1, 6))
    ax.legend(frameon=False, fontsize=8)
    save("cv_folds.png")
    return out


def hyperparameter_curves(params, cols, train, val):
    X, y, Xv, yv = train[cols].values, train["target"].values, val[cols].values, val["target"].values
    out = {}
    # Decision tree: full depth x leaf grid as a heat map of CV MAE.
    grid = params["tuning"]["grids"]["decision_tree"]
    gs = GridSearchCV(DecisionTreeRegressor(random_state=42), grid, cv=TimeSeriesSplit(5),
                      scoring="neg_mean_absolute_error", n_jobs=-1).fit(X, y)
    depths = [str(d) for d in grid["max_depth"]]
    leaves = grid["min_samples_leaf"]
    heat = np.zeros((len(depths), len(leaves)))
    for prm, s in zip(gs.cv_results_["params"], gs.cv_results_["mean_test_score"]):
        heat[depths.index(str(prm["max_depth"])), leaves.index(prm["min_samples_leaf"])] = -s
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    im = axes[0].imshow(heat, cmap="viridis_r", aspect="auto")
    for (i, j), v in np.ndenumerate(heat):
        axes[0].text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=6.5, color="white" if v > heat.mean() else "black")
    axes[0].set(title="Decision tree: CV MAE", xticks=range(len(leaves)), xticklabels=leaves, yticks=range(len(depths)),
                yticklabels=depths, xlabel="min_samples_leaf", ylabel="max_depth")
    out["decision_tree_grid"] = {"max_depth": depths, "min_samples_leaf": leaves, "cv_mae": heat.round(4).tolist()}
    # KNN: neighbours vs validation MAE.
    ks = [1, 3, 5, 10, 20, 40, 80]
    knn = [float(np.mean(np.abs(make_pipeline(StandardScaler(), KNeighborsRegressor(k)).fit(X, y).predict(Xv) - yv))) for k in ks]
    axes[1].plot(ks, knn, "o-", color=BLUE)
    axes[1].set(title="KNN: number of neighbours", xlabel="k (log scale)", ylabel="Validation MAE (kW)", xscale="log")
    out["knn_k"] = {"k": ks, "val_mae": np.round(knn, 4).tolist()}
    # Ridge: regularisation strength.
    alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
    rid = [float(np.mean(np.abs(make_pipeline(StandardScaler(), Ridge(alpha=a)).fit(X, y).predict(Xv) - yv))) for a in alphas]
    axes[2].plot(alphas, rid, "o-", color=ORANGE)
    axes[2].set(title="Ridge: regularisation strength", xlabel="alpha (log scale)", ylabel="Validation MAE (kW)", xscale="log")
    out["ridge_alpha"] = {"alpha": alphas, "val_mae": np.round(rid, 4).tolist()}
    save("hyperparameter_curves.png")
    return out


# ---------------------------------------------------------------- classification
def classification_analysis(params, cols, train, val, test):
    cp = params["classification"]
    thr = float(train["target"].quantile(cp["threshold_quantile"]))
    lab = lambda d: (d["target"] > thr).astype(int).values
    X, y, Xv, yv, Xt, yt = train[cols].values, lab(train), val[cols].values, lab(val), test[cols].values, lab(test)
    tuned_params = json.loads((ROOT / "models" / "peak_classifier.json").read_text())["hyperparameters"]
    models = {
        "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "Decision tree (default)": DecisionTreeClassifier(random_state=42),
        "Decision tree (tuned)": DecisionTreeClassifier(random_state=42, **tuned_params),
    }
    full = pd.concat([train, val])
    out = {"threshold_kw": thr, "classifiers": {}}
    fig_cm, axes_cm = plt.subplots(1, 3, figsize=(10, 3.2))
    fig_roc, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, (name, m) in zip(axes_cm, models.items()):
        m.fit(full[cols].values, lab(full))
        proba = m.predict_proba(Xt)[:, 1]
        pred = (proba >= 0.5).astype(int)
        cm = confusion_matrix(yt, pred)
        rep = classification_report(yt, pred, target_names=["Normal hour", "Peak hour"], output_dict=True)
        out["classifiers"][name] = {"confusion_matrix": cm.tolist(), "report": rep, "roc_auc": float(roc_auc_score(yt, proba))}
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v:,}", ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
        ax.set(title=name, xticks=[0, 1], yticks=[0, 1], xticklabels=["Normal", "Peak"], yticklabels=["Normal", "Peak"],
               xlabel="Predicted", ylabel="Actual")
        fpr, tpr, _ = roc_curve(yt, proba)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC {out['classifiers'][name]['roc_auc']:.3f})")
        pr, rc, _ = precision_recall_curve(yt, proba)
        ax_pr.plot(rc, pr, label=name)
        if name == "Decision tree (tuned)":
            sweep = []
            for t in np.linspace(0.05, 0.95, 19):
                pt = (proba >= t).astype(int)
                tp, fp, fn = ((pt == 1) & (yt == 1)).sum(), ((pt == 1) & (yt == 0)).sum(), ((pt == 0) & (yt == 1)).sum()
                p_, r_ = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
                sweep.append({"threshold": round(float(t), 2), "precision": float(p_), "recall": float(r_),
                              "f1": float(2 * p_ * r_ / max(p_ + r_, 1e-9)), "accuracy": float((pt == yt).mean())})
            out["threshold_sweep"] = sweep
    fig_cm.suptitle("Confusion matrices on the test set (refit on train + validation)", fontsize=10)
    plt.figure(fig_cm.number)
    save("classifier_confusion_matrices.png")
    ax_roc.plot([0, 1], [0, 1], color=GREY, ls="--", lw=0.8)
    ax_roc.set(title="ROC curves (test)", xlabel="False positive rate", ylabel="True positive rate")
    ax_roc.legend(frameon=False, fontsize=7.5, loc="lower right")
    ax_pr.axhline(yt.mean(), color=GREY, ls="--", lw=0.8)
    ax_pr.set(title="Precision-recall curves (test)", xlabel="Recall", ylabel="Precision")
    ax_pr.legend(frameon=False, fontsize=7.5)
    plt.figure(fig_roc.number)
    save("classifier_roc_pr.png")

    s = pd.DataFrame(out["threshold_sweep"])
    fig, ax = plt.subplots(figsize=(9, 3.2))
    for k, c in [("precision", BLUE), ("recall", ORANGE), ("f1", GREEN), ("accuracy", GREY)]:
        ax.plot(s["threshold"], s[k], "o-", color=c, ms=3, label=k.capitalize() if k != "f1" else "F1")
    ax.axvline(0.5, color="black", ls=":", lw=0.8)
    ax.set(title="Tuned decision tree: effect of the decision threshold (test)", xlabel="Probability threshold", ylim=(0, 1))
    ax.legend(frameon=False, ncol=4)
    save("classifier_threshold_sweep.png")
    return out


# ---------------------------------------------------------------- data analysis
def eda_extra(params):
    raw = pd.read_csv(resolve(params["data"]["raw_csv"]), parse_dates=["Datetime"], index_col="Datetime",
                      usecols=["Datetime", "Global_active_power"])
    hourly = pd.read_parquet(resolve(params["data"]["processed"]))
    out = {}
    miss = raw["Global_active_power"].isna().resample("MS").mean() * 100
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.bar(miss.index, miss.values, width=25, color=RED)
    ax.set(title="Share of missing one-minute readings per month (%)", ylabel="% missing")
    save("missing_by_month.png")
    out["missing_pct_by_month_max"] = {"month": str(miss.idxmax().date()), "pct": round(float(miss.max()), 2)}

    cols = ["Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity",
            "Sub_metering_1", "Sub_metering_2", "Sub_metering_3", "Sub_metering_other"]
    fig, axes = plt.subplots(2, 4, figsize=(11, 5))
    for ax, c in zip(axes.flat, cols):
        ax.hist(hourly[c], bins=60, color=BLUE)
        ax.set_title(c.replace("_", " "), fontsize=8.5)
    fig.suptitle("Distribution of each hourly variable", fontsize=10)
    save("distributions.png")
    out["describe"] = hourly[cols].describe().round(3).to_dict()

    y = hourly["Global_active_power"]
    lags = np.arange(1, 24 * 8 + 1)
    acf = [float(y.autocorr(l)) for l in lags]
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.bar(lags, acf, width=0.8, color=BLUE)
    for d in range(1, 9):
        ax.axvline(24 * d, color=ORANGE, lw=0.6, ls="--")
    ax.set(title="Autocorrelation of hourly consumption (dashed: multiples of 24 h)", xlabel="Lag (hours)", ylabel="Autocorrelation")
    save("autocorrelation.png")
    out["autocorr"] = {"lag_1": round(acf[0], 3), "lag_24": round(acf[23], 3), "lag_168": round(acf[167], 3)}

    week = y.groupby([y.index.dayofweek, y.index.hour]).mean().values
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(np.arange(168), week, color=BLUE)
    ax.set(title="Average weekly load profile (Monday 00:00 to Sunday 23:00)", xlabel="Hour of week", ylabel="kW",
           xticks=np.arange(0, 169, 24), xticklabels=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun", ""])
    save("weekly_profile.png")

    sub = hourly[["Sub_metering_1", "Sub_metering_2", "Sub_metering_3", "Sub_metering_other"]]
    prof = sub.groupby(sub.index.hour).mean() / 1000
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.stackplot(prof.index, prof.T.values, labels=["Kitchen", "Laundry", "Water heater / AC", "Other"],
                 colors=[BLUE, ORANGE, GREEN, "#c9ced6"])
    ax.set(title="Average energy by sub-meter and hour of day", xlabel="Hour", ylabel="kWh per hour")
    ax.legend(frameon=False, ncol=4, fontsize=8)
    save("submeter_by_hour.png")

    fig, ax = plt.subplots(figsize=(9, 3.2))
    data = [y[y.index.month == m].values for m in range(1, 13)]
    ax.boxplot(data, widths=0.6, flierprops={"markersize": 1.5, "alpha": 0.3})
    ax.set(title="Hourly consumption by month", xlabel="Month", ylabel="kW")
    save("monthly_boxplot.png")

    yearly = y.groupby([y.index.year, y.index.month]).mean().unstack(0)
    fig, ax = plt.subplots(figsize=(9, 3.2))
    for yr in yearly.columns:
        ax.plot(yearly.index, yearly[yr], "o-", ms=3, label=str(yr))
    ax.set(title="Monthly average consumption, year by year", xlabel="Month", ylabel="kW", xticks=range(1, 13))
    ax.legend(frameon=False, ncol=5)
    save("year_over_year.png")
    return out


def drift_over_time(params, train, test):
    preds = pd.read_parquet(ROOT / "reports" / "test_predictions.parquet")
    out = {"monthly_psi": {}, "monthly_mae": {}}
    months = sorted(set(test.index.to_period("M")))
    for m in months:
        part = test[test.index.to_period("M") == m]
        out["monthly_psi"][str(m)] = {c: round(psi(train[c].values, part[c].values), 4) for c in MONITORED if c != "hour"}
        pm = preds[preds.index.to_period("M") == m]
        out["monthly_mae"][str(m)] = round(float((pm["actual"] - pm["predicted"]).abs().mean()), 4)
    df = pd.DataFrame(out["monthly_psi"]).T
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4))
    for c in df.columns:
        axes[0].plot(range(len(df)), df[c], "o-", ms=3, label=c)
    axes[0].axhline(0.2, color=RED, ls="--", lw=0.8)
    axes[0].set(title="PSI against training data, month by month", xticks=range(len(df)), xticklabels=df.index, ylabel="PSI")
    axes[0].legend(frameon=False, fontsize=6.5)
    axes[1].bar(list(out["monthly_mae"]), list(out["monthly_mae"].values()), color=BLUE)
    axes[1].set(title="Test MAE by month", ylabel="MAE (kW)")
    for a in axes:
        a.tick_params(axis="x", rotation=30)
    save("monthly_drift.png")
    return out


def main() -> None:
    params = load_params()
    OUT.mkdir(parents=True, exist_ok=True)
    cols = feature_columns(params)
    df = pd.read_parquet(resolve(params["data"]["features"]))
    train, val, test = split(df, params)
    tuning = json.loads((ROOT / "reports" / "metrics.json").read_text())["tuning"]

    test_results, preds = all_models_on_test(params, cols, train, val, test, tuning)
    per_model_figures(test, preds)
    test_metric_bars(test_results)
    analysis = {
        "test_all_models": test_results,
        "horizon": horizon_curve(params),
        "learning_curves": learning_curves(params, cols, train),
        "cv_folds": cv_fold_scores(params, cols, train, tuning),
        "hyperparameters": hyperparameter_curves(params, cols, train, val),
        "classification": classification_analysis(params, cols, train, val, test),
        "eda": eda_extra(params),
        "drift": drift_over_time(params, train, test),
    }
    (ROOT / "reports" / "analysis.json").write_text(json.dumps(analysis, indent=2))
    log.info("Analysis written: %d figures", len(list(OUT.glob("*.png"))))


if __name__ == "__main__":
    main()
