"""Stage 7 - Evaluation: EDA, model diagnostics, classification and bias-slice figures for the report and website."""
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

from src.features import feature_columns
from src.train import split
from src.utils import ROOT, get_logger, load_params, resolve

log = get_logger("evaluate")
FIG = ROOT / "reports" / "figures"
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "serif", "font.size": 10})
BLUE, ORANGE, GREY = "#1f5fa8", "#e07b24", "#888888"
LABELS = {
    "persistence_baseline": "Persistence baseline", "seasonal_naive_24h": "Seasonal naive (24 h)",
    "linear_regression": "Linear regression", "ridge_regression": "Ridge regression",
    "decision_tree": "Decision tree", "knn": "K-nearest neighbours", "svm": "SVM (SVR)",
    "random_forest": "Random forest",
}


def save(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIG / name, bbox_inches="tight")
    plt.close()


def eda(hourly: pd.DataFrame) -> None:
    y = hourly["Global_active_power"]
    daily = y.resample("D").mean()
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(daily.index, daily, color=BLUE, lw=0.7, label="Daily mean")
    ax.plot(daily.rolling(30).mean(), color=ORANGE, lw=1.8, label="30-day rolling mean")
    ax.set(title="Household active power, Dec 2006 - Nov 2010", ylabel="kW")
    ax.legend(frameon=False)
    save("fig_eda_timeseries.png")

    pivot = y.groupby([y.index.dayofweek, y.index.hour]).mean().unstack()
    pivot.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig, ax = plt.subplots(figsize=(9, 3.2))
    sns.heatmap(pivot, cmap="YlOrRd", ax=ax, cbar_kws={"label": "kW"})
    ax.set(title="Average consumption by weekday and hour", xlabel="Hour of day", ylabel="")
    save("fig_eda_heatmap.png")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    y.groupby(y.index.month).mean().plot(kind="bar", color=BLUE, ax=axes[0])
    axes[0].set(title="Average by month", xlabel="Month", ylabel="kW")
    sub = hourly[["Sub_metering_1", "Sub_metering_2", "Sub_metering_3", "Sub_metering_other"]].sum()
    sub.index = ["Kitchen", "Laundry", "Water heater / AC", "Other"]
    axes[1].pie(sub, labels=sub.index, autopct="%1.0f%%", colors=[BLUE, ORANGE, "#5aa469", GREY])
    axes[1].set(title="Energy share by sub-meter")
    save("fig_eda_month_submeter.png")

    corr = hourly.corr()
    fig, ax = plt.subplots(figsize=(6.5, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, annot_kws={"size": 7})
    ax.set(title="Correlation between measured variables")
    save("fig_eda_correlation.png")


def model_figures(params: dict) -> None:
    m = json.loads((ROOT / "reports" / "metrics.json").read_text())["validation"]
    names = list(m)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    colors = [GREY if "baseline" in n or "naive" in n else BLUE for n in names]
    best = json.loads((ROOT / "models" / "metadata.json").read_text())["model_name"]  # deployed model
    colors[names.index(best)] = ORANGE
    labels = [LABELS.get(n, n) for n in names]
    axes[0].barh(labels, [m[n]["mae"] for n in names], color=colors)
    axes[0].set(title="Validation MAE (lower is better)", xlabel="kW")
    axes[0].invert_yaxis()
    axes[1].barh(labels, [m[n]["r2"] for n in names], color=colors)
    axes[1].set(title="Validation R² (higher is better)")
    axes[1].axvline(0, color="black", lw=0.6)
    axes[1].invert_yaxis()
    axes[1].tick_params(labelleft=False)
    save("fig_model_comparison.png")

    # Over/under-fitting: training vs validation error of every default model.
    ml = [n for n in names if "baseline" not in n and "naive" not in n]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    x = np.arange(len(ml))
    ax.bar(x - 0.2, [m[n]["train_mae"] for n in ml], 0.4, color=GREY, label="Training MAE")
    ax.bar(x + 0.2, [m[n]["mae"] for n in ml], 0.4, color=BLUE, label="Validation MAE")
    ax.set_xticks(x, [LABELS[n].replace(" ", "\n", 1) for n in ml], fontsize=8)
    ax.set(title="Training vs validation error (default hyper-parameters)", ylabel="MAE (kW)")
    ax.legend(frameon=False)
    save("fig_train_vs_val.png")

    # Default vs GridSearchCV-tuned validation error.
    t = json.loads((ROOT / "reports" / "metrics.json").read_text())["tuning"]
    tn = list(t)
    fig, ax = plt.subplots(figsize=(9, 3.2))
    x = np.arange(len(tn))
    ax.bar(x - 0.2, [t[n]["default_val"]["mae"] for n in tn], 0.4, color=GREY, label="Default")
    ax.bar(x + 0.2, [t[n]["tuned_val"]["mae"] for n in tn], 0.4, color=ORANGE, label="Tuned (GridSearchCV)")
    ax.set_xticks(x, [LABELS[n] for n in tn], fontsize=8)
    ax.set(title="Effect of hyper-parameter tuning on validation MAE", ylabel="MAE (kW)")
    ax.legend(frameon=False)
    save("fig_tuning.png")

    preds = pd.read_parquet(ROOT / "reports" / "test_predictions.parquet")
    week = preds.iloc[-24 * 14:]
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(week.index, week["actual"], color=GREY, lw=1.2, label="Actual")
    ax.plot(week.index, week["predicted"], color=ORANGE, lw=1.2, label="Predicted")
    ax.set(title="Next-hour forecast vs actual - last 14 days of test set", ylabel="kW")
    ax.legend(frameon=False)
    save("fig_actual_vs_pred.png")

    resid = preds["actual"] - preds["predicted"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].scatter(preds["actual"], preds["predicted"], s=3, alpha=0.3, color=BLUE)
    lim = [0, preds.max().max()]
    axes[0].plot(lim, lim, color="black", lw=0.8, ls="--")
    axes[0].set(title="Predicted vs actual (test)", xlabel="Actual kW", ylabel="Predicted kW")
    axes[1].hist(resid, bins=60, color=BLUE)
    axes[1].set(title="Residual distribution", xlabel="Actual - predicted (kW)")
    save("fig_residuals.png")

    err = preds.assign(err=resid.abs()).groupby(preds.index.hour)["err"].mean()
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.bar(err.index, err.values, color=BLUE)
    ax.set(title="Mean absolute error by hour of day (test)", xlabel="Hour", ylabel="MAE (kW)")
    save("fig_error_by_hour.png")


def importance(params: dict) -> None:
    meta = json.loads((ROOT / "models" / "metadata.json").read_text())
    model = joblib.load(ROOT / "models" / "model.joblib")
    df = pd.read_parquet(resolve(params["data"]["features"]))
    _, _, test = split(df, params)
    r = permutation_importance(model, test[meta["features"]].values, test["target"].values,
                               n_repeats=5, random_state=0, scoring="neg_mean_absolute_error")
    imp = pd.Series(r.importances_mean, index=meta["features"]).sort_values().tail(15)
    imp.to_frame("importance").to_csv(ROOT / "reports" / "feature_importance.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(imp.index, imp.values, color=BLUE)
    ax.set(title="Top 15 features (permutation importance)", xlabel="Increase in MAE when shuffled (kW)")
    save("fig_feature_importance.png")


def monitoring_fig() -> None:
    mon = json.loads((ROOT / "reports" / "monitoring.json").read_text())
    s = pd.Series(mon["psi"]).sort_values()
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.barh(s.index, s.values, color=[ORANGE if v > 0.2 else BLUE for v in s.values])
    ax.axvline(0.1, color=GREY, ls="--", lw=0.8)
    ax.axvline(0.2, color="red", ls="--", lw=0.8)
    ax.set(title="Data drift: PSI train vs test (dashed: 0.1 / 0.2 thresholds)", xlabel="PSI")
    save("fig_drift_psi.png")


def outlier_fig(hourly: pd.DataFrame) -> None:
    cols = ["Global_active_power", "Global_reactive_power", "Global_intensity"]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for ax, c in zip(axes, cols):
        ax.boxplot(hourly[c], vert=True, widths=0.5, flierprops={"markersize": 2, "alpha": 0.3})
        ax.set(title=c.replace("_", " "), xticks=[])
    fig.suptitle("Outliers by the IQR rule (kept: they are genuine peak-usage hours)", fontsize=10)
    save("fig_outliers.png")


def validation_curve_fig(params: dict) -> None:
    """Decision-tree depth vs error: under-fitting on the left, over-fitting on the right."""
    cols = feature_columns(params)
    train, val, _ = split(pd.read_parquet(resolve(params["data"]["features"])), params)
    depths = [1, 2, 3, 4, 6, 8, 10, 12, 15, 20, 25]
    tr, va = [], []
    for d in depths:
        m = DecisionTreeRegressor(max_depth=d, random_state=42).fit(train[cols].values, train["target"].values)
        tr.append(mean_absolute_error(train["target"], m.predict(train[cols].values)))
        va.append(mean_absolute_error(val["target"], m.predict(val[cols].values)))
    best = depths[int(np.argmin(va))]
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(depths, tr, "o-", color=GREY, label="Training MAE")
    ax.plot(depths, va, "o-", color=BLUE, label="Validation MAE")
    ax.axvline(best, color=ORANGE, ls="--", lw=1)
    ax.text(best + 0.3, max(va) * 0.95, f"best depth = {best}", color=ORANGE)
    ax.text(1.2, max(va) * 0.6, "under-fitting", color=GREY, style="italic")
    ax.text(18, max(va) * 0.6, "over-fitting", color=GREY, style="italic")
    ax.set(title="Validation curve: decision tree depth", xlabel="max_depth", ylabel="MAE (kW)")
    ax.legend(frameon=False)
    save("fig_validation_curve.png")


def classification_figs() -> None:
    c = json.loads((ROOT / "reports" / "classification.json").read_text())
    t = c["test"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3))
    cm = np.array(t["confusion_matrix"])
    axes[0].imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        axes[0].text(j, i, f"{v:,}", ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
    axes[0].set(title="Confusion matrix (test)", xticks=[0, 1], yticks=[0, 1],
                xticklabels=["Normal", "Peak"], yticklabels=["Normal", "Peak"], xlabel="Predicted", ylabel="Actual")
    axes[1].plot(c["roc"]["fpr"], c["roc"]["tpr"], color=ORANGE, lw=1.8)
    axes[1].plot([0, 1], [0, 1], color=GREY, ls="--", lw=0.8)
    axes[1].set(title=f"ROC curve (AUC = {t['roc_auc']:.3f})", xlabel="False positive rate", ylabel="True positive rate")
    axes[2].plot(c["pr"]["recall"], c["pr"]["precision"], color=BLUE, lw=1.8)
    axes[2].axhline(c["positive_rate_test"], color=GREY, ls="--", lw=0.8)
    axes[2].set(title="Precision-recall curve", xlabel="Recall", ylabel="Precision")
    save("fig_classification.png")

    v = c["validation"]
    names = list(v)
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    fig, ax = plt.subplots(figsize=(9, 3.2))
    x = np.arange(len(keys))
    for i, (n, col) in enumerate(zip(names, [GREY, BLUE, ORANGE])):
        ax.bar(x + (i - 1) * 0.27, [v[n][k] for k in keys], 0.27, color=col, label=n.replace("_", " "))
    ax.set_xticks(x, ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"])
    ax.set(title="Peak-hour classifiers on the validation set", ylim=(0, 1))
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    save("fig_classifier_comparison.png")


def fairness_slices() -> None:
    """Bias check: is the forecast equally good for every kind of hour and season?"""
    p = pd.read_parquet(ROOT / "reports" / "test_predictions.parquet")
    p["err"] = (p["actual"] - p["predicted"]).abs()
    idx = p.index
    slices = {
        "Day type": np.where(idx.dayofweek >= 5, "Weekend", "Weekday"),
        "Time of day": pd.cut(idx.hour, [-1, 5, 11, 17, 23], labels=["Night 0-5", "Morning 6-11",
                                                                     "Afternoon 12-17", "Evening 18-23"]),
        "Season": pd.Categorical(np.where(idx.month.isin([6, 7, 8]), "Summer (Jun-Aug)", "Autumn (Sep-Nov)"),
                                 categories=["Summer (Jun-Aug)", "Autumn (Sep-Nov)"]),
        "Usage level": pd.qcut(p["actual"], 3, labels=["Low use", "Medium use", "High use"]),
    }
    out = {}
    for name, groups in slices.items():
        g = p.groupby(pd.Series(groups, index=p.index), observed=True)["err"].agg(["mean", "count"])
        out[name] = {str(k): {"mae": round(float(r["mean"]), 4), "hours": int(r["count"])} for k, r in g.iterrows()}
    overall = float(p["err"].mean())
    (ROOT / "reports" / "fairness.json").write_text(json.dumps({"overall_mae": overall, "slices": out}, indent=2))

    fig, axes = plt.subplots(1, 4, figsize=(11, 3.2), sharey=True)
    for ax, (name, groups) in zip(axes, out.items()):
        ax.bar(list(groups), [v["mae"] for v in groups.values()], color=BLUE)
        ax.axhline(overall, color=ORANGE, ls="--", lw=1)
        ax.set_title(name)
        ax.tick_params(axis="x", labelrotation=30, labelsize=7.5)
    axes[0].set_ylabel("MAE (kW)")
    fig.suptitle("Error across sub-groups of the test period (dashed = overall MAE)", fontsize=10)
    save("fig_fairness.png")


def main() -> None:
    params = load_params()
    FIG.mkdir(parents=True, exist_ok=True)
    hourly = pd.read_parquet(resolve(params["data"]["processed"]))
    eda(hourly)
    outlier_fig(hourly)
    model_figures(params)
    validation_curve_fig(params)
    importance(params)
    classification_figs()
    fairness_slices()
    if (ROOT / "reports" / "monitoring.json").exists():
        monitoring_fig()
    log.info("Figures written to %s", FIG)


if __name__ == "__main__":
    main()
