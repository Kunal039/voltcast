"""Stage 5 - Evaluation: EDA and model diagnostic figures for the report and dashboard."""
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance

from src.train import split
from src.utils import ROOT, get_logger, load_params, resolve

log = get_logger("evaluate")
FIG = ROOT / "reports" / "figures"
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "serif", "font.size": 10})
BLUE, ORANGE, GREY = "#1f5fa8", "#e07b24", "#888888"


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
    best = min([n for n in names if colors[names.index(n)] == BLUE], key=lambda n: m[n]["rmse"])
    colors[names.index(best)] = ORANGE
    labels = [n.replace("_", "\n") for n in names]
    axes[0].bar(labels, [m[n]["mae"] for n in names], color=colors)
    axes[0].set(title="Validation MAE (lower is better)", ylabel="kW")
    axes[1].bar(labels, [m[n]["r2"] for n in names], color=colors)
    axes[1].set(title="Validation R² (higher is better)")
    axes[1].axhline(0, color="black", lw=0.6)
    for a in axes:
        a.tick_params(axis="x", labelsize=7)
    save("fig_model_comparison.png")

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


def main() -> None:
    params = load_params()
    FIG.mkdir(parents=True, exist_ok=True)
    eda(pd.read_parquet(resolve(params["data"]["processed"])))
    model_figures(params)
    importance(params)
    if (ROOT / "reports" / "monitoring.json").exists():
        monitoring_fig()
    log.info("Figures written to %s", FIG)


if __name__ == "__main__":
    main()
