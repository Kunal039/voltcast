"""Streamlit dashboard: explore data, generate forecasts, review model and monitoring results."""
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.predict import MIN_HISTORY_HOURS, forecast, load_model  # noqa: E402
from src.utils import ROOT, load_params  # noqa: E402

st.set_page_config(page_title="Household Electricity Forecasting", page_icon="⚡", layout="wide")
PARAMS = load_params()
TARIFF = 0.18  # EUR per kWh, used only for the cost estimate


@st.cache_data
def hourly() -> pd.DataFrame:
    return pd.read_parquet(ROOT / PARAMS["data"]["processed"])


@st.cache_data
def read_json(name: str) -> dict:
    return json.loads((ROOT / "reports" / name).read_text())


data = hourly()
_, meta = load_model()
metrics = read_json("metrics.json")
monitor = read_json("monitoring.json")

st.title("⚡ Household Electricity Consumption Prediction System")
st.caption("MLOps project · UCI Individual Household Electric Power Consumption dataset "
           "(Sceaux, France, Dec 2006 – Nov 2010, 2.07 M one-minute readings)")

tab_fc, tab_data, tab_model, tab_mon, tab_ops = st.tabs(
    ["🔮 Forecast", "📊 Data explorer", "🏆 Model performance", "🩺 Monitoring", "⚙️ MLOps pipeline"])

with tab_fc:
    c1, c2 = st.columns([1, 3])
    with c1:
        test_start = pd.Timestamp(meta["test_period"][0])
        min_d, max_d = (test_start + pd.Timedelta(days=8)).date(), data.index.max().date()
        day = st.date_input("Forecast from (test period)", value=max_d - pd.Timedelta(days=3),
                            min_value=min_d, max_value=max_d)
        hour = st.slider("Starting hour", 0, 23, 0)
        horizon = st.slider("Horizon (hours ahead)", 1, 48, 24)
        origin = pd.Timestamp(day) + pd.Timedelta(hours=hour)
    history = data[data.index < origin]
    fc = forecast(history.iloc[-MIN_HISTORY_HOURS * 2:], horizon)
    actual = data["Global_active_power"].reindex(fc["timestamp"])
    with c1:
        energy = fc["predicted_kw"].sum()
        st.metric("Predicted energy", f"{energy:.2f} kWh")
        st.metric("Estimated cost", f"€{energy * TARIFF:.2f}", help=f"At €{TARIFF}/kWh")
        st.metric("Peak hour", fc.loc[fc["predicted_kw"].idxmax(), "timestamp"].strftime("%a %H:%M"),
                  f"{fc['predicted_kw'].max():.2f} kW", delta_color="off")
        if actual.notna().all():
            st.metric("MAE vs actual", f"{(actual.values - fc['predicted_kw']).abs().mean():.3f} kW")
    with c2:
        past = data["Global_active_power"].loc[origin - pd.Timedelta(hours=72): origin - pd.Timedelta(hours=1)]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(fc["timestamp"]) + list(fc["timestamp"])[::-1],
                                 y=list(fc["upper_kw"]) + list(fc["lower_kw"])[::-1],
                                 fill="toself", fillcolor="rgba(224,123,36,0.15)", line=dict(width=0),
                                 name="95% interval", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=past.index, y=past, name="History", line=dict(color="#1f5fa8")))
        fig.add_trace(go.Scatter(x=fc["timestamp"], y=actual, name="Actual",
                                 line=dict(color="#888", dash="dot")))
        fig.add_trace(go.Scatter(x=fc["timestamp"], y=fc["predicted_kw"], name="Forecast",
                                 line=dict(color="#e07b24", width=3)))
        fig.add_vline(x=origin, line_dash="dash", line_color="black")
        fig.update_layout(height=430, yaxis_title="Active power (kW)", hovermode="x unified",
                          margin=dict(t=20, b=10), legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Forecast table"):
            st.dataframe(fc.assign(actual_kw=actual.values).round(3), use_container_width=True)

with tab_data:
    gran = st.radio("Granularity", ["Hourly", "Daily", "Weekly", "Monthly"], index=1, horizontal=True)
    rule = {"Hourly": "h", "Daily": "D", "Weekly": "W", "Monthly": "MS"}[gran]
    series = data["Global_active_power"].resample(rule).mean()
    st.plotly_chart(px.line(series, labels={"value": "kW", "datetime": ""}, height=320)
                    .update_layout(showlegend=False, margin=dict(t=10)), use_container_width=True)
    c1, c2 = st.columns(2)
    y = data["Global_active_power"]
    prof = y.groupby([y.index.dayofweek >= 5, y.index.hour]).mean().unstack(0)
    prof.columns = ["Weekday", "Weekend"]
    c1.plotly_chart(px.line(prof, title="Daily load profile", labels={"value": "kW", "datetime": "Hour"},
                            height=320), use_container_width=True)
    sub = data[["Sub_metering_1", "Sub_metering_2", "Sub_metering_3", "Sub_metering_other"]].sum() / 1000
    sub.index = ["Kitchen", "Laundry", "Water heater / AC", "Other"]
    c2.plotly_chart(px.pie(values=sub.values, names=sub.index, title="Energy share by sub-meter (kWh)",
                           height=320), use_container_width=True)
    stats = read_json("data_stats.json")
    st.info(f"Raw rows: {stats['raw_rows']:,} · missing: {stats['raw_missing_pct']}% · "
            f"hourly rows after cleaning: {stats['hourly_rows']:,}")

with tab_model:
    st.subheader(f"Production model: `{meta['model_name']}`")
    tm, bm = meta["test_metrics"], meta["baseline_test_metrics"]
    k = st.columns(4)
    k[0].metric("Test MAE", f"{tm['mae']:.3f} kW", f"{(tm['mae'] / bm['mae'] - 1) * 100:.1f}% vs baseline",
                delta_color="inverse")
    k[1].metric("Test RMSE", f"{tm['rmse']:.3f} kW", f"{(tm['rmse'] / bm['rmse'] - 1) * 100:.1f}% vs baseline",
                delta_color="inverse")
    k[2].metric("Test R²", f"{tm['r2']:.3f}", f"{tm['r2'] - bm['r2']:+.3f} vs baseline")
    k[3].metric("Features", len(meta["features"]))
    val = pd.DataFrame(metrics["validation"]).T.round(4)
    st.markdown("**Validation comparison (Jan – May 2010)**")
    st.dataframe(val.style.highlight_min(subset=["mae", "rmse", "mape"], color="#d4edda")
                 .highlight_max(subset=["r2"], color="#d4edda"), use_container_width=True)
    c1, c2 = st.columns(2)
    c1.image(str(ROOT / "reports/figures/fig_feature_importance.png"))
    c2.image(str(ROOT / "reports/figures/fig_residuals.png"))
    st.caption(f"MLflow run id: `{meta['mlflow_run_id']}` · trained {meta['trained_at']} · "
               f"train window {meta['train_period'][0]} → {meta['train_period'][1]}")

with tab_mon:
    if monitor["retraining_recommended"]:
        st.warning(f"Retraining recommended: drift in {', '.join(monitor['drifted_features']) or 'none'}; "
                   f"recent MAE change {monitor['mae_degradation_pct']:+.1f}%")
    else:
        st.success("Model healthy: no significant drift or degradation")
    c1, c2 = st.columns(2)
    psi = pd.Series(monitor["psi"]).sort_values()
    c1.plotly_chart(px.bar(psi, orientation="h", title="Population Stability Index (train vs test)",
                           color=psi > PARAMS["monitoring"]["psi_threshold"],
                           color_discrete_map={True: "#e07b24", False: "#1f5fa8"}, height=350)
                    .update_layout(showlegend=False, xaxis_title="PSI", yaxis_title=""),
                    use_container_width=True)
    preds = pd.read_parquet(ROOT / "reports" / "test_predictions.parquet")
    roll = (preds["actual"] - preds["predicted"]).abs().rolling(24 * 7).mean().dropna()
    c2.plotly_chart(px.line(roll, title="Rolling 7-day MAE on test period", height=350)
                    .add_hline(y=meta["test_metrics"]["mae"], line_dash="dash")
                    .update_layout(showlegend=False, yaxis_title="MAE (kW)", xaxis_title=""),
                    use_container_width=True)

with tab_ops:
    st.graphviz_chart("""
    digraph { rankdir=LR; node [shape=box style="rounded,filled" fillcolor="#eef3fa" fontname=Helvetica];
      A [label="UCI dataset\\n(2.07M rows)"]; B [label="Ingestion &\\nschema check"];
      C [label="Cleaning &\\nhourly resample"]; D [label="Feature\\nengineering"];
      E [label="Training 5 models\\n+ MLflow tracking"]; F [label="Model registry\\n(best model)"];
      G [label="FastAPI service\\n(Docker)"]; H [label="Streamlit\\ndashboard"];
      I [label="Monitoring\\n(PSI, MAE)"]; J [label="GitHub Actions\\nCI: tests + build"];
      A->B->C->D->E->F->G->H; G->I; I->E [label="retrain trigger" style=dashed]; J->E [style=dashed];
    }""")
    st.markdown("""
| Stage | Tool | Artifact |
|---|---|---|
| Versioned pipeline | `dvc.yaml`, `run_pipeline.py` | processed data, features |
| Experiment tracking | MLflow (SQLite backend) | runs, params, metrics, registered model |
| Serving | FastAPI + Uvicorn, Docker | `/predict`, `/health`, `/metrics` |
| CI | GitHub Actions, pytest | test report, Docker image |
| Monitoring | PSI drift + rolling MAE | `reports/monitoring.json` |
""")
