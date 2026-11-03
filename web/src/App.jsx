import { useEffect, useMemo, useState } from "react";
import Plot, { COLORS } from "./components/Plot.jsx";
import Forecast from "./components/Forecast.jsx";
import ModelPerformance from "./components/ModelPerformance.jsx";
import PeakAlerts from "./components/PeakAlerts.jsx";
import ResponsibleAI from "./components/ResponsibleAI.jsx";
import Section, { LABEL, hbar } from "./components/Section.jsx";

const REPO = "https://github.com/Kunal039/voltcast";
const DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

function Kpis({ D }) {
  const T = D.model.test_metrics, B = D.model.baseline_test_metrics;
  const pct = (a, b) => ((1 - a / b) * 100).toFixed(1);
  const items = [
    ["Test MAE", `${T.mae.toFixed(3)} kW`, `${pct(T.mae, B.mae)}% better than baseline`],
    ["Test RMSE", `${T.rmse.toFixed(3)} kW`, `${pct(T.rmse, B.rmse)}% better than baseline`],
    ["Test R²", T.r2.toFixed(3), `+${(T.r2 - B.r2).toFixed(3)} vs baseline`],
    ["Training data", `${(D.stats.raw_rows / 1e6).toFixed(2)} M`, "one-minute readings"],
    ["Model", LABEL[D.model.model_name], `${D.model.n_features} engineered features`],
  ];
  return (
    <div className="kpis">
      {items.map(([l, v, d]) => (
        <div className="kpi" key={l}><div className="label">{l}</div><div className="value">{v}</div><div className="delta">{d}</div></div>
      ))}
    </div>
  );
}

function DataExplorer({ D }) {
  const charts = useMemo(() => ({
    daily: [{ x: D.daily.x, y: D.daily.y, line: { color: COLORS.blue, width: 1 }, name: "Daily mean kW" }],
    heat: [{ type: "heatmap", z: D.heatmap, x: [...Array(24).keys()], y: DAYS, colorscale: "YlOrRd",
      reversescale: true, colorbar: { title: "kW", thickness: 10 } }],
    profile: [
      { y: D.profile.weekday, name: "Weekday", line: { color: COLORS.blue } },
      { y: D.profile.weekend, name: "Weekend", line: { color: COLORS.orange } },
    ],
  }), [D]);
  const s = D.stats;
  return (
    <Section title="Data explorer"
      sub={`${s.raw_rows.toLocaleString("en-US")} one-minute readings (${s.raw_missing_pct}% missing) from a house in Sceaux, France, cleaned into ${s.hourly_rows.toLocaleString("en-US")} hourly records.`}>
      <div className="card"><Plot data={charts.daily} title="Daily mean active power, Dec 2006 – Nov 2010" height={300}
        layout={{ xaxis: { type: "date", gridcolor: "#eef0f3" } }} /></div>
      <div className="grid2 gap">
        <div className="card"><Plot data={charts.heat} title="Average kW by weekday and hour"
          layout={{ hovermode: "closest", yaxis: { autorange: "reversed" } }} /></div>
        <div className="card"><Plot data={charts.profile} title="Daily load profile (kW by hour)" /></div>
      </div>
    </Section>
  );
}

function Monitoring({ D }) {
  const M = D.monitoring;
  const charts = useMemo(() => {
    const psi = Object.entries(M.psi).sort((a, b) => a[1] - b[1]);
    return {
      psi: [{ type: "bar", orientation: "h", x: psi.map((p) => p[1]), y: psi.map((p) => p[0]),
        marker: { color: psi.map((p) => (p[1] > 0.2 ? COLORS.orange : COLORS.blue)) } }],
      sub: [{ type: "pie", labels: Object.keys(D.submeter), values: Object.values(D.submeter), hole: 0.45,
        marker: { colors: [COLORS.blue, COLORS.orange, COLORS.green, "#c9ced6"] } }],
    };
  }, [D]); // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <Section title="Monitoring" sub="Data drift between training and recent data is measured with the Population Stability Index (PSI > 0.2 = significant).">
      {M.retraining_recommended && (
        <div className="alert"><b>Retraining recommended.</b> Drift detected in {M.drifted_features.join(", ")}; recent
          30-day MAE is {M.mae_degradation_pct}% above the reference. The training data is winter-heavy while the test
          period is summer, so this is genuine seasonal drift.</div>
      )}
      <div className="grid2">
        <div className="card"><Plot data={charts.psi} title="PSI: training vs test data" layout={{ ...hbar,
          shapes: [{ type: "line", x0: 0.2, x1: 0.2, yref: "paper", y0: 0, y1: 1, line: { color: "#b42318", dash: "dash" } }] }} /></div>
        <div className="card"><Plot data={charts.sub} title="Energy share by sub-meter (kWh)" layout={{ hovermode: "closest" }} /></div>
      </div>
    </Section>
  );
}

function Pipeline() {
  const stack = [
    ["Pipeline & versioning", "dvc.yaml · dvc.lock · Git"], ["Tracking & registry", "MLflow"],
    ["Serving", "Flask REST API · gunicorn"], ["Deployment", "Docker · Kubernetes · Vercel"],
    ["CI / CT", "GitHub Actions · pytest"], ["Monitoring", "Prometheus · Grafana · PSI drift"],
  ];
  return (
    <Section title="MLOps pipeline" sub="Every stage is a tested Python module driven by one config file; GitHub Actions re-runs it on each push and weekly.">
      <div className="card">
        <img className="arch" src="/img/architecture.png"
          alt="Architecture: data ingestion, preprocessing, features, training with MLflow, registry, Flask API, web app, Prometheus and Grafana monitoring and CI/CT" />
        <div className="stack">{stack.map(([t, d]) => <div key={t}><b>{t}</b>{d}</div>)}</div>
      </div>
    </Section>
  );
}

const ENDPOINTS = [
  ["GET", "/api/health", "Liveness check and loaded model"],
  ["GET", "/api/model-info", "Model, hyper-parameters, training window, test metrics"],
  ["GET", "/api/forecast?origin=&horizon=", "Forecast from a point in the test period, with actual values"],
  ["POST", "/api/predict", "Forecast from your own 169+ hourly readings (JSON)"],
  ["GET", "/api/sample-request", "Ready-made body for POST /api/predict"],
  ["GET", "/api/metrics", "Prometheus metrics: requests, latency, forecasts, peak alerts"],
];

function ApiUsage() {
  const host = window.location.origin;
  return (
    <Section id="api" title="REST API (Flask)" sub="The same models are served by a Flask REST API. Prometheus scrapes /api/metrics; Grafana shows the dashboard (see docker-compose.yml).">
      <pre>{`curl "${host}/api/forecast?origin=2010-11-20T08:00&horizon=6"
curl -s ${host}/api/sample-request > req.json
curl -X POST ${host}/api/predict -H "Content-Type: application/json" -d @req.json`}</pre>
      <table className="endpoints">
        <tbody>
          {ENDPOINTS.map(([m, path, d]) => <tr key={path}><td><code>{m}</code></td><td><code>{path}</code></td><td>{d}</td></tr>)}
        </tbody>
      </table>
    </Section>
  );
}

export default function App() {
  const [D, setD] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch("/data/dashboard.json")
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(r.statusText))))
      .then(setD)
      .catch((e) => setError(e.message));
  }, []);

  return (
    <>
      <header>
        <div className="wrap">
          <h1>⚡ VoltCast</h1>
          <p className="tagline">Household Electricity Consumption Prediction System</p>
          <p>An end-to-end MLOps project that forecasts a household's hourly electricity use for the next 1–48 hours,
            trained on 2.07 million smart-meter readings from the UCI Individual Household Electric Power Consumption dataset.</p>
          <div className="links">
            <a href="#forecast">Try a forecast</a>
            <a href="#api">REST API</a>
            <a href={REPO}>GitHub repository</a>
          </div>
        </div>
      </header>
      <main className="wrap">
        {error && <div className="alert">Could not load dashboard data: {error}</div>}
        {!D ? <p className="sub loading">Loading dashboard…</p> : (
          <>
            <Kpis D={D} />
            <section id="forecast">
              <h2>Live forecast</h2>
              <p className="sub">Pick a moment in the held-out test period (Jun–Nov 2010). The deployed model forecasts
                the following hours, and the chart shows what actually happened.</p>
              <Forecast range={D.forecast_range} />
            </section>
            <ModelPerformance D={D} />
            <PeakAlerts D={D} />
            <DataExplorer D={D} />
            <Monitoring D={D} />
            <ResponsibleAI D={D} />
            <Pipeline />
            <ApiUsage />
          </>
        )}
      </main>
      <footer>
        <div className="wrap">VoltCast · MLOps project · Chandigarh University · Kunal Mittal, Ansh Kumar, Vishal Kumar Shrivastava ·
          Data: G. Hebrail &amp; A. Berard, UCI Machine Learning Repository (CC BY 4.0)</div>
      </footer>
    </>
  );
}
