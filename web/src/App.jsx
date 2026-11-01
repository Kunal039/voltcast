import { useEffect, useMemo, useState } from "react";
import Plot, { COLORS } from "./components/Plot.jsx";
import Forecast from "./components/Forecast.jsx";

const REPO = "https://github.com/Kunal039/voltcast";
const DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
const pretty = (s) => s.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
const isBaseline = (n) => /baseline|naive/.test(n);
const hbar = { margin: { t: 36, r: 12, b: 40, l: 160 }, hovermode: "closest",
  xaxis: { type: "linear", gridcolor: "#eef0f3" }, yaxis: { type: "category", automargin: true } };

function Section({ title, sub, children }) {
  return (
    <section>
      <h2>{title}</h2>
      {sub && <p className="sub">{sub}</p>}
      {children}
    </section>
  );
}

function Kpis({ D }) {
  const T = D.model.test_metrics, B = D.model.baseline_test_metrics;
  const pct = (a, b) => ((1 - a / b) * 100).toFixed(1);
  const items = [
    ["Test MAE", `${T.mae.toFixed(3)} kW`, `${pct(T.mae, B.mae)}% better than baseline`],
    ["Test RMSE", `${T.rmse.toFixed(3)} kW`, `${pct(T.rmse, B.rmse)}% better than baseline`],
    ["Test R²", T.r2.toFixed(3), `+${(T.r2 - B.r2).toFixed(3)} vs baseline`],
    ["Training data", `${(D.stats.raw_rows / 1e6).toFixed(2)} M`, "one-minute readings"],
    ["Model", "Gradient Boosting", `${D.model.n_features} engineered features`],
  ];
  return (
    <div className="kpis">
      {items.map(([l, v, d]) => (
        <div className="kpi" key={l}><div className="label">{l}</div><div className="value">{v}</div><div className="delta">{d}</div></div>
      ))}
    </div>
  );
}

function ModelPerformance({ D }) {
  const V = D.metrics.validation;
  const names = Object.keys(V);
  const best = names.filter((n) => !isBaseline(n)).reduce((a, b) => (V[a].rmse < V[b].rmse ? a : b));
  const charts = useMemo(() => ({
    mae: [{ type: "bar", x: names.map((n) => pretty(n).replace(/ /g, "<br>")), y: names.map((n) => V[n].mae),
      marker: { color: names.map((n) => (n === best ? COLORS.orange : isBaseline(n) ? COLORS.grey : COLORS.blue)) } }],
    test: [
      { x: D.test_window.x, y: D.test_window.actual, name: "Actual", line: { color: COLORS.grey, width: 1.5 } },
      { x: D.test_window.x, y: D.test_window.predicted, name: "Predicted (next hour)", line: { color: COLORS.orange, width: 1.5 } },
    ],
    imp: [{ type: "bar", orientation: "h", x: D.importance.values, y: D.importance.names, marker: { color: COLORS.blue } }],
  }), [D]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <Section title="Model performance"
      sub="Five candidates trained on 2006–2009, compared on Jan–May 2010 (validation) and tracked in MLflow. The winner is refit and scored once on the untouched test period.">
      <div className="grid2">
        <div className="card">
          <table>
            <thead><tr><th>Model</th><th>MAE</th><th>RMSE</th><th>R²</th><th>Fit (s)</th></tr></thead>
            <tbody>
              {names.map((n) => (
                <tr key={n} className={n === best ? "best" : ""}>
                  <td>{pretty(n)}</td><td>{V[n].mae.toFixed(3)}</td><td>{V[n].rmse.toFixed(3)}</td>
                  <td>{V[n].r2.toFixed(3)}</td><td>{V[n].fit_seconds.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="card">
          <Plot data={charts.mae} title="Validation MAE (kW, lower is better)" height={300}
            layout={{ hovermode: "closest", xaxis: { tickangle: 0 }, margin: { t: 36, r: 12, b: 70, l: 50 } }} />
        </div>
      </div>
      <div className="grid2 gap">
        <div className="card"><Plot data={charts.test} title="Test set: last 14 days, next-hour forecasts"
          layout={{ xaxis: { type: "date", gridcolor: "#eef0f3" } }} /></div>
        <div className="card"><Plot data={charts.imp} title="Top features (permutation importance)" layout={hbar} /></div>
      </div>
    </Section>
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
      sub={`${s.raw_rows.toLocaleString()} one-minute readings (${s.raw_missing_pct}% missing) from a house in Sceaux, France, cleaned into ${s.hourly_rows.toLocaleString()} hourly records.`}>
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
    ["Pipeline", "run_pipeline.py · dvc.yaml"], ["Tracking & registry", "MLflow"],
    ["Serving", "FastAPI · Docker · Vercel"], ["Web", "React · Vite · Plotly"],
    ["CI / CT", "GitHub Actions · pytest"], ["Monitoring", "PSI drift · rolling MAE"],
  ];
  return (
    <Section title="MLOps pipeline" sub="Every stage is a tested Python module driven by one config file; GitHub Actions re-runs it on each push and weekly.">
      <div className="card">
        <img className="arch" src="/img/architecture.png"
          alt="Architecture: data ingestion, preprocessing, features, training with MLflow, registry, FastAPI, dashboard, monitoring and CI/CT" />
        <div className="stack">{stack.map(([t, d]) => <div key={t}><b>{t}</b>{d}</div>)}</div>
      </div>
    </Section>
  );
}

function ApiUsage() {
  const host = window.location.origin;
  return (
    <Section title="Use the API" sub="The same model is exposed as a REST API.">
      <pre>{`curl "${host}/api/forecast?origin=2010-11-20T08:00&horizon=6"
curl -s ${host}/api/sample-request > req.json
curl -X POST ${host}/api/predict -H "Content-Type: application/json" -d @req.json`}</pre>
      <p className="sub">Full interactive docs at <a href="/api/docs">/api/docs</a>.</p>
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
            <a href="/api/docs">API docs</a>
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
            <DataExplorer D={D} />
            <Monitoring D={D} />
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
