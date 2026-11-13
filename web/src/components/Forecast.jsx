import { useCallback, useEffect, useMemo, useState } from "react";
import Plot, { COLORS } from "./Plot.jsx";

const pad = (h) => String(h).padStart(2, "0");

export default function Forecast({ range }) {
  const [lo, hi] = range.map((s) => s.slice(0, 10));
  const [day, setDay] = useState("2010-11-20");
  const [hour, setHour] = useState(8);
  const [horizon, setHorizon] = useState(24);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const origin = `${day}T${pad(hour)}:00`;

  const run = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const r = await fetch(`/api/forecast?origin=${origin}&horizon=${horizon}`);
      const body = await r.json();
      if (!r.ok) throw new Error(body.detail || r.statusText);
      setResult({ ...body, origin });
    } catch (e) {
      setError(`Forecast failed: ${e.message}`);
    } finally {
      setLoading(false);
    }
  }, [origin, horizon]);

  // Show a forecast as soon as the page opens.
  useEffect(() => { run(); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const chart = useMemo(() => {
    if (!result) return null;
    const f = result.forecast;
    const xs = f.map((p) => p.timestamp);
    return {
      data: [
        { x: [...xs, ...[...xs].reverse()], y: [...f.map((p) => p.upper_kw), ...f.map((p) => p.lower_kw).reverse()],
          fill: "toself", fillcolor: "rgba(224,123,36,.15)", line: { width: 0 }, name: "95% interval", hoverinfo: "skip" },
        { x: result.history.map((p) => p.timestamp), y: result.history.map((p) => p.kw), name: "History",
          line: { color: COLORS.blue, width: 1.6 } },
        { x: xs, y: f.map((p) => p.actual_kw), name: "Actual", line: { color: "#6b7280", dash: "dot", width: 1.6 } },
        { x: xs, y: f.map((p) => p.predicted_kw), name: "Forecast", line: { color: COLORS.orange, width: 3 } },
        { x: f.filter((p) => p.peak_alert).map((p) => p.timestamp), y: f.filter((p) => p.peak_alert).map((p) => p.predicted_kw),
          name: "Peak-hour alert", mode: "markers", marker: { color: "#b42318", size: 9, symbol: "triangle-up" },
          customdata: f.filter((p) => p.peak_alert).map((p) => `${(p.peak_probability * 100).toFixed(0)}% peak probability`),
          hovertemplate: "Peak-hour alert<br>%{customdata}<extra></extra>" },
      ],
      layout: {
        margin: { t: 20, r: 12, b: 64, l: 50 },
        xaxis: { type: "date", gridcolor: "#eef0f3" },
        yaxis: { title: "kW", gridcolor: "#eef0f3" },
        shapes: [{ type: "line", x0: result.origin, x1: result.origin, yref: "paper", y0: 0, y1: 1,
          line: { dash: "dash", color: "#14181f" } }],
      },
    };
  }, [result]);

  const stats = useMemo(() => {
    if (!result) return [];
    const f = result.forecast;
    const peak = f.reduce((a, b) => (a.predicted_kw > b.predicted_kw ? a : b));
    const paired = f.filter((p) => p.actual_kw !== null);
    const mae = paired.reduce((s, p) => s + Math.abs(p.actual_kw - p.predicted_kw), 0) / paired.length;
    const when = new Date(peak.timestamp).toLocaleString("en-GB", { weekday: "short", hour: "2-digit", minute: "2-digit" });
    return [
      [`${result.predicted_energy_kwh.toFixed(2)} kWh`, "predicted energy"],
      [`${when} · ${peak.predicted_kw.toFixed(2)} kW`, "predicted peak"],
      [paired.length ? `${mae.toFixed(3)} kW` : "–", "MAE vs actual"],
      [`${f.filter((p) => p.peak_alert).length} of ${f.length} hours`, "peak-hour alerts"],
    ];
  }, [result]);

  return (
    <div className="card fc">
      <div>
        <label htmlFor="day">Forecast from</label>
        <input type="date" id="day" min={lo} max={hi} value={day} onChange={(e) => setDay(e.target.value)} />
        <label htmlFor="hour">Starting hour: <b>{pad(hour)}:00</b></label>
        <input type="range" id="hour" min="0" max="23" value={hour} onChange={(e) => setHour(+e.target.value)} />
        <label htmlFor="horizon">Hours ahead: <b>{horizon}</b></label>
        <input type="range" id="horizon" min="1" max="48" value={horizon} onChange={(e) => setHorizon(+e.target.value)} />
        <button onClick={run} disabled={loading}>{loading ? "Forecasting…" : "Run forecast"}</button>
        {error && <div className="err">{error}</div>}
        <div className="stats">
          {stats.map(([v, s]) => <div className="stat" key={s}><b>{v}</b><span>{s}</span></div>)}
        </div>
      </div>
      {chart ? <Plot data={chart.data} layout={chart.layout} height={420} /> : <div className="placeholder">Loading forecast…</div>}
    </div>
  );
}
