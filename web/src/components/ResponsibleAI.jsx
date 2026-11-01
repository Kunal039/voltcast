import { useMemo } from "react";
import Plot, { COLORS } from "./Plot.jsx";
import Section from "./Section.jsx";

const POINTS = [
  ["Fairness & bias", "Error is compared across groups of hours (chart). Forecasts are weakest in evening peaks and high-use hours, and the model learned from one French household, so it must be re-validated before use on other homes or climates."],
  ["Transparency", "Feature importance, a model card, and numbers anyone can reproduce with dvc repro. Forecasts carry a 95% interval instead of a single figure."],
  ["Privacy", "Meter data can reveal when people are home. Only a public anonymised dataset is used, and the prediction log stores forecasts, never the readings sent in."],
  ["Security", "Every reading is validated, request size is limited to 2 MB, and rejected requests are counted in Prometheus so misuse shows up in Grafana."],
];

export default function ResponsibleAI({ D }) {
  const F = D.fairness;
  const { data, layout } = useMemo(() => {
    const groups = Object.entries(F.slices);
    return {
      data: groups.map(([name, g], i) => ({
        type: "bar", x: Object.keys(g), y: Object.values(g).map((v) => v.mae), name,
        xaxis: `x${i + 1}`, yaxis: `y${i + 1}`, marker: { color: COLORS.blue }, showlegend: false,
      })),
      layout: {
        hovermode: "closest",
        grid: { rows: 1, columns: groups.length, pattern: "independent", xgap: 0.12 },
        margin: { t: 40, r: 12, b: 90, l: 50 },
        annotations: groups.map(([name], i) => ({
          text: `<b>${name}</b>`, xref: `x${i + 1} domain`, yref: `y${i + 1} domain`, x: 0.5, y: 1.13, showarrow: false,
        })),
        shapes: groups.map((_, i) => ({
          type: "line", xref: `x${i + 1} domain`, yref: `y${i + 1}`, x0: 0, x1: 1, y0: F.overall_mae, y1: F.overall_mae,
          line: { color: COLORS.orange, dash: "dash" },
        })),
        ...Object.fromEntries(groups.flatMap((_, i) => [
          [`xaxis${i ? i + 1 : ""}`, { tickangle: -30, gridcolor: "#eef0f3" }],
          [`yaxis${i ? i + 1 : ""}`, { range: [0, 0.5], gridcolor: "#eef0f3", showticklabels: i === 0, title: i === 0 ? "MAE (kW)" : "" }],
        ])),
      },
    };
  }, [D]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <Section id="ethics" title="Responsible AI" sub="Ethical considerations from the course: fairness, transparency, privacy and security.">
      <div className="card">
        <Plot data={data} layout={layout} height={330} />
        <p className="note">Test MAE for each group of hours. Dashed line: overall MAE ({F.overall_mae.toFixed(3)} kW).</p>
      </div>
      <div className="stack gap">{POINTS.map(([t, d]) => <div key={t}><b>{t}</b>{d}</div>)}</div>
    </Section>
  );
}
