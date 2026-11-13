import { useMemo } from "react";
import Plot, { COLORS } from "./Plot.jsx";
import Section from "./Section.jsx";

export default function PeakAlerts({ D }) {
  const C = D.classification;
  const T = C.test;
  const [tn, fp] = T.confusion_matrix[0];
  const [fn, tp] = T.confusion_matrix[1];

  const charts = useMemo(() => ({
    cm: [{
      type: "heatmap", z: [[fn, tp], [tn, fp]], x: ["Predicted normal", "Predicted peak"], y: ["Actual peak", "Actual normal"],
      colorscale: [[0, "#f1f5fb"], [1, "#7fa6d6"]], showscale: false, text: [[fn, tp], [tn, fp]].map((r) => r.map((v) => v.toLocaleString("en-US"))),
      texttemplate: "%{text}", textfont: { size: 16, color: "#14181f" },
    }],
    roc: [
      { x: C.roc.fpr, y: C.roc.tpr, name: `Selected model, AUC ${T.roc_auc.toFixed(3)}`, line: { color: COLORS.orange, width: 2.5 } },
      { x: [0, 1], y: [0, 1], name: "Random guess", line: { color: COLORS.grey, dash: "dash" } },
    ],
  }), [D]); // eslint-disable-line react-hooks/exhaustive-deps

  const kpis = [["Accuracy", T.accuracy], ["Precision", T.precision], ["Recall", T.recall], ["F1-score", T.f1], ["ROC-AUC", T.roc_auc]];
  return (
    <Section id="peaks" title="Peak-hour alerts (classification)"
      sub={`A second model flags hours above ${C.threshold_kw.toFixed(2)} kW (the busiest 25% of training hours). A default decision tree, a GridSearchCV-tuned decision tree and a random forest were compared on validation F1; these are the scores of the selected one (${C.selected.replace(/_/g, " ")}) on the held-out test period.`}>
      <div className="kpis inline">
        {kpis.map(([l, v]) => (
          <div className="kpi" key={l}><div className="label">{l}</div><div className="value">{(v * 100).toFixed(1)}%</div></div>
        ))}
      </div>
      <div className="grid2 gap">
        <div className="card">
          <Plot data={charts.cm} title="Confusion matrix (test)" height={320}
            layout={{ hovermode: "closest", margin: { t: 36, r: 12, b: 50, l: 100 } }} />
        </div>
        <div className="card">
          <Plot data={charts.roc} title="ROC curve (test)" height={320}
            layout={{ hovermode: "closest", legend: { x: 0.35, y: 0.08, bgcolor: "rgba(255,255,255,.8)" },
              xaxis: { title: "False positive rate", range: [0, 1] },
              yaxis: { title: "True positive rate", range: [0, 1.02] } }} />
        </div>
      </div>
      <p className="note">High recall, moderate precision: the alert catches {(T.recall * 100).toFixed(0)}% of real peak hours at the cost
        of some false alarms, the safer trade-off when the goal is to avoid unexpected peaks. For comparison, flagging hours where the
        regression forecast exceeds the threshold catches only{" "}
        {(C.forecast_threshold_rule_test.recall * 100).toFixed(0)}% of peaks. The live forecast uses this classifier for every forecast hour.</p>
    </Section>
  );
}
