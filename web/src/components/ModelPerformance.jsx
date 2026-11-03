import { useMemo } from "react";
import Plot, { COLORS } from "./Plot.jsx";
import Section, { LABEL, hbar, isBaseline } from "./Section.jsx";

const grouped = { hovermode: "closest", barmode: "group", xaxis: { tickangle: -25 },
  margin: { t: 36, r: 12, b: 90, l: 50 }, legend: { orientation: "h", y: 1.1, x: 0 } };

export default function ModelPerformance({ D }) {
  const V = D.metrics.validation;
  const tuning = D.metrics.tuning;
  const names = Object.keys(V);
  const ml = names.filter((n) => !isBaseline(n));
  const best = D.model.model_name; // the deployed model (best syllabus algorithm on validation)
  const dt = tuning.decision_tree;

  const charts = useMemo(() => {
    const tn = Object.keys(tuning);
    return {
      fit: [
        { type: "bar", x: ml.map((n) => LABEL[n]), y: ml.map((n) => V[n].train_mae), name: "Training MAE", marker: { color: COLORS.grey } },
        { type: "bar", x: ml.map((n) => LABEL[n]), y: ml.map((n) => V[n].mae), name: "Validation MAE", marker: { color: COLORS.blue } },
      ],
      tuning: [
        { type: "bar", x: tn.map((n) => LABEL[n]), y: tn.map((n) => tuning[n].default_val.mae), name: "Default", marker: { color: COLORS.grey } },
        { type: "bar", x: tn.map((n) => LABEL[n]), y: tn.map((n) => tuning[n].tuned_val.mae), name: "Tuned (GridSearchCV)", marker: { color: COLORS.orange } },
      ],
      test: [
        { x: D.test_window.x, y: D.test_window.actual, name: "Actual", line: { color: COLORS.grey, width: 1.5 } },
        { x: D.test_window.x, y: D.test_window.predicted, name: "Predicted (next hour)", line: { color: COLORS.orange, width: 1.5 } },
      ],
      imp: [{ type: "bar", orientation: "h", x: D.importance.values, y: D.importance.names, marker: { color: COLORS.blue } }],
    };
  }, [D]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <Section title="Model performance"
      sub="Nine models trained on 2006–2009 and compared on Jan–May 2010 in MLflow. Only syllabus algorithms (linear and ridge regression, decision tree, KNN, SVM, random forest) may be deployed; gradient boosting and two naive baselines are kept as reference points. The deployed model (highlighted) is refit and scored once on the untouched test period.">
      <div className="grid2">
        <div className="card">
          <table>
            <thead><tr><th>Model</th><th>Train MAE</th><th>Val MAE</th><th>RMSE</th><th>R²</th></tr></thead>
            <tbody>
              {names.map((n) => (
                <tr key={n} className={n === best ? "best" : ""}>
                  <td>{LABEL[n]}</td><td>{V[n].train_mae.toFixed(3)}</td><td>{V[n].mae.toFixed(3)}</td>
                  <td>{V[n].rmse.toFixed(3)}</td><td>{V[n].r2.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="card">
          <Plot data={charts.fit} title="Over-fitting check: training vs validation MAE (kW)" height={340} layout={grouped} />
          <p className="note">The default decision tree has zero training error but the worst validation error: it memorises the training set.</p>
        </div>
      </div>
      <div className="grid2 gap">
        <div className="card">
          <Plot data={charts.tuning} title="Tuning with GridSearchCV (time-series folds)" height={340} layout={grouped} />
          <p className="note">Limiting the tree (max_depth {String(dt.best_params.max_depth)}, min_samples_leaf {dt.best_params.min_samples_leaf})
            cuts its validation MAE from {dt.default_val.mae.toFixed(3)} to {dt.tuned_val.mae.toFixed(3)} kW.</p>
        </div>
        <div className="card"><Plot data={charts.imp} title="Top features (permutation importance)" layout={hbar} /></div>
      </div>
      <div className="card gap">
        <Plot data={charts.test} title="Test set: last 14 days, next-hour forecasts" layout={{ xaxis: { type: "date", gridcolor: "#eef0f3" } }} />
      </div>
    </Section>
  );
}
