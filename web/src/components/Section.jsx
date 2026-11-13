export const LABEL = {
  persistence_baseline: "Persistence baseline", seasonal_naive_24h: "Seasonal naive (24 h)",
  linear_regression: "Linear regression", ridge_regression: "Ridge regression", decision_tree: "Decision tree",
  knn: "K-nearest neighbours", svm: "SVM (SVR)", random_forest: "Random forest",
};

export const isBaseline = (n) => /baseline|naive/.test(n);

export const hbar = { margin: { t: 36, r: 12, b: 40, l: 160 }, hovermode: "closest",
  xaxis: { type: "linear", gridcolor: "#eef0f3" }, yaxis: { type: "category", automargin: true } };

export default function Section({ id, title, sub, children }) {
  return (
    <section id={id}>
      <h2>{title}</h2>
      {sub && <p className="sub">{sub}</p>}
      {children}
    </section>
  );
}
