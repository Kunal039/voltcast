"""Builds the presentation explanation PDF (talking points for the project presentation and viva)."""
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, ListFlowable, ListItem, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "VoltCast_Project_Explanation.pdf"
LIVE = "https://voltcast-forecast.vercel.app"
REPO = "https://github.com/Kunal039/voltcast"

meta = json.loads((ROOT / "models" / "metadata.json").read_text())
clf = json.loads((ROOT / "reports" / "classification.json").read_text())
metrics = json.loads((ROOT / "reports" / "metrics.json").read_text())
mon = json.loads((ROOT / "reports" / "monitoring.json").read_text())
stats = json.loads((ROOT / "reports" / "data_stats.json").read_text())
analysis_path = ROOT / "reports" / "analysis.json"
AN = json.loads(analysis_path.read_text()) if analysis_path.exists() else None
T, B, CT = meta["test_metrics"], meta["baseline_test_metrics"], clf["test"]
V, TUN = metrics["validation"], metrics["tuning"]
RF_TEST = AN["test_all_models"][meta["model_name"]] if AN else None
LBL = {"persistence_baseline": "Persistence baseline", "seasonal_naive_24h": "Seasonal naive (24 h)",
       "linear_regression": "Linear regression", "ridge_regression": "Ridge regression", "decision_tree": "Decision tree",
       "knn": "K-nearest neighbours", "svm": "SVM (SVR)", "random_forest": "Random forest"}
CLF = {"decision_tree_default": "Decision tree (default)", "decision_tree_tuned": "Decision tree (tuned)",
       "random_forest": "Random forest classifier"}

BLUE, LIGHT, GREY = colors.HexColor("#1f5fa8"), colors.HexColor("#e8f0fb"), colors.HexColor("#5d6675")
ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=15, textColor=BLUE, spaceBefore=10, spaceAfter=6)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=11.5, textColor=colors.black, spaceBefore=8, spaceAfter=4)
BODY = ParagraphStyle("B", parent=ss["BodyText"], fontSize=10, leading=14, spaceAfter=5)
QUOTE = ParagraphStyle("Q", parent=BODY, leftIndent=10, borderPadding=6, backColor=LIGHT, textColor=colors.black,
                       spaceBefore=4, spaceAfter=8)
SMALL = ParagraphStyle("S", parent=BODY, fontSize=8.5, leading=11, textColor=GREY)
CELL = ParagraphStyle("C", parent=BODY, fontSize=8.8, leading=11, spaceAfter=0)


def p(text, style=BODY):
    return Paragraph(text, style)


def bullets(items):
    return ListFlowable([ListItem(p(i), leftIndent=12) for i in items], bulletType="bullet", start="•", leftIndent=12)


def table(rows, widths, header=True):
    data = [[p(str(c), CELL) for c in r] for r in rows]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b8c2cf")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), LIGHT))
    t.setStyle(TableStyle(style))
    return t


def link(url):
    return f'<link href="{url}" color="#1f5fa8"><u>{url}</u></link>'


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(GREY)
    canvas.drawString(2 * cm, 1.2 * cm, f"VoltCast - project explanation    Live: {LIVE}    Code: {REPO}")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, str(doc.page))
    canvas.restoreState()


story = []
# ---------------------------------------------------------------- cover
story += [Spacer(1, 3 * cm),
          p("VoltCast", ParagraphStyle("T", parent=ss["Title"], fontSize=34, leading=42, textColor=BLUE, spaceAfter=10)),
          p("Household Electricity Consumption Prediction System", ParagraphStyle("ST", parent=ss["Title"], fontSize=16)),
          Spacer(1, 0.4 * cm),
          p("Project explanation for the presentation and viva", ParagraphStyle("c", parent=BODY, alignment=TA_CENTER, fontSize=12)),
          p("MLOps-1 (25CSH-320) Mini Project, Chandigarh University", ParagraphStyle("c2", parent=BODY, alignment=TA_CENTER)),
          Spacer(1, 1.2 * cm),
          table([["Live deployment", link(LIVE)], ["GitHub repository", link(REPO)],
                 ["Team", "Kunal Mittal (24BCS12354), Ansh Kumar (24BCS12097), Vishal Kumar Shrivastava (24BCS12629)"]],
                [4.2, 12.0], header=False),
          Spacer(1, 1.2 * cm),
          table([["Result", "Value"],
                 ["Forecasting model", f"{LBL[meta['model_name']]} (syllabus algorithm)"],
                 ["Forecast error (test MAE)", f"{T['mae']:.3f} kW, {(1 - T['mae'] / B['mae']) * 100:.0f}% lower than the baseline"],
                 ["R² (test)", f"{T['r2']:.3f}"],
                 ["Accuracy (1 - WAPE)", f"{RF_TEST['accuracy_1_minus_wape'] * 100:.1f}%" if RF_TEST else "see report"],
                 ["Peak-hour alert model", CLF[clf['selected']]],
                 ["Peak-hour alert accuracy / ROC-AUC", f"{CT['accuracy'] * 100:.1f}% / {CT['roc_auc']:.3f}"]],
                [7.0, 9.2]),
          PageBreak()]

# ---------------------------------------------------------------- 1-3
story += [p("1. One-line summary (start with this)", H1),
          p("\"VoltCast is an MLOps project that predicts how much electricity a household will use over the next 1 to "
            "48 hours, and warns about peak-usage hours. It is not just a model in a notebook: it is a complete, "
            "automated system that is versioned, tested, deployed on the cloud and monitored.\"", QUOTE),
          p(f"Live: {link(LIVE)}<br/>Code: {link(REPO)}"),
          p("2. The problem (why it matters)", H1),
          bullets(["Households get a monthly bill but no idea how much electricity they will use tomorrow, or when the "
                   "peaks will be.",
                   "Electricity companies need short-term, household-level forecasts to manage peak demand; India is "
                   "installing about 250 million smart meters, so this data is becoming common.",
                   "A model built once and left alone becomes inaccurate over time (for example when seasons change). "
                   "That is why <b>MLOps</b> is needed: automation, testing, monitoring and retraining."]),
          p("3. The data", H1),
          bullets([f"<b>UCI Individual Household Electric Power Consumption</b> (public): {stats['raw_rows']:,} "
                   "one-minute readings from one house near Paris, Dec 2006 - Nov 2010.",
                   "Columns: active power (kW), reactive power, voltage, current, and 3 sub-meters (kitchen, laundry, "
                   "water heater / AC).",
                   f"{stats['raw_missing_pct']}% of readings were missing: short gaps are interpolated, longer gaps "
                   "filled with the same hour of the previous week.",
                   "Outliers were checked with the IQR rule and kept, because they are real peak usage.",
                   f"Minute readings were converted to <b>{stats['hourly_rows']:,} hourly records</b>.",
                   "The dataset is in the repository (data/household_power_consumption.csv.gz)."]),
          p("4. Feature engineering", H1),
          p("31 input features, all built only from the <b>past</b>, so the model never sees the hour it predicts "
            "(an automated test checks this):"),
          bullets(["Lag features: usage 1, 2, 3, 6, 12, 24, 48 and 168 hours ago (168 h = same hour last week).",
                   "Rolling averages and standard deviations over 3, 6, 24 and 168 hours.",
                   "Calendar: hour, weekday, month, weekend, holiday (with sin/cos encodings).",
                   "Previous hour's readings: voltage, current, reactive power and the three sub-meters."]),
          PageBreak()]

# ---------------------------------------------------------------- 5 ML
rows = [["Model", "Train MAE", "Validation MAE", "Validation R²"]]
for n, v in V.items():
    name = f"<b>{LBL[n]} (deployed)</b>" if n == meta["model_name"] else LBL[n]
    rows.append([name, f"{v['train_mae']:.3f}", f"{v['mae']:.3f}", f"{v['r2']:.3f}"])
dt = TUN["decision_tree"]
cvs = clf["validation"]
story += [p("5. The machine learning (syllabus algorithms only)", H1),
          p("<b>Split by time, not randomly:</b> train 2006-2009, validate Jan-May 2010, test Jun-Nov 2010. A random "
            "split would leak future data into training and give fake high scores."),
          p("<b>Algorithms compared</b> (all from the syllabus): linear regression, ridge regression, decision tree, "
            "KNN, SVM and random forest, plus two simple baselines (\"same as last hour\" and \"same hour yesterday\")."),
          table(rows, [6.0, 3.0, 3.6, 3.6]),
          Spacer(1, 6),
          p("<b>Hyper-parameter tuning:</b> GridSearchCV with time-series cross-validation (Experiment 4)."),
          p(f"<b>Key teaching point - over-fitting:</b> the default decision tree has <b>{V['decision_tree']['train_mae']:.3f} "
            f"error on training data</b> but the worst validation error ({V['decision_tree']['mae']:.3f} kW): it memorised "
            f"the data. Tuning it (max_depth={dt['best_params']['max_depth']}, min_samples_leaf="
            f"{dt['best_params']['min_samples_leaf']}) cuts the error to {dt['tuned_val']['mae']:.3f} kW."),
          p(f"<b>Winner: {LBL[meta['model_name']]}</b> (lowest validation error). Results on the unseen test period:"),
          table([["Metric", "Value"],
                 ["MAE (average error)", f"{T['mae']:.3f} kW"],
                 ["MSE / RMSE", f"{T['mse']:.3f} / {T['rmse']:.3f} kW"],
                 ["R²", f"{T['r2']:.3f}"],
                 ["Accuracy (1 - WAPE)", f"{RF_TEST['accuracy_1_minus_wape'] * 100:.1f}%" if RF_TEST else "-"],
                 ["Forecasts within ±0.5 kW", f"{RF_TEST['within_0_5kw_pct']:.1f}%" if RF_TEST else "-"],
                 ["Improvement over baseline", f"{(1 - T['mae'] / B['mae']) * 100:.1f}% lower error"]], [8.0, 8.2]),
          Spacer(1, 6),
          p(f"<b>Second model - peak-hour alert</b> (classification, Experiment 3): predicts whether the next hour is a peak "
            f"hour (above {clf['threshold_kw']:.2f} kW, the busiest 25% of hours). Three classifiers were compared:"),
          table([["Classifier (validation)", "Train F1", "Accuracy", "F1", "ROC-AUC"]] +
                [[f"<b>{CLF[n]} (selected)</b>" if n == clf["selected"] else CLF[n], f"{s['train_f1']:.3f}",
                  f"{s['accuracy']:.3f}", f"{s['f1']:.3f}", f"{s['roc_auc']:.3f}"] for n, s in cvs.items()],
                [6.0, 2.4, 2.6, 2.4, 2.8]),
          Spacer(1, 6),
          table([["Selected classifier on the test set", "Value"],
                 ["Accuracy", f"{CT['accuracy'] * 100:.1f}%"], ["Precision", f"{CT['precision'] * 100:.1f}%"],
                 ["Recall (share of real peaks caught)", f"{CT['recall'] * 100:.1f}%"], ["F1-score", f"{CT['f1']:.3f}"],
                 ["ROC-AUC", f"{CT['roc_auc']:.3f}"],
                 ["Confusion matrix", f"{CT['confusion_matrix'][1][1]} peaks caught, {CT['confusion_matrix'][1][0]} missed, "
                                      f"{CT['confusion_matrix'][0][1]} false alarms"]], [8.0, 8.2]),
          Spacer(1, 6),
          p("<b>If asked \"why is accuracy only about 68%?\"</b> - \"We predict one single house hour by hour; people "
            "switching on an oven or heater at random cannot be predicted from past readings. We also used a strict "
            "time-based split, which gives honest, lower scores than shuffled splits. About 81% of forecasts are within "
            "half a kilowatt, which is the practical measure.\"", QUOTE),
          KeepTogether([Image(str(ROOT / "reports" / "figures" / "fig_train_vs_val.png"), width=16 * cm, height=6 * cm),
                        p("Training vs validation error: the default decision tree's zero training error is over-fitting.", SMALL)]),
          Spacer(1, 10)]

# ---------------------------------------------------------------- 6 MLOps
story += [p("6. The MLOps pipeline (the core of the course)", H1),
          Image(str(ROOT / "report" / "diagrams" / "fig_architecture.png"), width=16.5 * cm, height=7.8 * cm),
          Spacer(1, 6),
          table([["Step", "Tool", "What it does", "Syllabus"],
                 ["Version control", "Git + GitHub", "All code tracked; commits throughout the project", "Exp 5"],
                 ["Data and pipeline versioning", "DVC", "One command (dvc repro) reruns everything; dvc.lock records exact versions", "Exp 9"],
                 ["Experiment tracking", "MLflow", "Logs every model's parameters and scores; model registry", "Model versioning"],
                 ["API", "Flask REST API", "/api/forecast, /api/predict, /api/health, /api/metrics; rejects bad input (HTTP 422)", "Exp 7"],
                 ["Containerisation", "Docker + Compose", "5 services: API + web app, dashboard, MLflow, Prometheus, Grafana", "Exp 6"],
                 ["Orchestration", "Kubernetes", "2 replicas, health checks, autoscaling (manifests)", "Project-based learning"],
                 ["CI/CD", "GitHub Actions + Vercel", "Every push: rerun pipeline, quality gates, 11 tests, Docker build, auto-deploy", "Exp 8"],
                 ["Cloud", "Vercel (serverless on AWS)", f"Public web app and API at {LIVE}", "Unit 3: cloud deployment"],
                 ["Monitoring", "Prometheus + Grafana", "Live dashboard of requests, speed, predictions, errors", "Exp 10"],
                 ["Drift detection", "PSI", "Flags when new data differs from training data; recommends retraining", "Monitoring"]],
                [3.4, 3.2, 7.2, 2.7]),
          Spacer(1, 6),
          p(f"<b>Strong point to mention:</b> the drift monitor actually caught real drift. The model trained mostly on "
            f"winters while the test period was summer; water-heater/AC usage changed a lot (PSI "
            f"{mon['psi']['Sub_metering_3_lag1']:.2f}, above the 0.2 threshold), so the system flagged "
            "\"retraining recommended\" on its own."),
          PageBreak()]

# ---------------------------------------------------------------- 7-10
story += [p("7. Ethics (Unit 3)", H1),
          bullets(["<b>Fairness and bias:</b> error was checked across groups of hours; it is worse in evenings and "
                   "high-use hours, and the model is trained on one French house, so it must be retrained before use "
                   "elsewhere.",
                   "<b>Privacy:</b> meter data can reveal when people are home; only a public anonymised dataset is used "
                   "and the prediction log does not store user readings.",
                   "<b>Security:</b> all API input is validated, request size is limited, and rejected requests are "
                   "counted in monitoring.",
                   "<b>Transparency:</b> model card (MODEL_CARD.md), feature importance, and results anyone can "
                   "reproduce with dvc repro."]),
          p("8. Live demonstration (about 3 minutes)", H1),
          ListFlowable([ListItem(p(t), leftIndent=14) for t in [
              f"Open {link(LIVE)}, pick a date, click <b>Run forecast</b>: show the forecast against the actual line and "
              "the red peak-hour markers.",
              "Scroll to <b>Model performance</b>: the comparison table and the over-fitting chart.",
              "Show <b>Peak-hour alerts</b>: confusion matrix and ROC curve.",
              f"Open the repository {link(REPO)}, then the <b>Actions</b> tab: green CI/CD runs.",
              "Optional (Docker running): Grafana at localhost:3000 for the monitoring dashboard."]],
              bulletType="1", leftIndent=14),
          p("9. Team roles", H1),
          table([["Member", "Role and contribution"],
                 ["Kunal Mittal (24BCS12354)", "Data & ML lead: data pipeline, feature engineering, models and tuning, "
                                               "peak-hour classifier, evaluation"],
                 ["Ansh Kumar (24BCS12097)", "Deployment lead: Flask API, React web app, Streamlit, Docker, Vercel and "
                                             "Kubernetes"],
                 ["Vishal Kumar Shrivastava (24BCS12629)", "MLOps & QA lead: CI/CD, DVC, MLflow, Prometheus and Grafana, "
                                                           "testing, report"]], [5.4, 10.8]),
          p("10. Likely viva questions", H1),
          table([["Question", "Answer"],
                 ["Why random forest?", "Lowest validation error of the syllabus algorithms. It averages many decision "
                                        "trees trained on different samples, which fixes the single tree's over-fitting."],
                 ["Why a time-based split?", "A random split lets the model learn from the future and gives fake high "
                                             "scores. Real forecasting always predicts forward in time."],
                 ["What does DVC do that Git does not?", "Versions large data and pipeline outputs by content hash, and "
                                                         "re-runs only the stages whose inputs changed."],
                 ["What happens on git push?", "GitHub Actions re-runs the pipeline and tests; Vercel builds and "
                                               "redeploys the live site automatically."],
                 ["How do you know when to retrain?", "If PSI > 0.2 or the error gets 20% worse, the monitor flags it; "
                                                      "the pipeline also retrains every week."],
                 ["Why GridSearchCV with TimeSeriesSplit?", "It tries every hyper-parameter combination while every "
                                                             "fold trains on the past and validates on the following period."],
                 ["What is over-fitting here?", "The default decision tree: zero training error, worst validation error. "
                                                "Limiting depth and leaf size fixes it."],
                 ["Precision vs recall for the alert?", f"Recall {CT['recall'] * 100:.0f}% means most real peaks are caught; "
                                                       f"precision {CT['precision'] * 100:.0f}% means some alerts are false alarms "
                                                       "- the safer trade-off for warnings."]], [5.0, 11.2]),
]

doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                        bottomMargin=1.8 * cm, title="VoltCast - Project Explanation",
                        author="Kunal Mittal, Ansh Kumar, Vishal Kumar Shrivastava")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("Saved", OUT)
