"""Builds the project report (.docx) in the Chandigarh University format from pipeline outputs."""
import json
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
DIAG = ROOT / "report" / "diagrams"
SHOT = ROOT / "reports" / "screenshots"
OUT = ROOT / "report" / "Household_Electricity_Consumption_Prediction_Report.docx"

TITLE = "HOUSEHOLD ELECTRICITY CONSUMPTION PREDICTION SYSTEM"
STUDENTS = ["Kunal Mittal (24BCS12354)", "Ansh Kumar (24BCS12097)",
            "Vishal Kumar Shrivastava (24BCS12629)"]
DEGREE = "BACHELOR OF ENGINEERING"
BRANCH = "COMPUTER SCIENCE & ENGINEERING"
MONTH_YEAR = "November 2026"

metrics = json.loads((ROOT / "reports" / "metrics.json").read_text())
meta = json.loads((ROOT / "models" / "metadata.json").read_text())
mon = json.loads((ROOT / "reports" / "monitoring.json").read_text())
stats = json.loads((ROOT / "reports" / "data_stats.json").read_text())
imp = pd.read_csv(ROOT / "reports" / "feature_importance.csv", index_col=0)["importance"].sort_values(ascending=False)
V, T, B = metrics["validation"], meta["test_metrics"], meta["baseline_test_metrics"]
MAE_GAIN = (1 - T["mae"] / B["mae"]) * 100
RMSE_GAIN = (1 - T["rmse"] / B["rmse"]) * 100
AN = json.loads((ROOT / "reports" / "analysis.json").read_text())
AFIG = ROOT / "reports" / "analysis"
TERM = ROOT / "reports" / "terminal"
TERM_MANIFEST = json.loads((TERM / "manifest.json").read_text()) if (TERM / "manifest.json").exists() else {}

doc = Document()
fig_no = {"ch": 0, "fig": 0, "tab": 0}


# ---------- low-level formatting helpers ----------
def set_font(run, size=12, bold=False, italic=False):
    run.font.name = "Times New Roman"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    run.font.size = Pt(size)
    run.bold, run.italic = bold, italic
    run.font.color.rgb = RGBColor(0, 0, 0)
    return run


def para(text="", size=12, bold=False, italic=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY, spacing=1.5,
         after=6, style=None):
    p = doc.add_paragraph(style=style)
    p.alignment = align
    pf = p.paragraph_format
    pf.line_spacing = spacing
    pf.space_after = Pt(after)
    if text:
        add_rich(p, text, size, bold, italic)
    return p


def add_rich(p, text, size=12, bold=False, italic=False):
    """Supports **bold** spans inside a paragraph."""
    parts = text.split("**")
    for i, part in enumerate(parts):
        if part:
            set_font(p.add_run(part), size, bold or i % 2 == 1, italic)


def bullets(items, numbered=False):
    for it in items:
        p = para(style="List Number" if numbered else "List Bullet", after=3)
        add_rich(p, it)


def page_break():
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def field(p, instr, size=12, placeholder=""):
    r = p.add_run()
    set_font(r, size)
    for tag, attr in [("w:fldChar", "begin"), ("w:instrText", instr), ("w:fldChar", "separate")]:
        el = OxmlElement(tag)
        if tag == "w:fldChar":
            el.set(qn("w:fldCharType"), attr)
        else:
            el.set(qn("xml:space"), "preserve")
            el.text = attr
        r._r.append(el)
    t = p.add_run(placeholder)
    set_font(t, size)
    end = p.add_run()
    e = OxmlElement("w:fldChar")
    e.set(qn("w:fldCharType"), "end")
    end._r.append(e)


def heading(text, level):
    h = doc.add_heading(level=level)
    h.paragraph_format.space_before = Pt(12 if level > 1 else 0)
    h.paragraph_format.space_after = Pt(8)
    h.paragraph_format.line_spacing = 1.5
    set_font(h.add_run(text), 14 if level == 2 else 12, bold=True)
    return h


def chapter(num, title):
    page_break()
    fig_no.update(ch=num, fig=0, tab=0)
    h = doc.add_heading(level=1)
    h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    h.paragraph_format.space_after = Pt(18)
    r = set_font(h.add_run(f"CHAPTER {num}. "), 16, bold=True)
    r.add_break()
    set_font(h.add_run(title), 16, bold=True)


def unnumbered(title, label=None):
    page_break()
    if label:
        fig_no.update(ch=label, fig=0, tab=0)
    h = doc.add_heading(level=1)
    h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    h.paragraph_format.space_after = Pt(18)
    set_font(h.add_run(title), 16, bold=True)


def caption(kind, text):
    key = "fig" if kind == "Figure" else "tab"
    fig_no[key] += 1
    p = doc.add_paragraph(style="Caption")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(10)
    p.paragraph_format.keep_with_next = kind == "Table"  # a table caption stays on the table's page
    set_font(p.add_run(f"{kind} {fig_no['ch']}."), 11, bold=True)
    # \r pins the number so unnumbered Heading 1 sections don't reset it; the field keeps lists working.
    field(p, f"SEQ {kind} \\r {fig_no[key]} \\* ARABIC", 11, str(fig_no[key]))
    set_font(p.add_run(f": {text}"), 11, italic=True)


def figure(path, text, width=15.5):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(path), width=Cm(width))
    caption("Figure", text)


def table(header, rows, text, widths=None):
    caption("Table", text)
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    # Repeat the header row on every page the table spans, and keep it with the first data row.
    trPr = t.rows[0]._tr.get_or_add_trPr()
    hdr = OxmlElement("w:tblHeader")
    hdr.set(qn("w:val"), "true")
    trPr.append(hdr)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]
        c.text = ""
        set_font(c.paragraphs[0].add_run(str(h)), 10.5, bold=True)
        c.paragraphs[0].paragraph_format.keep_with_next = True
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:fill"), "D9E2F3")
        c._tc.get_or_add_tcPr().append(shd)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            add_rich(cells[i].paragraphs[0], str(v), 10.5)
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w)
    para(after=4)


def shade(paragraph, fill="F2F2F2"):
    pPr = paragraph._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), fill)
    pPr.append(shd)


def code_listing(rel_path: str, title: str) -> None:
    """Full source of one project file, with line numbers, in a shaded monospace block."""
    heading(f"{rel_path}: {title}", 3)
    lines = (ROOT / rel_path).read_text(encoding="utf-8").rstrip("\n").split("\n")
    for i, line in enumerate(lines, 1):
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.space_before = pf.space_after = Pt(0)
        pf.line_spacing = 1.0
        shade(p)
        r = p.add_run(f"{i:4d}  {line}")
        r.font.name = "Consolas"
        r._element.rPr.rFonts.set(qn("w:eastAsia"), "Consolas")
        r.font.size = Pt(7.5)
    para(after=6)


def terminal(name: str, text: str, width: float = 16) -> None:
    """Terminal evidence: every image of a captured command (see report/capture_terminal.py)."""
    info = TERM_MANIFEST.get(name)
    if not info:
        return
    imgs = info["images"]
    for k, img in enumerate(imgs, 1):
        part = f" (part {k} of {len(imgs)})" if len(imgs) > 1 else ""
        figure(TERM / img, f"{text}{part}, run {info['time']}", width)


# ---------- document-wide setup ----------
for s in doc.sections:
    s.page_height, s.page_width = Cm(29.7), Cm(21.0)
    s.left_margin, s.right_margin, s.top_margin, s.bottom_margin = Cm(3.0), Cm(2.5), Cm(2.5), Cm(2.5)
normal = doc.styles["Normal"]
normal.font.name = "Times New Roman"
normal.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
normal.font.size = Pt(12)
for name in ["Heading 1", "Heading 2", "Heading 3", "Caption", "List Bullet", "List Number"]:
    st = doc.styles[name]
    st.font.name = "Times New Roman"
    st.font.color.rgb = RGBColor(0, 0, 0)
# Ask Word to refresh TOC / list-of-figures fields when the file is opened.
upd = OxmlElement("w:updateFields")
upd.set(qn("w:val"), "true")
doc.settings.element.append(upd)

# ---------- title page ----------
C = WD_ALIGN_PARAGRAPH.CENTER
for _ in range(3):
    para(align=C)
para(TITLE, 18, True, align=C, spacing=1.5, after=24)
para("A PROJECT REPORT", 14, True, align=C, after=24)
para("Submitted by", 14, True, True, align=C, after=12)
for i, name in enumerate(STUDENTS):
    para(name, 16, True, align=C, spacing=1.15, after=30 if i == len(STUDENTS) - 1 else 2)
para("in partial fulfillment for the award of the degree of", 14, True, True, align=C, spacing=1.5, after=12)
para(DEGREE, 16, True, align=C, after=6)
para("IN", 14, align=C, after=6)
para(BRANCH, 14, align=C, after=60)
para("Chandigarh University", 14, True, align=C, after=6)
para(MONTH_YEAR, 14, align=C)

# ---------- bonafide certificate ----------
page_break()
para("BONAFIDE CERTIFICATE", 16, True, align=C, after=24)
para(f"Certified that this project report “{TITLE.title()}” is the bonafide work of "
     f"“Kunal Mittal, Ansh Kumar and Vishal Kumar Shrivastava” who carried out the project work under my/our "
     "supervision.", 14, after=60)
t = doc.add_table(rows=1, cols=2)
left = ["", "SIGNATURE", "", "", "HEAD OF THE DEPARTMENT", "", ""]
right = ["", "SIGNATURE", "", "", "SUPERVISOR", "", ""]
for cell, lines in zip(t.rows[0].cells, [left, right]):
    cell.text = ""
    for i, l in enumerate(lines):
        p = cell.paragraphs[0] if i == 0 else cell.add_paragraph()
        set_font(p.add_run(l), 12, bold=l.isupper() and not l.startswith("<<"))
para(after=40)
para("Submitted for the project viva-voce examination held on ____________________", 12, after=60)
t = doc.add_table(rows=1, cols=2)
for cell, l in zip(t.rows[0].cells, ["INTERNAL EXAMINER", "EXTERNAL EXAMINER"]):
    cell.text = ""
    set_font(cell.paragraphs[0].add_run(l), 12, True)
t.rows[0].cells[1].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT

# ---------- TOC, lists ----------
page_break()
para("TABLE OF CONTENTS", 16, True, align=C, after=12)
field(para(align=WD_ALIGN_PARAGRAPH.LEFT), 'TOC \\o "1-2" \\h \\z \\u', 12,
      "Right-click and choose “Update Field” to build the table of contents.")
page_break()
para("List of Figures", 16, True, align=C, after=12)
field(para(align=WD_ALIGN_PARAGRAPH.LEFT), 'TOC \\h \\z \\c "Figure"', 12,
      "Right-click and choose “Update Field” to build the list of figures.")
page_break()
para("List of Tables", 16, True, align=C, after=12)
field(para(align=WD_ALIGN_PARAGRAPH.LEFT), 'TOC \\h \\z \\c "Table"', 12,
      "Right-click and choose “Update Field” to build the list of tables.")

page_break()
para("List of Standards", 16, True, align=C, after=12)
table(["Standard", "Publishing Agency", "About the standard", "Where used"], [
    ["ISO/IEC 23053:2022", "ISO / IEC", "Framework for Artificial Intelligence systems using Machine Learning; "
     "defines the ML pipeline components (data acquisition, pre-processing, training, evaluation, deployment).",
     "Ch. 3.2, 3.6"],
    ["ISO/IEC 25010:2023", "ISO / IEC", "Systems and software quality model (functional suitability, "
     "reliability, maintainability, portability).", "Ch. 3.2, 4.1"],
    ["IEC 62056 (DLMS/COSEM)", "IEC", "Electricity metering data exchange; defines how smart meters expose "
     "load-profile readings such as those used as model inputs.", "Ch. 3.1"],
    ["IS 16444", "Bureau of Indian Standards", "A.C. static direct-connected smart energy meters; relevant for "
     "deploying the system on Indian household smart meters.", "Ch. 3.2"],
    ["ISO 50001:2018", "ISO", "Energy management systems: requirements for measuring, analysing and "
     "improving energy performance.", "Ch. 1.1, 3.2"],
    ["OpenMetrics (Prometheus exposition format)", "CNCF", "Text format for exposing service metrics; the "
     "API's /api/metrics endpoint follows it so Prometheus can scrape it.", "Ch. 3.6, 4.1"],
    ["Digital Personal Data Protection Act, 2023", "Government of India", "Rules for processing digital "
     "personal data; applies when real household meter data is used.", "Ch. 3.2, 4.1"],
], "Standards referred to in the project", [3.2, 2.8, 7.0, 2.2])

# ---------- abstract etc ----------
clf = json.loads((ROOT / "reports" / "classification.json").read_text())
fair = json.loads((ROOT / "reports" / "fairness.json").read_text())
TUN = metrics["tuning"]
CT, CR = clf["test"], clf["forecast_threshold_rule_test"]
LBL = {"persistence_baseline": "Persistence baseline", "seasonal_naive_24h": "Seasonal naive (24 h)",
       "linear_regression": "Linear regression", "ridge_regression": "Ridge regression",
       "decision_tree": "Decision tree", "knn": "K-nearest neighbours (KNN)", "svm": "Support vector machine (SVR)",
       "random_forest": "Random forest", "hist_gradient_boosting": "Histogram gradient boosting"}
nf = lambda k=1: f"{fig_no['ch']}.{fig_no['fig'] + k}"   # number the next figure will get
nt = lambda k=1: f"{fig_no['ch']}.{fig_no['tab'] + k}"   # number the next table will get
CANDIDATES_ORDER = list(V)
DT = TUN["decision_tree"]
OUT_GAP = stats["outliers"]["Global_active_power"]

unnumbered("ABSTRACT")
para("Residential buildings account for a large share of national electricity demand, yet most households "
     "receive only a monthly bill and have no forward-looking view of their consumption. Utilities, in turn, "
     "need short-term load forecasts at household level to plan demand response as smart meters are rolled "
     "out. This project develops **VoltCast**, a Household Electricity Consumption Prediction System that "
     "forecasts a household's hourly active power for the next 1–48 hours, flags likely peak-consumption hours, "
     "and is built end to end with Machine Learning Operations (MLOps) practice.")
para(f"The UCI “Individual Household Electric Power Consumption” dataset ({stats['raw_rows']:,} one-minute "
     f"readings over almost four years) is cleaned of {stats['raw_missing_pct']}% missing values, checked for "
     f"outliers, resampled to {stats['hourly_rows']:,} hourly records and turned into 31 leakage-free features. "
     "Nine regression models covering the algorithms of the course (linear regression, decision tree, k-nearest "
     "neighbours, support vector machine, random forest) plus ridge regression, gradient boosting and two naive "
     "baselines are trained on a chronological split, tuned with GridSearchCV using time-series cross-validation, "
     "and tracked in MLflow. Histogram gradient boosting gives the best forecast: test MAE "
     f"{T['mae']:.3f} kW, MSE {T['mse']:.3f}, R² {T['r2']:.3f}, a {MAE_GAIN:.1f}% MAE reduction over the "
     f"persistence baseline. A tuned decision-tree classifier detects peak hours with {CT['accuracy']*100:.1f}% "
     f"accuracy, {CT['recall']*100:.1f}% recall and ROC-AUC {CT['roc_auc']:.3f}. The experiments also show "
     f"over-fitting clearly: the default decision tree reaches zero training error but the worst validation "
     f"error, which tuning reduces from {DT['default_val']['mae']:.3f} to {DT['tuned_val']['mae']:.3f} kW.")
para("The models are versioned with DVC and the MLflow model registry, and served by a Flask REST API with "
     "input validation. They run in Docker (a five-service Docker Compose stack, plus Kubernetes manifests) and "
     "on the Vercel cloud behind a React web application. A GitHub Actions pipeline reproduces the DVC pipeline, "
     "applies quality gates, runs 11 automated tests and builds the container on every change and weekly. "
     "Prometheus collects service metrics that Grafana displays, every prediction is logged, and a drift "
     "monitor based on the Population Stability Index correctly detected seasonal drift and recommended "
     "retraining. Fairness, transparency, privacy and security are analysed, including error across groups "
     "of hours.")
para("Keywords: load forecasting, smart meter, gradient boosting, decision tree, GridSearchCV, MLOps, MLflow, "
     "DVC, Flask, Docker, Prometheus, Grafana, responsible AI.", italic=True)

unnumbered("GRAPHICAL ABSTRACT")
figure(DIAG / "fig_architecture.png", "Graphical abstract: end-to-end MLOps architecture of VoltCast", 16)

unnumbered("ABBREVIATIONS")
table(["Abbreviation", "Full form"], [
    ["AI", "Artificial Intelligence"], ["API", "Application Programming Interface"],
    ["ARIMA", "Auto-Regressive Integrated Moving Average"], ["AUC", "Area Under the (ROC) Curve"],
    ["CI/CD", "Continuous Integration / Continuous Deployment"], ["CO", "Course Outcome"],
    ["CT", "Continuous Training"], ["CV", "Cross-Validation"], ["CNN", "Convolutional Neural Network"],
    ["DVC", "Data Version Control"], ["EDA", "Exploratory Data Analysis"],
    ["FN / FP / TN / TP", "False negative / false positive / true negative / true positive"],
    ["HGB", "Histogram-based Gradient Boosting"], ["IEA", "International Energy Agency"],
    ["IQR", "Inter-Quartile Range"], ["K8s", "Kubernetes"], ["KNN", "K-Nearest Neighbours"],
    ["kW / kWh", "Kilowatt / Kilowatt-hour"], ["LSTM", "Long Short-Term Memory"],
    ["MAE", "Mean Absolute Error"], ["MAPE", "Mean Absolute Percentage Error"], ["ML", "Machine Learning"],
    ["MLOps", "Machine Learning Operations"], ["MSE", "Mean Squared Error"],
    ["NILM", "Non-Intrusive Load Monitoring"], ["PSI", "Population Stability Index"],
    ["REST", "Representational State Transfer"], ["RF", "Random Forest"], ["RMSE", "Root Mean Squared Error"],
    ["ROC", "Receiver Operating Characteristic"], ["R²", "Coefficient of determination"],
    ["SVM / SVR", "Support Vector Machine / Support Vector Regression"],
    ["UCI", "University of California, Irvine (ML Repository)"],
], "List of abbreviations", [4, 11])

unnumbered("SYMBOLS")
table(["Symbol", "Meaning"], [
    ["yₜ", "Hourly mean global active power at hour t (kW)"],
    ["ŷₜ", "Predicted active power at hour t (kW)"],
    ["lag_k", "Value of the target k hours earlier, yₜ₋ₖ"],
    ["μ_w, σ_w", "Rolling mean and standard deviation over the previous w hours"],
    ["n", "Number of evaluation samples"],
    ["α", "Ridge regularisation strength"],
    ["η", "Learning rate of gradient boosting"],
    ["k", "Number of neighbours in KNN"],
    ["C, ε", "SVR penalty and epsilon-insensitive margin"],
    ["σ_r", "Standard deviation of test residuals (used for prediction intervals)"],
    ["h", "Forecast horizon in hours"],
    ["τ", "Peak-hour threshold (75th percentile of training consumption, kW)"],
    ["pᵢ, qᵢ", "Share of reference and current data in bin i (PSI)"],
], "List of symbols", [4, 11])

# ================= CHAPTER 1 =================
chapter(1, "INTRODUCTION")
heading("1.1. Identification of Client / Need / Relevant Contemporary issue", 2)
para("Electricity use in buildings is one of the largest and fastest-growing components of energy demand. "
     "The International Energy Agency reports that buildings account for roughly 30% of global final energy "
     "consumption, and space cooling, water heating and appliances in homes are major drivers of evening peak "
     "demand [1]. In India, the domestic (household) sector consumes about one quarter of all electricity "
     "sold, and its share keeps rising with appliance ownership and air-conditioner adoption [2].")
para("Three groups of stakeholders need short-term household consumption forecasts:")
bullets([
    "**Households** want to know how much energy and money they will use tomorrow, so they can shift flexible "
    "loads (washing machines, water heaters) and avoid bill shock; time-of-day tariffs make this more valuable.",
    "**Distribution utilities** run demand-response and peak-shaving programmes and must forecast load at fine "
    "granularity. India's Revamped Distribution Sector Scheme targets the installation of about 250 million "
    "prepaid smart meters, which will generate exactly the kind of high-frequency data this project uses [3].",
    "**Smart-home and energy-management vendors** need a deployable, monitored prediction service rather than a "
    "one-off notebook model, which is the gap that MLOps practice addresses [4], [5].",
])
para("Studies of production ML systems show that the model code is a small part of a real ML system and that "
     "most failures come from data dependencies, configuration and lack of monitoring [4], [23]. A consumption "
     "predictor that silently degrades when seasons or occupants change is a contemporary, documented issue.")

heading("1.2. Identification of Problem", 2)
para("Households and utilities lack a reliable, continuously maintained way to anticipate a single "
     "home's electricity demand over the next hours and to know in advance which hours will be peak hours. "
     "Household load is highly volatile, depends on human routines, weekday, season and appliance usage, and "
     "drifts over time. Meter readings contain gaps, noise and extreme values. Models built once and left "
     "unattended become stale, and there is no feedback that tells the operator when this happens.")

heading("1.3. Identification of Tasks", 2)
para(f"The work was divided into the following tasks (Table {nt()}), which also define the structure of this report:")
table(["#", "Task", "Output", "Chapter"], [
    ["T1", "Survey of the problem, existing forecasting methods and MLOps literature", "Problem definition, objectives", "1, 2"],
    ["T2", "Selection of a public, realistic smart-meter dataset and EDA", "Data understanding, figures", "3.1"],
    ["T3", "Design of features, constraints (incl. ethics) and alternative designs", "Selected design", "3"],
    ["T4", "Data pipeline: ingestion, cleaning, outliers, resampling, features, versioning with DVC", "Versioned datasets", "3.6, 4.1"],
    ["T5", "Model development: 9 algorithms, GridSearchCV tuning, over-fitting analysis, MLflow registry", "Registered forecaster", "4.1"],
    ["T6", "Peak-hour classifier with accuracy, precision, recall, F1, ROC", "Peak alert model", "4.1"],
    ["T7", "Deployment: Flask REST API, Docker, Kubernetes, Vercel cloud, React web app", "Running services", "4.1"],
    ["T8", "Automation and operations: CI/CD, tests, Prometheus, Grafana, logging, drift monitoring", "CI pipeline, dashboards", "4.1"],
    ["T9", "Responsible-AI analysis: fairness, transparency, privacy, security", "Model card, bias analysis", "3.2, 4.1"],
    ["T10", "Validation, documentation, report and user manual", "This report", "4, 5, Manual"],
], "Identified tasks", [1.0, 7.3, 4.2, 2.0])

heading("1.4. Timeline", 2)
para(f"The project was executed over twelve weeks. Figure {nf()} shows the Gantt chart; tasks overlap where the "
     "pipeline allowed parallel work (for example, the web application was built while CI was configured).")
figure(DIAG / "fig_gantt.png", "Gantt chart of the project timeline")

heading("1.5. Organization of the Report", 2)
bullets([
    "**Chapter 1** introduces the need, the problem, the tasks and the timeline.",
    "**Chapter 2** reviews the history of load forecasting and existing statistical, machine-learning and deep-"
    "learning solutions, compares them, and defines the problem and objectives mapped to the course outcomes.",
    "**Chapter 3** evaluates features and constraints (including ethical ones), compares two alternative "
    "designs, selects one and presents the implementation methodology.",
    "**Chapter 4** presents the implementation and validates the results: model comparison, tuning, "
    "classification metrics, deployment, CI/CD, monitoring and the responsible-AI analysis.",
    "**Chapter 5** concludes, discusses deviations from expectations and lists future work.",
    "References, Appendix and a step-by-step User Manual follow.",
])

# ================= CHAPTER 2 =================
chapter(2, "LITERATURE REVIEW/BACKGROUND STUDY")
heading("2.1. Timeline of the reported problem", 2)
table(["Period", "Milestone"], [
    ["1970s", "Box–Jenkins ARIMA methodology becomes the standard for utility-level load forecasting [6]."],
    ["1992", "Hart introduces Non-Intrusive Load Monitoring, showing household appliance loads can be inferred "
             "from aggregate meter data [7]."],
    ["2000s", "Machine-learning regressors (SVR, Random Forest [8], Gradient Boosting [9]) applied to building "
              "energy prediction."],
    ["2012", "UCI publishes the Individual Household Electric Power Consumption dataset, which becomes a "
             "benchmark for household-level forecasting [10]."],
    ["2015", "Sculley et al. document “hidden technical debt” in production ML systems [4]."],
    ["2016–2019", "Deep learning (LSTM, CNN-LSTM) applied to household load forecasting [11], [12]; "
                  "reviews of data-driven building energy prediction published [13]."],
    ["2018", "MLflow released to manage the ML lifecycle (tracking, projects, registry) [14]."],
    ["2016–present", "Probabilistic forecasting becomes the norm for load forecasting [15]; national "
                     "smart-meter roll-outs (e.g. India's RDSS) make household data abundant [3]."],
    ["2023", "MLOps formally defined and its architecture surveyed in IEEE Access [5]; India enacts the Digital "
             "Personal Data Protection Act, relevant to smart-meter data [25]."],
], "Timeline of the reported problem", [3.0, 12.0])

heading("2.2. Existing solutions", 2)
para("**Statistical time-series models.** ARIMA and exponential smoothing [6], [16] model linear "
     "autocorrelation and seasonality. They are interpretable and cheap but assume stationarity and struggle "
     "with the abrupt, appliance-driven spikes of a single household.")
para("**Classical machine learning.** The standard supervised-learning toolkit, namely linear regression, "
     "decision trees, k-nearest neighbours, support vector machines and ensembles [21], [22], has been applied "
     "to building energy. Fan et al. combined data-mining techniques to predict next-day building consumption "
     "and peak demand [17]. Tree ensembles such as random forest [8] and gradient boosting [9], [18] learn "
     "non-linear interactions from engineered lag and calendar features and are robust to scale and outliers.")
para("**Deep learning.** Marino et al. applied sequence-to-sequence LSTMs to the same UCI dataset [11]; Kim and "
     "Cho combined CNN and LSTM layers and reported lower error than linear and LSTM baselines [12]. These "
     "models capture long dependencies but require more data, GPU training and careful tuning.")
para("**MLOps platforms.** Kreuzberger et al. describe the components of a production ML system: versioned "
     "data and code, experiment tracking, model registry, automated pipelines, serving and monitoring [5]. "
     "Ameisen frames the same lifecycle from a product perspective, from prototype to monitored "
     "deployment [23]. MLflow [14], DVC and drift-detection methods [19] are widely used building blocks.")

para("The following paragraphs summarise the theory of every algorithm used in this project, following the "
     "course textbooks [21]–[23]. Throughout, xₜ is the feature vector of hour t and yₜ the consumption to predict.")
para("**Linear regression.** Models ŷ = w·x + b and chooses w, b to minimise the sum of squared errors "
     "Σ(yₜ − ŷₜ)² (ordinary least squares). It is fast and interpretable, since each weight is the change in "
     "consumption per unit of a feature, but it can only represent linear relationships, so interactions such as "
     "“evening AND winter” must be engineered by hand. Inputs are standardised so the weights are comparable.")
para("**Ridge regression (regularisation).** Adds a penalty α·Σw² to the least-squares loss. Large α shrinks the "
     "weights towards zero, which reduces variance and the effect of correlated features (the lag and rolling "
     "features here are strongly correlated) at the cost of some bias. α is chosen by cross-validation.")
para("**Decision tree.** Recursively splits the data on the feature and threshold that most reduce the "
     "mean squared error of the target in the two child nodes (the CART algorithm); a leaf predicts the mean of its "
     "training rows. Trees capture non-linear rules and need no scaling, but an unrestricted tree keeps splitting "
     "until every leaf is pure and memorises noise. max_depth and min_samples_leaf limit this and act as "
     "regularisation. For classification, splits minimise Gini impurity instead.")
para("**K-nearest neighbours (KNN).** Predicts the average target of the k training rows closest to the new row "
     "in Euclidean distance (optionally weighted by 1/distance). It makes no assumptions about the shape of the "
     "relationship, but it is sensitive to feature scale (so inputs are standardised), slow to predict on large "
     "data, and small k over-fits while large k under-fits.")
para("**Support vector machine for regression (SVR).** Fits a function that stays within ε of as many targets as "
     "possible while keeping the model as flat as possible; only points outside the ε-tube (support vectors) "
     "influence it. The RBF kernel lets it learn non-linear functions. C trades flatness against errors outside "
     "the tube. Training cost grows roughly quadratically with the number of rows, which is why SVR is the slowest "
     "model here.")
para("**Random forest.** Averages many decision trees, each grown on a bootstrap sample of the rows and choosing "
     "each split from a random subset of features (bagging) [8]. Averaging de-correlated trees reduces variance "
     "sharply, so the forest generalises far better than a single tree while keeping its non-linearity.")
para("**Gradient boosting.** Builds trees sequentially; each new tree is fitted to the residual errors (the "
     "negative gradient of the loss) of the current ensemble, and its prediction is added with a small learning "
     "rate η [9]. The histogram variant bins each feature into at most 255 values, which makes split finding very "
     "fast on tens of thousands of rows [18]. Boosting reduces bias step by step, while the learning rate, number of "
     "leaves and L2 penalty control over-fitting.")
para("**Logistic regression.** For the peak-hour classifier, models the probability of the positive class as "
     "σ(w·x + b), where σ is the sigmoid function, and is fitted by maximising the log-likelihood. It serves as the "
     "linear reference for the decision-tree classifier.")
para("**Model evaluation.** Because the data is a time series, ordinary k-fold cross-validation would train on "
     "the future and test on the past. TimeSeriesSplit instead creates expanding windows: fold 1 trains on the "
     "first sixth of the training period and validates on the next sixth, fold 5 trains on five sixths and "
     "validates on the last. GridSearchCV evaluates every hyper-parameter combination on these folds and keeps "
     "the one with the best mean score [21], [22].")
para("**MLOps building blocks.** Version control (Git) tracks code; data and pipeline versioning (DVC) "
     "tracks datasets, parameters and outputs by content hash, so any result can be reproduced from a commit. "
     "Experiment tracking and model registry (MLflow) records every run's parameters, metrics and model "
     "artifacts and numbers model versions. Containerisation (Docker) packages the code with its exact "
     "dependencies, and orchestration (Docker Compose, Kubernetes) runs and scales several containers as "
     "microservices. CI/CD (GitHub Actions, Vercel) automatically tests, builds and deploys on every change; "
     "continuous training re-runs the pipeline on a schedule. Monitoring (Prometheus collects time-series "
     "metrics by scraping HTTP endpoints; Grafana visualises them) and drift detection close the loop by "
     "telling the team when the model needs retraining [5], [23].")

heading("2.3. Bibliometric analysis", 2)
table(["Work", "Approach", "Key features", "Effectiveness", "Drawbacks"], [
    ["Box & Jenkins [6]", "ARIMA", "Linear autoregression, differencing", "Good for aggregate, smooth loads",
     "Poor on volatile single-house data; manual order selection"],
    ["Fan et al. [17]", "Ensemble data mining", "Feature selection + multiple regressors", "Accurate next-day "
     "building forecasts", "Commercial buildings; no deployment"],
    ["Marino et al. [11]", "LSTM seq2seq", "Learns temporal patterns directly", "Competitive on UCI data at "
     "hourly scale", "Overfits at minute scale; heavy training"],
    ["Kim & Cho [12]", "CNN-LSTM", "Convolutional feature extraction + LSTM", "Low error on UCI dataset",
     "Black box; GPU cost; no monitoring"],
    ["Amasyali & El-Gohary [13]", "Review", "Survey of 60+ studies", "Shows ML methods outperform "
     "statistical ones in most cases", "Notes lack of deployed, maintained systems"],
    ["Kreuzberger et al. [5]", "MLOps architecture", "CI/CD/CT, registry, monitoring", "Reference "
     "architecture", "Not specific to energy forecasting"],
], "Bibliometric comparison of existing solutions", [3.0, 2.3, 3.3, 3.3, 3.3])

heading("2.4. Review Summary", 2)
para("The literature shows that (i) machine-learning models with good lag and calendar features consistently "
     "beat purely statistical models on household data; (ii) deep networks can be marginally more accurate "
     "but at much higher training and operational cost; and (iii) almost all published household forecasting "
     "work stops at an offline accuracy number, with no serving, automation, monitoring or ethical review. "
     "This project therefore compares the standard algorithms under one protocol, tunes them, and wraps the "
     "best one in a complete MLOps pipeline, evaluated honestly against naive baselines on a chronological "
     "hold-out.")

heading("2.5. Problem Definition", 2)
para("**What is to be done:** build a system that, given the most recent week of hourly smart-meter readings "
     "for a household, forecasts the household's mean active power (kW) for each of the next 1–48 hours "
     "with a prediction interval, flags likely peak hours, and can be retrained, tested, deployed and "
     "monitored automatically.")
para("**How it is to be done:** using the UCI household dataset, a reproducible DVC pipeline, supervised "
     "regression and classification models compared and tuned under MLflow tracking, a Flask REST API packaged "
     "with Docker and deployed to the cloud, a GitHub Actions CI/CD workflow, and Prometheus/Grafana and "
     "PSI-based monitoring.")
para("**What is not to be done:** appliance-level disaggregation (NILM), control of household devices, "
     "billing integration, or collection of new personal data. Forecasts are for a single household and are "
     "not intended for grid-level dispatch or decisions about individuals.")

heading("2.6. Goals/Objectives", 2)
bullets([
    "Clean and resample the 2.07 M-row dataset into a gap-free hourly series with an auditable data report, "
    "including missing-value and outlier analysis.",
    "Engineer at least 25 features that use only past information (verified by an automated leakage test).",
    "Train, tune (GridSearchCV) and compare the course algorithms, namely linear regression, decision tree, KNN, "
    "SVM and random forest, under experiment tracking on a strict chronological split.",
    "Beat the persistence baseline on the held-out test period by at least 15% in MAE.",
    "Build a peak-hour classifier and report accuracy, precision, recall, F1, the confusion matrix and ROC.",
    "Serve predictions through a Flask REST API with input validation and response time under 1 s, "
    "containerised with Docker and deployed to a cloud platform.",
    "Automate pipeline, quality gates, tests and container build with CI/CD; version data and models with DVC.",
    "Monitor the service with Prometheus and Grafana, log predictions, and detect drift (PSI > 0.2).",
    "Analyse fairness, transparency, privacy and security of the system.",
], numbered=True)
para(f"Table {nt()} maps these objectives to the course outcomes of MLOps-1 (25CSH-320).")
table(["Course outcome", "How the project addresses it", "Section"], [
    ["CO1: ML fundamentals, types, applications", "Supervised regression and classification applied to a "
     "real energy-sector problem; algorithm survey", "1.1, 2.2"],
    ["CO2: Preprocessing, feature engineering, data handling", "Missing values, IQR outliers, resampling, "
     "scaling, 31 engineered features, chronological split", "3.1, 3.3, 4.1"],
    ["CO3: Develop and evaluate models with various algorithms", "9 regressors and 3 classifiers, "
     "GridSearchCV with time-series CV, over/under-fitting analysis, MSE/MAE/R² and accuracy/precision/"
     "recall/F1/ROC", "3.6, 4.1"],
    ["CO4: Deploy with APIs, microservices and cloud", "Flask REST API, Docker Compose microservices, "
     "Kubernetes manifests, Vercel cloud deployment", "4.1"],
    ["CO5: CI/CD and ethical considerations", "GitHub Actions + DVC + Vercel CD; fairness, transparency, "
     "privacy and security analysis; model card", "3.2, 4.1"],
], "Mapping of project objectives to course outcomes", [4.2, 8.3, 2.5])

# ================= CHAPTER 3 =================
chapter(3, "DESIGN FLOW/PROCESS")
heading("3.1. Evaluation & Selection of Specifications/Features", 2)
para(f"**Dataset.** The UCI dataset [10] contains {stats['raw_rows']:,} one-minute measurements from "
     f"{stats['start'][:10]} to {stats['end'][:10]} for one house in Sceaux, France: global active and reactive "
     "power, voltage, current intensity and three sub-meters (kitchen; laundry room; electric water heater and "
     "air conditioner). It was chosen because it is public, long enough to cover four winters, contains real "
     "gaps, and mirrors the load-profile channels exposed by IEC 62056 smart meters. It is stored as a single "
     f"CSV file versioned with DVC. Figures {nf()}–{nf(3)} summarise the exploratory analysis.")
figure(FIG / "fig_eda_timeseries.png", "Daily mean active power with 30-day rolling mean (strong yearly seasonality)")
figure(FIG / "fig_eda_heatmap.png", "Average consumption by weekday and hour (morning and evening peaks)")
figure(FIG / "fig_eda_month_submeter.png", "Monthly pattern and share of energy per sub-meter")
para("The EDA shows a strong winter peak (consumption roughly doubles from August to December), two daily "
     "peaks around 07:00 and 20:00–21:00, higher daytime use at weekends, and an August holiday dip.")
para(f"Figures {nf()}–{nf(7)} extend the analysis. Missing readings are concentrated in a few months, the worst "
     f"being {AN['eda']['missing_pct_by_month_max']['month'][:7]} with {AN['eda']['missing_pct_by_month_max']['pct']}% "
     "missing (long outages), which justified the two-step gap filling below. All variables are right-skewed, "
     "with many low-consumption hours and a long tail of peaks. The autocorrelation of hourly consumption is "
     f"{AN['eda']['autocorr']['lag_1']} at one hour, with repeating peaks every 24 hours "
     f"({AN['eda']['autocorr']['lag_24']} at 24 h, {AN['eda']['autocorr']['lag_168']} at one week), which "
     "is why the lag features at 1, 24 and 168 hours were chosen.")
figure(AFIG / "missing_by_month.png", "Share of missing one-minute readings per month")
figure(AFIG / "distributions.png", "Distribution of each hourly variable", 15.5)
figure(AFIG / "autocorrelation.png", "Autocorrelation of hourly consumption up to eight days")
figure(AFIG / "weekly_profile.png", "Average weekly load profile")
figure(AFIG / "submeter_by_hour.png", "Average energy by sub-meter and hour of day")
figure(AFIG / "monthly_boxplot.png", "Hourly consumption by month (box plots)")
figure(AFIG / "year_over_year.png", "Monthly average consumption, year by year")
desc = AN["eda"]["describe"]
table(["Variable", "Mean", "Std", "Min", "25%", "Median", "75%", "Max"],
      [[c.replace("_", " "), *[f"{desc[c][k]:.3f}" for k in ["mean", "std", "min", "25%", "50%", "75%", "max"]]]
       for c in desc], "Descriptive statistics of the hourly data", [4.2, 1.5, 1.5, 1.5, 1.5, 1.6, 1.5, 1.5])
para(f"**Outliers.** With the inter-quartile-range rule (outside Q1 − 1.5·IQR … Q3 + 1.5·IQR), "
     f"{OUT_GAP['count']} hourly values of active power ({OUT_GAP['pct']}%) lie above {OUT_GAP['upper']} kW "
     f"(Figure {nf()}). These are genuine peak-usage hours (heaters, ovens), exactly the hours the forecaster "
     "must learn, so they are kept rather than removed. Instead, distance- and margin-based models (KNN, SVM) "
     "receive standardised inputs so that large-valued columns do not dominate.")
figure(FIG / "fig_outliers.png", "Box plots of hourly readings; points beyond the whiskers are IQR outliers", 14)
para(f"These findings drove the feature specification in Table {nt()}.")
table(["Feature group", "Features", "Rationale"], [
    ["Autoregressive lags", "lag 1, 2, 3, 6, 12, 24, 48, 168 h", "Short-term persistence, same hour "
     "yesterday and last week"],
    ["Rolling statistics", "mean and std over 3, 6, 24, 168 h", "Recent level and volatility"],
    ["Calendar", "hour, weekday, month, weekend, French public holiday", "Routines and seasonality"],
    ["Cyclical encodings", "sin/cos of hour and month", "23:00 and 00:00 are neighbours"],
    ["Electrical context (t−1)", "reactive power, voltage, intensity, 3 sub-meters", "Which "
     "appliances were active"],
], "Selected feature specification (31 features)", [3.6, 5.5, 5.9])

heading("3.2. Design Constraints", 2)
table(["Constraint", "Consideration in the design"], [
    ["Standards", "ML life-cycle stages follow ISO/IEC 23053; software quality attributes of ISO/IEC 25010 "
     "(reliability, maintainability) drive testing and modularity; metrics exposed in the OpenMetrics format."],
    ["Economic / Cost", "Must run on a student laptop, free CI runners and a free cloud tier: no GPU, "
     "only open-source tools (scikit-learn, MLflow, DVC, Flask, Docker, Prometheus, Grafana)."],
    ["Environmental", "Low-compute models reduce the energy footprint of retraining; the purpose of the system "
     "is itself to support energy saving (ISO 50001 spirit)."],
    ["Health & Safety", "Advisory system only: it never switches appliances, so incorrect forecasts cannot cause "
     "physical harm."],
    ["Professional / Ethical", "Honest evaluation against naive baselines on unseen future data; no "
     "data leakage; limitations and error differences between groups of hours are disclosed (model card)."],
    ["Fairness & bias", "The data comes from one French household (2006–2010); quality of service is "
     "measured across day type, time of day, season and usage level; the model is not claimed to "
     "generalise to other homes or climates without re-validation."],
    ["Privacy", "Fine-grained meter data can reveal occupancy and routines. Only a public anonymised "
     "dataset is used; the prediction log stores forecasts, never submitted readings; deployment on real "
     "data must follow India's Digital Personal Data Protection Act, 2023 [25]."],
    ["Security", "All API input is validated (types, ranges, minimum history, duplicates); request size is "
     "limited; only models produced by this pipeline are loaded; rejected requests are monitored."],
    ["Manufacturability / Deployability", "Must be reproducible on any machine: pinned configuration, "
     "DVC pipeline and lock file, Docker image, single-command reproduction."],
    ["Regulatory", "Indian smart meters follow IS 16444; the API input schema maps to the hourly load "
     "profile such meters provide."],
], "Design constraints", [4.0, 11.0])

heading("3.3. Analysis of Features and finalization subject to constraints", 2)
bullets([
    "**Removed:** minute-level forecasting. At one-minute resolution the series is dominated by appliance "
    "switching noise; hourly resolution matches billing and demand-response use and reduces data 60×.",
    "**Removed:** weather features. They would improve accuracy but no free, historical, hourly weather "
    "feed for Sceaux 2006–2010 could be integrated within the cost constraint.",
    "**Removed:** deep LSTM model as the production candidate, due to the no-GPU and explainability "
    "constraints (kept as future work).",
    "**Modified:** missing data handling. Gaps up to 6 h are time-interpolated; longer outages are filled "
    "with the same hour of the previous week to preserve the daily profile.",
    "**Modified:** outliers are reported and kept (they are real peaks); feature scaling is applied only "
    "where the algorithm needs it (linear models, KNN, SVM).",
    "**Added:** a peak-hour classifier, because demand response needs a yes/no alert as well as a number.",
    "**Added:** prediction intervals (±1.96σ_r√h) so users can see forecast uncertainty.",
    "**Added:** automated leakage test, quality gates in CI, and a bias analysis across groups of hours to "
    "satisfy the professional/ethical constraints.",
])

heading("3.4. Design Flow", 2)
para(f"Two alternative designs were developed to meet the specification (Figure {nf()}).")
figure(DIAG / "fig_design_alternatives.png", "Alternative designs considered")
para("**Design A – Notebook-centric deep learning.** All steps (cleaning, features, an LSTM model) live "
     "in a single Jupyter notebook. The trained network is pickled manually and predictions are produced "
     "with an ad-hoc script. This is the typical approach in the literature [11], [12].")
para("**Design B – Modular MLOps pipeline.** Each stage is an independent, tested Python module driven by "
     "one configuration file (params.yaml) and declared as a DVC pipeline. All course algorithms are compared "
     "and tuned under MLflow tracking; the best is registered and served by a Flask REST API in Docker, "
     "deployed to the cloud behind a React web app; GitHub Actions automates the pipeline, and Prometheus, "
     "Grafana and a drift monitor close the loop.")

heading("3.5. Design selection", 2)
table(["Criterion", "Design A", "Design B"], [
    ["Expected accuracy", "High (with tuning)", "High (gradient boosting)"],
    ["Training cost / hardware", "GPU preferred, minutes–hours", "CPU, about 4 min including tuning"],
    ["Reproducibility", "Low: hidden notebook state", "High: params.yaml + DVC + Docker"],
    ["Experiment tracking", "Manual", "Automatic (MLflow)"],
    ["Deployment", "Manual script", "Flask API + container + cloud + web app"],
    ["Testing / CI", "None", "11 automated tests, CI quality gates"],
    ["Monitoring & retraining", "None", "Prometheus, Grafana, PSI drift, weekly CT"],
    ["Explainability & ethics", "Low", "Feature importance, bias analysis, model card"],
    ["Fit with constraints", "Violates cost & reproducibility", "Satisfies all"],
], "Comparison of alternative designs", [5.0, 5.0, 5.0])
para("**Design B is selected.** It meets every constraint in Section 3.2, its accuracy is competitive, and "
     "it delivers the operational capabilities (automation, deployment, monitoring, responsible AI) that are "
     "the core outcomes of the course and that the literature identifies as missing.")

heading("3.6. Implementation plan/methodology", 2)
para(f"Figure {nf()} shows the algorithm executed by the pipeline and Figure {nf(2)} the detailed block diagram.")
figure(DIAG / "fig_flowchart.png", "Flowchart of the training and deployment algorithm", 10)
figure(DIAG / "fig_architecture.png", "Detailed block diagram of the MLOps architecture")
para(f"The system is further described by a context data-flow diagram (Figure {nf()}), a level-1 data-flow "
     f"diagram showing processes and data stores (Figure {nf(2)}), a use-case diagram (Figure {nf(3)}), a "
     f"sequence diagram of a forecast request (Figure {nf(4)}), the deployment diagram (Figure {nf(5)}), the "
     f"CI/CD pipeline (Figure {nf(6)}) and the DVC stage graph (Figure {nf(7)}).")
figure(DIAG / "dfd_level0.png", "Data-flow diagram, level 0 (context)", 15)
figure(DIAG / "dfd_level1.png", "Data-flow diagram, level 1", 15.5)
figure(DIAG / "use_case.png", "Use-case diagram", 14)
figure(DIAG / "sequence_forecast.png", "Sequence diagram: serving a forecast request", 15.5)
figure(DIAG / "deployment.png", "Deployment diagram: Docker Compose, Vercel and Kubernetes", 15.5)
figure(DIAG / "cicd.png", "CI/CD pipeline", 15.5)
figure(DIAG / "dvc_dag.png", "DVC pipeline graph", 15)
para(f"Each pipeline stage follows its own algorithm, shown as flowcharts in Figures {nf()}–{nf(5)}.")
figure(DIAG / "flow_preprocess.png", "Flowchart of the preprocessing stage", 10)
figure(DIAG / "flow_training.png", "Flowchart of the training and tuning stage", 10.5)
figure(DIAG / "flow_classifier.png", "Flowchart of the peak-hour classifier stage", 10)
figure(DIAG / "flow_prediction.png", "Flowchart of serving a prediction", 11.5)
figure(DIAG / "flow_monitoring.png", "Flowchart of drift monitoring and the retraining decision", 11.5)
para("**Data split.** To respect time ordering, data is split chronologically: training up to 31 Dec 2009 "
     f"(26,503 h), validation Jan–May 2010 (3,624 h) and test {meta['test_period'][0][:10]} to "
     f"{meta['test_period'][1][:10]} (4,294 h). Models are selected on validation, refit on "
     "train+validation, and scored once on test. A random split is deliberately avoided: it would leak "
     "future information into training and overstate accuracy.")
para("**Regression models.** Persistence (ŷₜ = yₜ₋₁) and seasonal-naive (ŷₜ = yₜ₋₂₄) "
     "baselines; linear regression and ridge regression (standardised inputs); decision tree; k-nearest "
     "neighbours (k = 10, standardised); support vector regression (RBF kernel, C = 1, ε = 0.1, standardised); "
     "random forest (200 trees, depth 16) [8]; histogram gradient boosting (500 iterations, η = 0.05, "
     "31 leaves) [9], [18]. All are implemented with scikit-learn [20].")
para("**Hyper-parameter tuning.** GridSearchCV searches the grids in params.yaml using TimeSeriesSplit with "
     "five folds, so every fold trains on the past and validates on the following period, which is the "
     "time-series form of cross-validation [21]. Tuned and default variants are both compared on the "
     "validation window; regularisation is controlled through ridge α, tree depth and minimum leaf size.")
para("**Peak-hour classification.** An hour is labelled a peak hour when consumption exceeds "
     f"τ = {clf['threshold_kw']:.2f} kW, the 75th percentile of training consumption. Logistic regression, a "
     "default decision tree and a GridSearchCV-tuned decision tree (F1 scoring, balanced class weights "
     "allowed) are compared on validation F1.")
para("**Metrics.** Regression: MAE = (1/n)Σ|yₜ − ŷₜ|, MSE = (1/n)Σ(yₜ − ŷₜ)², RMSE = √MSE, "
     "R² and MAPE; RMSE selects the model because peak errors matter most for demand response. "
     "Classification: accuracy = (TP+TN)/n, precision = TP/(TP+FP), recall = TP/(TP+FN), F1 = harmonic mean "
     "of precision and recall, plus the confusion matrix and ROC-AUC [21], [22].")
para("**Monitoring.** PSI = Σ(qᵢ − pᵢ)·ln(qᵢ/pᵢ) over 10 quantile bins of the training "
     "distribution; PSI > 0.2 signals significant drift. Retraining is recommended when any key feature "
     "drifts or recent 30-day MAE degrades more than 20%. At run time Prometheus scrapes request, latency, "
     "forecast and alert counters from the API every 15 s.")
table(["Module", "Responsibility"], [
    ["src/data_ingestion.py", "Download the UCI archive, convert to CSV, verify schema"],
    ["src/preprocess.py", "Parse, fill gaps, resample hourly, IQR outlier report"],
    ["src/features.py", "Build 31 leakage-free features"],
    ["src/train.py", "9 regressors, GridSearchCV, MLflow tracking and model registry"],
    ["src/classify.py", "Peak-hour classifier and classification metrics"],
    ["src/monitor.py", "PSI drift and performance degradation, retraining decision"],
    ["src/evaluate.py", "EDA, diagnostics, validation curve, classification and bias figures"],
    ["src/predict.py", "Recursive multi-step forecasting with intervals and peak alerts"],
    ["serving/main.py, app.py", "Flask REST API, Pydantic validation, Prometheus metrics, prediction log"],
    ["web/ (React), dashboard/app.py", "Web application (Vercel) and Streamlit dashboard"],
    ["monitoring/, docker-compose.yml, k8s/", "Prometheus, Grafana, five-service stack, Kubernetes manifests"],
    ["dvc.yaml, .github/workflows/ci.yml", "Versioned pipeline and CI/CD automation"],
    ["tests/test_pipeline.py", "11 unit, leakage, classification, API and monitoring tests"],
], "Software modules of the implementation", [6.0, 9.0])

# ================= CHAPTER 4 =================
chapter(4, "RESULTS ANALYSIS AND VALIDATION")
heading("4.1. Implementation of solution", 2)
para("**Modern tools used.** Analysis: pandas, NumPy, Matplotlib, Seaborn, Plotly. Design drawings: "
     "architecture, flow and Gantt diagrams generated in Matplotlib. Modelling: scikit-learn. Experiment "
     "management and versioning: MLflow (tracking and model registry), DVC, Git/GitHub. Deployment: Flask, "
     "gunicorn, Docker, Docker Compose, Kubernetes manifests, Vercel. Operations: GitHub Actions, Prometheus, "
     "Grafana. Testing and validation: pytest, PSI drift analysis. Report preparation: python-docx, generated "
     "directly from pipeline outputs, so every number in this chapter is reproducible with dvc repro.")

heading("4.1.1. Data preparation", 3)
para(f"{stats['raw_missing_pct']}% of minute readings were missing ('?'). After cleaning and hourly resampling "
     f"the series has {stats['hourly_rows']:,} gap-free records with mean 1.09 kW (median 0.80 kW, max 6.56 kW). "
     "Feature building drops the first 168 h (needed for the weekly lag), leaving 34,421 modelling rows. "
     f"Figure {nf()} shows that current intensity is almost perfectly correlated with active power, while "
     "voltage is weakly negatively correlated.")
figure(FIG / "fig_eda_correlation.png", "Correlation matrix of the measured variables", 12)

heading("4.1.2. Model development and comparison", 3)
para(f"Table {nt()} compares all nine regression models with default hyper-parameters; every run is logged in "
     f"MLflow (Figure {nf(2)} in this section shows the tracking UI).")
rows = []
for n, m in V.items():
    r = [LBL[n], f"{m['train_mae']:.3f}", f"{m['mae']:.3f}", f"{m['mse']:.3f}", f"{m['rmse']:.3f}", f"{m['r2']:.3f}",
         f"{m['fit_seconds']:.1f}"]
    rows.append([f"**{c}**" for c in r] if n == meta["model_name"] else r)
table(["Model", "Train MAE", "Val MAE", "Val MSE", "Val RMSE", "Val R²", "Fit (s)"], rows,
      "Training and validation performance of all models (default hyper-parameters)", [4.8, 1.7, 1.7, 1.7, 1.8, 1.6, 1.4])
figure(FIG / "fig_model_comparison.png", "Validation MAE and R² of all models (best in orange, baselines in grey)")
figure(SHOT / "mlflow.png", "MLflow tracking UI: runs of every model, tuned variant and final model", 15)
para(f"Gradient boosting has the lowest validation RMSE ({V['hist_gradient_boosting']['rmse']:.3f} kW), "
     f"closely followed by random forest and the SVM. Linear and ridge regression are identical at α = 1 and "
     f"form a strong linear reference (R² {V['linear_regression']['r2']:.3f}). The seasonal-naive model has "
     "negative R²: yesterday's same hour is a worse predictor than the mean, which confirms that household "
     "load is driven by irregular routines.")

heading("4.1.3. Over-fitting, under-fitting and regularisation", 3)
para(f"Figure {nf()} compares training and validation error. The default decision tree has a training MAE of "
     f"{V['decision_tree']['train_mae']:.3f} kW (it memorises every training hour) but the worst validation MAE "
     f"of the machine-learning models ({V['decision_tree']['mae']:.3f} kW): a textbook case of over-fitting. "
     "Linear regression shows the opposite pattern, with training and validation errors almost equal, "
     f"indicating mild under-fitting. The validation curve in Figure {nf(2)} shows both regimes for the tree: "
     "shallow trees under-fit, deep trees over-fit, and the best depth lies in between.")
figure(FIG / "fig_train_vs_val.png", "Training vs validation MAE of each model (default hyper-parameters)")
figure(FIG / "fig_validation_curve.png", "Validation curve of the decision tree: error against max_depth")
lc = AN["learning_curves"]
para(f"Learning curves (Figure {nf()}) show how training and cross-validation error change as the models see more "
     "history. For linear regression both curves meet quickly at a high error, the signature of under-fitting "
     "(more data does not help). The unrestricted decision tree keeps a near-zero training error and a large gap, "
     "the signature of over-fitting. Random forest and gradient boosting narrow the gap as data grows "
     f"(gradient boosting CV MAE {lc['hist_gradient_boosting']['cv_mae'][0]:.3f} → "
     f"{lc['hist_gradient_boosting']['cv_mae'][-1]:.3f} kW), so more historical data would still help them.")
figure(AFIG / "learning_curves.png", "Learning curves with time-series cross-validation", 16)

heading("4.1.4. Hyper-parameter tuning with GridSearchCV", 3)
para(f"Table {nt()} lists the GridSearchCV results (five time-series folds on the training period).")
table(["Model", "Best parameters", "Grid size", "CV MAE", "Val MAE default", "Val MAE tuned"],
      [[LBL[n], ", ".join(f"{k}={v}" for k, v in t["best_params"].items()), str(t["candidates_tried"]),
        f"{t['cv_mae']:.3f}", f"{t['default_val']['mae']:.3f}", f"{t['tuned_val']['mae']:.3f}"] for n, t in TUN.items()],
      "Hyper-parameter tuning results", [3.6, 4.6, 1.4, 1.6, 1.9, 1.9])
figure(FIG / "fig_tuning.png", "Validation MAE before and after GridSearchCV tuning")
hp = AN["hyperparameters"]
para(f"Figure {nf()} shows how the error responds to individual hyper-parameters. The decision-tree heat map "
     "covers the whole GridSearchCV grid: deep trees with tiny leaves are worst, and a large minimum leaf size makes "
     f"depth almost irrelevant. KNN improves steadily from k = 1 (validation MAE {hp['knn_k']['val_mae'][0]:.3f}) to "
     f"k = 80 ({hp['knn_k']['val_mae'][-1]:.3f}). Ridge is flat until α becomes very large, when strong "
     "regularisation starts to under-fit.")
figure(AFIG / "hyperparameter_curves.png", "Hyper-parameter response: decision-tree grid, KNN neighbours, ridge alpha", 16)
cvf = AN["cv_folds"]
para(f"Table {nt()} and Figure {nf()} give the error in each time-series fold for the tuned models. Every model is "
     "worse in fold 2, whose validation window is a winter period with high, variable consumption, and the ranking "
     "of models is consistent across folds, which supports the model choice.")
table(["Model", "Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5", "Mean", "Std"],
      [[LBL[n], *[f"{v:.3f}" for v in f], f"{sum(f) / len(f):.3f}", f"{pd.Series(f).std():.3f}"] for n, f in cvf.items()],
      "Cross-validation MAE (kW) per time-series fold, tuned models", [4.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.3])
figure(AFIG / "cv_folds.png", "MAE in each time-series cross-validation fold")
para(f"Tuning helps most where the default was poorly regularised: the decision tree improves from "
     f"{DT['default_val']['mae']:.3f} to {DT['tuned_val']['mae']:.3f} kW once depth and minimum leaf size are "
     f"limited. KNN gains by averaging over more neighbours with distance weighting; ridge prefers strong "
     f"regularisation (α = {TUN['ridge_regression']['best_params']['alpha']}) but gains little. For random forest "
     "and gradient boosting the defaults were already well chosen, and the small differences are within the "
     "noise between cross-validation and the validation window. Default gradient boosting therefore remains the "
     "selected model.")

heading("4.1.5. Test-set results of the forecaster", 3)
para(f"The selected model was refit on train+validation and evaluated once on the untouched test period (Table {nt()}).")
table(["Model", "MAE (kW)", "MSE", "RMSE (kW)", "R²", "MAPE (%)"], [
    ["Persistence baseline", f"{B['mae']:.3f}", f"{B['mse']:.3f}", f"{B['rmse']:.3f}", f"{B['r2']:.3f}", f"{B['mape']:.1f}"],
    ["**Histogram gradient boosting**", f"**{T['mae']:.3f}**", f"**{T['mse']:.3f}**", f"**{T['rmse']:.3f}**",
     f"**{T['r2']:.3f}**", f"**{T['mape']:.1f}**"],
    ["Improvement", f"{MAE_GAIN:.1f}%", f"{(1 - T['mse'] / B['mse']) * 100:.1f}%", f"{RMSE_GAIN:.1f}%",
     f"+{T['r2'] - B['r2']:.3f}", "–"],
], "Held-out test performance (Jun–Nov 2010)", [5.0, 2.0, 2.0, 2.0, 2.0, 2.0])
TA = AN["test_all_models"]
para(f"To confirm that selecting on the validation window did not favour one model by chance, every model "
     f"(default and tuned) was also refit on train+validation and scored on the test period. Tables {nt()} and "
     f"{nt(2)} give the complete results; Figure {nf()} compares the main ones.")
order = list(CANDIDATES_ORDER) + [n for n in TA if n.endswith("_tuned")]
nm = lambda n: LBL[n.removesuffix("_tuned")] + (" (tuned)" if n.endswith("_tuned") else "")
table(["Model", "MAE", "MSE", "RMSE", "R²", "MAPE %", "Median AE", "Max error"],
      [[f"**{nm(n)}**" if n == meta["model_name"] else nm(n), *[f"{TA[n][k]:.3f}" for k in ["mae", "mse", "rmse", "r2"]],
        f"{TA[n]['mape']:.1f}", f"{TA[n]['median_ae']:.3f}", f"{TA[n]['max_error']:.2f}"] for n in order],
      "Test-set error metrics of every model (kW, kW²)", [5.0, 1.4, 1.4, 1.5, 1.4, 1.5, 1.7, 1.6])
table(["Model", "Accuracy (1 − WAPE)", "Within ±0.25 kW", "Within ±0.5 kW", "Explained variance", "Fit (s)", "ms / prediction"],
      [[f"**{nm(n)}**" if n == meta["model_name"] else nm(n), f"{TA[n]['accuracy_1_minus_wape'] * 100:.1f}%",
        f"{TA[n]['within_0_25kw_pct']:.1f}%", f"{TA[n]['within_0_5kw_pct']:.1f}%", f"{TA[n]['explained_variance']:.3f}",
        f"{TA[n]['fit_seconds']:.1f}", f"{TA[n]['predict_ms_per_row']:.4f}"] for n in order],
      "Test-set accuracy-style metrics, training time and prediction time of every model",
      [4.6, 2.2, 1.9, 1.9, 2.0, 1.3, 1.6])
best_t = TA[meta["model_name"]]
para(f"Accuracy-style metrics make the regression results easier to read. Accuracy as 1 − WAPE (weighted absolute "
     f"percentage error) is the share of total consumption predicted correctly: "
     f"{best_t['accuracy_1_minus_wape'] * 100:.1f}% for the deployed model against "
     f"{TA['persistence_baseline']['accuracy_1_minus_wape'] * 100:.1f}% for persistence. "
     f"{best_t['within_0_25kw_pct']:.1f}% of hourly forecasts are within ±0.25 kW of the actual value and "
     f"{best_t['within_0_5kw_pct']:.1f}% within ±0.5 kW. Gradient boosting is also best on the test set, ahead of "
     "random forest and SVM, and the tuned variants rank as on validation. This confirms the selection was not a "
     "lucky choice.")
figure(AFIG / "test_metrics_all_models.png", "Test MAE, RMSE, R² and accuracy (1 − WAPE) of every model", 16)
para(f"Figures {nf()}–{nf(3)} show the behaviour of each model on the test set: a week of forecasts, the "
     "predicted-vs-actual scatter, and the residual distribution.")
figure(AFIG / "per_model_forecasts.png", "Forecasts of every model over the last seven days of the test set", 16)
figure(AFIG / "per_model_scatter.png", "Predicted vs actual for every model (test set)", 16)
figure(AFIG / "per_model_residuals.png", "Residual distribution of every model (test set)", 16)
para(f"The forecasts follow the daily cycle and the timing of evening peaks closely (Figure {nf()}). Residuals are "
     f"centred near zero but right-skewed (Figure {nf(2)}): the largest errors are sudden short spikes, such as an "
     "oven or water heater switching on, which are inherently unpredictable from past meter data alone.")
figure(FIG / "fig_actual_vs_pred.png", "Next-hour forecast vs actual consumption, last 14 days of the test set")
figure(FIG / "fig_residuals.png", "Predicted vs actual scatter and residual distribution")
top = ", ".join(f"{k} ({v:.3f})" for k, v in imp.head(4).items())
para(f"**Explainability.** Permutation importance (Figure {nf()}) shows the inputs the model relies on: {top}. "
     "The previous hour's current and power carry the household's present state, the water-heater/AC "
     "sub-meter signals heavy thermal loads, and the hour encodings capture daily routines, which is physically "
     "consistent.")
figure(FIG / "fig_feature_importance.png", "Top 15 features by permutation importance", 13)
hz = AN["horizon"]
para(f"**Error by forecast horizon.** The deployed service forecasts up to 48 hours ahead by feeding each "
     f"prediction back as input (recursive forecasting). Figure {nf()} and Table {nt()} show the error for "
     f"{hz['origins']} random starting points in the test period. Error grows from {hz['mae_by_hour'][0]:.3f} kW one "
     f"hour ahead to {hz['mae_by_hour'][-1]:.3f} kW 24 hours ahead as uncertainty accumulates, but stays below the "
     f"same-hour-yesterday baseline at every horizon ({hz['naive_mae_by_hour'][0]:.3f} kW). This is why the API "
     "widens its prediction interval with √h.")
figure(AFIG / "horizon_curve.png", "Forecast MAE by hours ahead, compared with the seasonal-naive baseline")
table(["Hours ahead", *[str(h) for h in (1, 2, 3, 6, 12, 18, 24)]],
      [["VoltCast MAE (kW)", *[f"{hz['mae_by_hour'][h - 1]:.3f}" for h in (1, 2, 3, 6, 12, 18, 24)]],
       ["Seasonal naive MAE (kW)", *[f"{hz['naive_mae_by_hour'][h - 1]:.3f}" for h in (1, 2, 3, 6, 12, 18, 24)]]],
      "Forecast error by horizon", [4.4, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5])

heading("4.1.6. Peak-hour classification", 3)
cv = clf["validation"]
para(f"Peak hours make up 25% of training hours and {clf['positive_rate_test']*100:.1f}% of the (summer-heavy) "
     f"test period. Table {nt()} compares the classifiers on the validation window.")
table(["Classifier", "Train F1", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
      [[n.replace("_", " ").capitalize(), f"{s['train_f1']:.3f}", f"{s['accuracy']:.3f}", f"{s['precision']:.3f}",
        f"{s['recall']:.3f}", f"{s['f1']:.3f}", f"{s['roc_auc']:.3f}"] for n, s in cv.items()],
      "Peak-hour classifiers on the validation set", [4.4, 1.6, 1.7, 1.7, 1.6, 1.4, 1.8])
bp = clf["best_params"]
para(f"The default decision tree again over-fits (training F1 {cv['decision_tree_default']['train_f1']:.2f}). "
     f"GridSearchCV selects max_depth = {bp['max_depth']}, min_samples_leaf = {bp['min_samples_leaf']} and "
     f"class_weight = {bp['class_weight']}, raising validation F1 from {cv['decision_tree_default']['f1']:.3f} "
     f"to {cv['decision_tree_tuned']['f1']:.3f}, so the tuned tree is selected. On the test period it reaches "
     f"accuracy {CT['accuracy']:.3f}, precision {CT['precision']:.3f}, recall {CT['recall']:.3f}, F1 "
     f"{CT['f1']:.3f} and ROC-AUC {CT['roc_auc']:.3f} (Figure {nf()}).")
figure(FIG / "fig_classification.png", "Peak-hour classifier on the test set: confusion matrix, ROC and precision-recall curves")
CA = AN["classification"]["classifiers"]
para(f"Figure {nf()} compares the confusion matrices of all three classifiers on the test set, and Figure "
     f"{nf(2)} their ROC and precision-recall curves. Table {nt()} is the full classification report of each "
     "classifier, giving precision, recall, F1 and support for both classes plus accuracy and the macro and "
     "weighted averages.")
figure(AFIG / "classifier_confusion_matrices.png", "Confusion matrices of the three classifiers (test set)", 16)
figure(AFIG / "classifier_roc_pr.png", "ROC and precision-recall curves of the three classifiers (test set)", 16)
rows = []
for name, c in CA.items():
    rep_ = c["report"]
    for cls in ["Normal hour", "Peak hour", "macro avg", "weighted avg"]:
        r_ = rep_[cls]
        rows.append([name if cls == "Normal hour" else "", cls, f"{r_['precision']:.3f}", f"{r_['recall']:.3f}",
                     f"{r_['f1-score']:.3f}", f"{int(r_['support'])}"])
    rows.append(["", "accuracy", "", "", f"{rep_['accuracy']:.3f}", f"{int(rep_['macro avg']['support'])}"])
table(["Classifier", "Class", "Precision", "Recall", "F1-score", "Support"], rows,
      "Classification reports on the test set", [4.4, 3.0, 1.9, 1.9, 1.9, 1.9])
sw = AN["classification"]["threshold_sweep"]
para(f"The tuned tree outputs a probability; the alert uses 0.5 as the cut-off. Figure {nf()} and Table {nt()} show "
     "the trade-off for other cut-offs: a lower threshold catches more peaks (recall) but raises more false "
     "alarms (lower precision), and a higher threshold does the opposite. The F1-score is highest near the middle "
     "of the range, which supports the default.")
figure(AFIG / "classifier_threshold_sweep.png", "Precision, recall, F1 and accuracy against the decision threshold")
table(["Threshold", "Precision", "Recall", "F1", "Accuracy"],
      [[f"{r_['threshold']:.2f}", f"{r_['precision']:.3f}", f"{r_['recall']:.3f}", f"{r_['f1']:.3f}", f"{r_['accuracy']:.3f}"]
       for r_ in sw[::2]], "Effect of the probability threshold (tuned decision tree, test set)", [3, 3, 3, 3, 3])
cm = CT["confusion_matrix"]
para(f"Of {cm[1][0] + cm[1][1]} real peak hours, {cm[1][1]} were caught and {cm[1][0]} missed; "
     f"{cm[0][1]} normal hours raised a false alarm. High recall with moderate precision is the safer trade-off "
     "for an alert, since a missed peak costs more than an unnecessary reminder. A simpler rule, flagging hours "
     f"where the regression forecast exceeds τ, is more accurate ({CR['accuracy']:.3f}) but catches only "
     f"{CR['recall']*100:.0f}% of peaks. The deployed system uses the classifier for the next hour, and the "
     "forecast-threshold rule for later hours, where the classifier's inputs are no longer current. On 40 random "
     "test origins this rule was 77% accurate over hours 2–24, against 57% for the classifier.")

heading("4.1.7. Deployment", 3)
para("The forecaster and classifier are served by a Flask REST API (Table "
     f"{nt()}). Every request body is validated with Pydantic, which checks types, non-negative readings, positive "
     "voltage, a horizon of 1–48 h, at least 169 hours of history and no duplicate timestamps. Invalid input "
     "returns HTTP 422 with an explanation.")
table(["Method", "Endpoint", "Purpose"], [
    ["GET", "/api/health", "Liveness and loaded model"],
    ["GET", "/api/model-info", "Model, hyper-parameters, training window, test metrics, peak threshold"],
    ["POST", "/api/predict", "Forecast + peak alerts from user-supplied hourly readings (JSON)"],
    ["GET", "/api/forecast", "Forecast from a point in the test period, returned with actual values"],
    ["GET", "/api/sample-request", "Ready-made request body for /api/predict"],
    ["GET", "/api/metrics", "Prometheus metrics (OpenMetrics text format)"],
], "REST API endpoints", [1.6, 4.4, 9.0])
para("**On-premise (Docker).** A multi-stage Dockerfile builds the React web app with Node and then the Python "
     "runtime with gunicorn. Docker Compose starts five microservices: the API with the web app, the Streamlit "
     "dashboard, the MLflow UI, Prometheus and Grafana. The full stack was built and run locally and every "
     "service responded. **Kubernetes.** k8s/deployment.yaml defines a two-replica Deployment with readiness "
     "and liveness probes on /api/health, a NodePort Service and a CPU-based HorizontalPodAutoscaler (2–5 pods). "
     "The manifests are validated in CI; they were not applied to a live cluster in this project. "
     "**Cloud.** The same Flask app runs as a serverless function on Vercel, with the React app served from its "
     "CDN, at https://voltcast-forecast.vercel.app. A 24-hour forecast request completes in well under one second.")
figure(SHOT / "web_forecast.png", "VoltCast web application (React) with a live 24-hour forecast and peak-hour alerts", 15.5)

heading("4.1.8. CI/CD and versioning", 3)
para("The GitHub Actions workflow runs on every push, on pull requests and weekly (continuous training). It "
     "installs the dependencies, reproduces the pipeline with dvc repro, prints the DVC-tracked metrics, "
     "applies two quality gates (forecaster MAE below the persistence baseline; classifier ROC-AUC above 0.75), "
     "runs the 11 tests, builds the React app, validates the Docker Compose, Prometheus/Grafana and Kubernetes "
     "files, builds the Docker image and uploads the models and reports as artifacts. Vercel's Git integration "
     "provides continuous deployment: every push to the main branch is built and published automatically.")
para("**Versioning.** DVC records each stage's inputs, parameters and outputs in dvc.lock, so a given Git "
     "commit identifies the exact dataset, parameters, models and metrics; dvc repro re-runs only the stages "
     "whose inputs changed. The dataset is kept in the DVC cache rather than in Git. In addition, every trained "
     "forecaster is registered as a new version of voltcast-forecaster in the MLflow model registry.")
para(f"**Automated testing.** Table {nt()} lists the pytest suite.")
table(["Test", "What it validates", "Result"], [
    ["test_resample_fills_gaps", "Missing hours are filled; output is gap-free", "Pass"],
    ["test_features_have_no_leakage", "lag_1 and rolling means use only past values", "Pass"],
    ["test_metrics_perfect_prediction", "Regression metric implementation", "Pass"],
    ["test_psi_detects_shift", "PSI < 0.1 for same distribution, > 0.2 for shifted", "Pass"],
    ["test_classification_scores", "Accuracy, precision, recall and confusion matrix", "Pass"],
    ["test_health_endpoint", "Service liveness and model loading", "Pass"],
    ["test_predict_endpoint_returns_horizon", "Horizon, interval ordering, peak alerts, strictly valid JSON", "Pass"],
    ["test_predict_rejects_short_history", "Too little history returns HTTP 422", "Pass"],
    ["test_predict_rejects_invalid_reading", "Negative voltage returns HTTP 422", "Pass"],
    ["test_prometheus_metrics_exposed", "Prometheus counters and histograms are published", "Pass"],
    ["test_model_beats_baseline", "Quality gate: model MAE < baseline MAE", "Pass"],
], "Automated test suite results", [5.6, 7.4, 2.0])

heading("4.1.9. Monitoring and logging", 3)
para("Prometheus scrapes /api/metrics every 15 seconds (Figure "
     f"{nf()}). The API exports request counts and latency histograms per endpoint, the number of forecast hours "
     "served, the distribution of forecast values, peak alerts, rejected requests and the loaded model's test "
     f"MAE. A provisioned Grafana dashboard visualises them (Figure {nf(2)}); invalid requests sent "
     "on purpose during the test run appear under 'Rejected requests'. Every prediction is also written "
     "as one JSON line to logs/predictions.jsonl (time, model, horizon, forecast values, number of peak hours), "
     "without the submitted meter readings.")
figure(SHOT / "prometheus_targets.png", "Prometheus target page: the VoltCast API is being scraped (state UP)", 14)
figure(SHOT / "grafana.png", "Grafana dashboard “VoltCast – Model Monitoring” fed by Prometheus", 15.5)
para(f"**Data drift.** Figure {nf()} shows the PSI between training and test distributions.")
figure(FIG / "fig_drift_psi.png", "Population Stability Index of monitored features (train vs test)", 13)
drifted = ", ".join(mon["drifted_features"])
para(f"Significant drift was detected in {drifted} (PSI {mon['psi']['Sub_metering_3_lag1']:.2f} for the "
     "water-heater/AC sub-meter). This is genuine seasonal drift: the training data covers three winters, "
     "while the test period is June–November, when heating load is low. The 30-day rolling MAE "
     f"({mon['recent_30d_mae']:.3f} kW) was {mon['mae_degradation_pct']:.1f}% above the reference test MAE as "
     "consumption climbed again in November. The monitor therefore set retraining_recommended = true, which "
     "is the behaviour a production MLOps loop requires.")
dr = AN["drift"]
para(f"Figure {nf()} follows drift month by month through the test period. The water-heater/AC sub-meter is "
     "far from its training distribution in every summer month and moves back towards it in November, as heating "
     f"returns. Test MAE follows the same seasonal pattern ({min(dr['monthly_mae'].values()):.3f}–"
     f"{max(dr['monthly_mae'].values()):.3f} kW; Table {nt()}).")
figure(AFIG / "monthly_drift.png", "PSI against the training data and test MAE, month by month", 16)
mp = dr["monthly_psi"]
feats_ = list(next(iter(mp.values())))
table(["Month", *[f.replace("_lag1", "").replace("_", " ") for f in feats_], "MAE (kW)"],
      [[m, *[f"{mp[m][f]:.3f}" for f in feats_], f"{dr['monthly_mae'][m]:.3f}"] for m in mp],
      "Monthly PSI of monitored features and monthly test MAE", [1.8, *[1.9] * len(feats_), 1.6])

heading("4.1.10. Ethical considerations", 3)
fs = fair["slices"]
para(f"**Fairness and bias.** The system makes no decisions about people, so fairness is assessed as equal "
     f"quality of service across conditions [24]. Figure {nf()} breaks the test error down by group. Error is "
     f"lowest at night ({fs['Time of day']['Night 0-5']['mae']:.2f} kW) and highest in the evening "
     f"({fs['Time of day']['Evening 18-23']['mae']:.2f} kW) and in high-use hours "
     f"({fs['Usage level']['High use']['mae']:.2f} kW), and higher at weekends "
     f"({fs['Day type']['Weekend']['mae']:.2f} vs {fs['Day type']['Weekday']['mae']:.2f} kW) and in autumn "
     f"({fs['Season']['Autumn (Sep-Nov)']['mae']:.2f} vs {fs['Season']['Summer (Jun-Aug)']['mae']:.2f} kW). "
     "The forecasts are therefore least reliable exactly when demand is highest, and this is stated to "
     "users. There is also sampling bias: one French household from 2006–2010 cannot represent Indian homes "
     "with air-conditioning loads, so the model must be re-validated before any wider use.")
figure(FIG / "fig_fairness.png", "Test error across sub-groups (dashed line: overall MAE)", 15.5)
para("**Transparency.** Feature importance is published, every number is reproducible from the repository, "
     "forecasts carry a 95% interval, and a model card (MODEL_CARD.md) documents intended use, data, "
     "performance, limitations and monitoring. **Privacy.** Smart-meter data can reveal when residents are "
     "home or asleep. The project uses only a public anonymised dataset; the API does not store submitted "
     "readings; real deployments should aggregate data, obtain consent and comply with the Digital Personal "
     "Data Protection Act, 2023 [25]. **Security.** Input validation, a 2 MB request limit, loading only models "
     "produced by the pipeline, and monitoring of rejected requests protect the service against malformed or "
     "abusive input.")
figure(SHOT / "web_ethics.png", "Responsible-AI section of the web application", 15.5)

heading("4.1.11. Execution evidence (terminal output)", 3)
para("The figures in this section are the terminal output of the commands that build, test, deploy and "
     "monitor the system, captured while they ran on the development machine (Windows 11, PowerShell). Each "
     "image shows the exact command line and its unedited output; long outputs are split into numbered parts. "
     "The raw text of every capture is kept in reports/terminal/ and the script that produced them is "
     "report/capture_terminal.py.")
terminal("env_versions", "Python and library versions used")
terminal("git_log", "Git history of the project repository")
terminal("dvc_dag", "dvc dag: the pipeline stages and their dependencies")
terminal("dvc_repro", "dvc repro --force: full pipeline run (ingest, preprocess, features, training with GridSearchCV, "
                      "classifier, monitoring, evaluation, export, analysis)")
terminal("dvc_status", "dvc status after the run")
terminal("dvc_metrics", "dvc metrics show: every tracked metric")
terminal("pytest", "pytest -v: the automated test suite")
terminal("mlflow_runs", "MLflow: tracked runs of the forecasting experiment")
terminal("mlflow_registry", "MLflow model registry: versions of voltcast-forecaster")
terminal("npm_build", "Building the React web application")
terminal("docker_up", "docker compose up: building the image and starting the five services")
terminal("docker_images", "Docker image and Compose services")
terminal("api_health", "Flask API: health check and model information")
terminal("api_forecast", "Flask API: 4-hour forecast with peak alerts")
terminal("api_predict", "Flask API: POST /api/predict with one week of readings")
terminal("api_invalid", "Flask API: invalid requests are rejected with HTTP 422")
terminal("prometheus_metrics", "Prometheus metrics exposed by the API at /api/metrics")
terminal("prometheus_query", "Prometheus: scrape target health and a PromQL query")
terminal("prediction_log", "Prediction log (logs/predictions.jsonl)")
terminal("grafana_api", "Grafana: provisioned dashboard and data source")
terminal("k8s_validate", "Kubernetes manifest and Docker Compose configuration validation")
terminal("ci_runs", "GitHub Actions: recent CI/CD runs")
terminal("ci_run_view", "GitHub Actions: steps of the latest run")
terminal("vercel_ls", "Vercel: production deployments created from GitHub pushes")
terminal("live_site", "The live deployment answering requests")

heading("4.1.11. Achievement of objectives", 3)
table(["Objective (Section 2.6)", "Target", "Achieved"], [
    ["Clean hourly series", "Gap-free, documented", f"{stats['hourly_rows']:,} hourly rows, 0 gaps, outlier report"],
    ["Feature engineering", "≥ 25 leakage-free", "31 features, leakage test passes"],
    ["Course algorithms compared and tuned", "LR, DT, KNN, SVM, RF", "9 regressors, 5 tuned with GridSearchCV"],
    ["Improvement over baseline", "≥ 15% MAE", f"{MAE_GAIN:.1f}% MAE, {RMSE_GAIN:.1f}% RMSE"],
    ["Peak-hour classifier", "Accuracy, P, R, F1, ROC", f"Accuracy {CT['accuracy']:.3f}, ROC-AUC {CT['roc_auc']:.3f}"],
    ["Flask API, Docker, cloud", "< 1 s response", "Flask + Docker + Vercel, sub-second"],
    ["CI/CD and versioning", "Pipeline + gates + tests + build", "GitHub Actions, DVC, MLflow registry"],
    ["Monitoring and logging", "Prometheus, Grafana, drift", f"Dashboard live, {len(mon['drifted_features'])} drifted features flagged"],
    ["Ethics", "Fairness, privacy, security", "Bias analysis, model card, validation"],
], "Achievement of objectives", [5.2, 4.0, 5.8])

# ================= CHAPTER 5 =================
chapter(5, "CONCLUSION AND FUTURE WORK")
heading("5.1. Conclusion", 2)
para("VoltCast is a complete, working Household Electricity Consumption Prediction System built on MLOps "
     "principles. From 2.07 million raw smart-meter readings, a DVC-versioned pipeline produces a clean hourly "
     "series and 31 leakage-free features. Nine regression models, covering every algorithm in the course, "
     "are compared and tuned with GridSearchCV under MLflow tracking. The selected gradient boosting model "
     f"forecasts with a test MAE of {T['mae']:.3f} kW and R² of {T['r2']:.3f}, {MAE_GAIN:.1f}% better than the "
     f"persistence baseline. A tuned decision tree flags peak hours with {CT['recall']*100:.0f}% recall and "
     f"ROC-AUC {CT['roc_auc']:.3f}. The models are served by a Flask API in Docker and on the Vercel cloud, "
     "protected by CI/CD quality gates and tests, observed through Prometheus and Grafana, and reviewed for "
     "fairness, privacy and security.")
para("**Expected vs obtained results.** The ≥ 15% improvement over the baseline was met. However, the absolute "
     f"R² (~{T['r2']:.2f}) is lower than the >0.9 values sometimes reported in the literature, for three "
     "reasons. (i) We forecast a single household at hourly resolution, where behaviour-driven spikes are "
     "inherently unpredictable. (ii) We use a strict chronological split and refuse features computed from the "
     "hour being predicted, whereas some studies shuffle data or use same-hour intensity, which inflates "
     "scores. (iii) The test period has a different seasonal distribution from training, which the monitor "
     "correctly flagged. MAPE (~39%) is inflated by very low night-time loads, where small absolute errors "
     "become large percentages; MAE, MSE and R² are the more meaningful measures here. Hyper-parameter tuning "
     "helped the poorly regularised models (decision tree, KNN) but not the ensembles, whose defaults were "
     "already near-optimal. The peak classifier's precision on the test period is lower than on validation "
     "because summer has fewer peak hours, which is another effect of seasonal drift.")
heading("5.2. Future work", 2)
bullets([
    "**Weather and occupancy inputs:** add outdoor temperature (heating/cooling degree hours) and holiday "
    "calendars from an API; expected to reduce winter peak error substantially.",
    "**Deep and probabilistic models:** evaluate LSTM/Temporal Fusion Transformer and quantile gradient "
    "boosting for calibrated prediction intervals [12], [15].",
    "**Multi-household, Indian data:** train a global model across many homes using data from IS 16444 "
    "smart meters, with federated learning to protect privacy.",
    "**Automated retraining:** connect the monitor's retraining flag to a workflow trigger and promote a new "
    "model only if it beats the current one (champion/challenger) in the MLflow registry.",
    "**Production Kubernetes:** apply the manifests to a managed cluster (AWS EKS, GCP GKE or Azure AKS) "
    "with Prometheus service discovery and Grafana alerting rules.",
    "**Actionable recommendations:** combine forecasts with time-of-day tariffs to suggest the cheapest "
    "time to run flexible appliances.",
])

# ================= REFERENCES =================
unnumbered("REFERENCES")
refs = [
    "International Energy Agency, “Buildings – Energy System,” IEA, Paris. [Online]. Available: https://www.iea.org/energy-system/buildings",
    "Central Electricity Authority, “General Review (All India Electricity Statistics),” Ministry of Power, Government of India, New Delhi.",
    "Ministry of Power, Government of India, “Revamped Distribution Sector Scheme (RDSS),” 2021. [Online]. Available: https://recindia.nic.in/revamped-distribution-sector-scheme",
    "D. Sculley et al., “Hidden technical debt in machine learning systems,” in Advances in Neural Information Processing Systems 28, 2015.",
    "D. Kreuzberger, N. Kühl and S. Hirschl, “Machine Learning Operations (MLOps): Overview, definition, and architecture,” IEEE Access, vol. 11, pp. 31866–31879, 2023.",
    "G. E. P. Box and G. M. Jenkins, Time Series Analysis: Forecasting and Control. San Francisco: Holden-Day, 1970.",
    "G. W. Hart, “Nonintrusive appliance load monitoring,” Proceedings of the IEEE, vol. 80, no. 12, pp. 1870–1891, 1992.",
    "L. Breiman, “Random forests,” Machine Learning, vol. 45, no. 1, pp. 5–32, 2001.",
    "J. H. Friedman, “Greedy function approximation: A gradient boosting machine,” Annals of Statistics, vol. 29, no. 5, pp. 1189–1232, 2001.",
    "G. Hebrail and A. Berard, “Individual Household Electric Power Consumption,” UCI Machine Learning Repository, 2012. doi: 10.24432/C58K54.",
    "D. L. Marino, K. Amarasinghe and M. Manic, “Building energy load forecasting using deep neural networks,” in Proc. IECON 2016 – 42nd Annual Conference of the IEEE Industrial Electronics Society, 2016, pp. 7046–7051.",
    "T.-Y. Kim and S.-B. Cho, “Predicting residential energy consumption using CNN-LSTM neural networks,” Energy, vol. 182, pp. 72–81, 2019.",
    "K. Amasyali and N. M. El-Gohary, “A review of data-driven building energy consumption prediction studies,” Renewable and Sustainable Energy Reviews, vol. 81, pp. 1192–1205, 2018.",
    "M. Zaharia et al., “Accelerating the machine learning lifecycle with MLflow,” IEEE Data Engineering Bulletin, vol. 41, no. 4, pp. 39–45, 2018.",
    "T. Hong and S. Fan, “Probabilistic electric load forecasting: A tutorial review,” International Journal of Forecasting, vol. 32, no. 3, pp. 914–938, 2016.",
    "R. J. Hyndman and G. Athanasopoulos, Forecasting: Principles and Practice, 3rd ed. Melbourne: OTexts, 2021.",
    "C. Fan, F. Xiao and S. Wang, “Development of prediction models for next-day building energy consumption and peak power demand using data mining techniques,” Applied Energy, vol. 127, pp. 1–10, 2014.",
    "G. Ke et al., “LightGBM: A highly efficient gradient boosting decision tree,” in Advances in Neural Information Processing Systems 30, 2017.",
    "S. Rabanser, S. Günnemann and Z. Lipton, “Failing loudly: An empirical study of methods for detecting dataset shift,” in Advances in Neural Information Processing Systems 32, 2019.",
    "F. Pedregosa et al., “Scikit-learn: Machine learning in Python,” Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011.",
    "A. Géron, Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 3rd ed. Sebastopol, CA: O'Reilly Media, 2022.",
    "A. C. Müller and S. Guido, Introduction to Machine Learning with Python: A Guide for Data Scientists. Sebastopol, CA: O'Reilly Media, 2016.",
    "E. Ameisen, Building Machine Learning Powered Applications: Going from Idea to Product. Sebastopol, CA: O'Reilly Media, 2020.",
    "A. Ng, Machine Learning Yearning, 2018. [Online]. Available: https://www.deeplearning.ai/machine-learning-yearning/",
    "Government of India, “The Digital Personal Data Protection Act, 2023,” Act No. 22 of 2023, Ministry of Law and Justice, New Delhi, 2023.",
]
for i, r in enumerate(refs, 1):
    p = para(f"[{i}] {r}", 11, align=WD_ALIGN_PARAGRAPH.LEFT, spacing=1.15, after=6)
    p.paragraph_format.left_indent = Cm(0.9)
    p.paragraph_format.first_line_indent = Cm(-0.9)

# ================= APPENDIX =================
unnumbered("APPENDIX", "A")
heading("1. Plagiarism Report", 2)
para()
heading("2. Design Checklist", 2)
table(["Item", "Status"], [
    ["Problem identified and justified with documented sources", "✔"],
    ["Literature survey with comparison of existing solutions", "✔"],
    ["Measurable objectives defined and mapped to course outcomes", "✔"],
    ["Design constraints (standards, cost, ethics, privacy, security, safety) considered", "✔"],
    ["At least two alternative designs compared and one selected with reasons", "✔"],
    ["Flowchart / block diagram of implementation", "✔"],
    ["Data preprocessing: missing values, outliers, normalisation, chronological split", "✔"],
    ["Linear regression, decision tree, KNN, SVM and random forest implemented and compared", "✔"],
    ["GridSearchCV tuning, cross-validation, over-/under-fitting analysis", "✔"],
    ["Classification metrics: accuracy, precision, recall, F1, confusion matrix, ROC", "✔"],
    ["Version control with Git; data and pipeline versioning with DVC; MLflow registry", "✔"],
    ["Flask REST API, Docker containerisation, cloud deployment, Kubernetes manifests", "✔"],
    ["CI/CD with GitHub Actions and automatic deployment", "✔"],
    ["Monitoring with Prometheus and Grafana; prediction logging; drift detection", "✔"],
    ["Fairness, transparency, privacy and security analysed (model card)", "✔"],
    ["User manual with step-by-step instructions", "✔"],
], "Design checklist", [12.5, 2.5])

heading("3. Syllabus experiments performed in the project", 2)
para("The practical experiments of MLOps-1 (25CSH-320) were carried out on this project's data instead of the "
     "sample datasets suggested in the syllabus. Table A.2 maps each experiment to the code and the evidence in "
     "this report.")
table(["Exp.", "Aim (syllabus)", "How it was done in VoltCast", "Code", "Evidence"], [
    ["1", "Preprocess data: cleaning, missing values, normalisation, split",
     "1.25% missing readings filled (interpolation + weekly fill), IQR outlier report, StandardScaler for "
     "linear/KNN/SVM, chronological train/validation/test split", "src/preprocess.py, src/train.py", "3.1, 4.1.1"],
    ["2", "Linear regression evaluated with MSE", f"Linear regression: test MSE {TA['linear_regression']['mse']:.3f}, "
     f"MAE {TA['linear_regression']['mae']:.3f}, R² {TA['linear_regression']['r2']:.3f}", "src/train.py", "4.1.2, 4.1.5"],
    ["3", "Decision tree classifier: accuracy, confusion matrix, precision-recall",
     f"Peak-hour decision tree: accuracy {CT['accuracy']:.3f}, precision {CT['precision']:.3f}, recall "
     f"{CT['recall']:.3f}; confusion matrix, ROC and PR curves", "src/classify.py", "4.1.6"],
    ["4", "GridSearchCV tuning; tuned vs default", "Grids for 5 regressors and the classifier, TimeSeriesSplit(5); "
     f"tree MAE {DT['default_val']['mae']:.3f} → {DT['tuned_val']['mae']:.3f}", "src/train.py, src/classify.py", "4.1.4"],
    ["5", "Version-controlled repository; ML in a DevOps pipeline", "Git + GitHub (Kunal039/voltcast), DVC "
     "pipeline, GitHub Actions", "dvc.yaml, .github/", "4.1.8, 4.1.11"],
    ["6", "Containerise the model with Docker", "Multi-stage Dockerfile (Node build + Python runtime), "
     "five-service Docker Compose stack built and run", "Dockerfile, docker-compose.yml", "4.1.7, 4.1.11"],
    ["7", "Flask REST API; Prometheus metrics", "Flask blueprint with 6 endpoints, Pydantic validation, "
     "Prometheus counters and histograms", "serving/main.py, app.py", "4.1.7, 4.1.9"],
    ["8", "CI/CD with GitHub Actions", "dvc repro, quality gates, pytest, React and Docker builds; Vercel "
     "deploys every push", ".github/workflows/ci.yml", "4.1.8, 4.1.11"],
    ["9", "Model versioning with DVC", "dvc.yaml with 9 stages, dvc.lock, dvc metrics; MLflow model registry",
     "dvc.yaml, dvc.lock", "4.1.8, 4.1.11"],
    ["10", "Monitoring and logging with Prometheus and Grafana", "Prometheus scrape config, provisioned Grafana "
     "dashboard, JSON prediction log, PSI drift", "monitoring/, src/monitor.py", "4.1.9, 4.1.11"],
    ["11–15", "Mini project phases 1–5", "Problem and data (Ch. 1–3), models (4.1.2–4.1.6), deployment (4.1.7), "
     "automation (4.1.8), monitoring and ethics (4.1.9–4.1.10)", "whole repository", "Ch. 1–5"],
], "Syllabus experiments mapped to the project", [1.2, 3.4, 5.6, 2.6, 2.2])

heading("4. Source code", 2)
para("This section lists the complete source code of the project as submitted (repository "
     "https://github.com/Kunal039/voltcast). Generated files (trained models, metrics, figures), third-party "
     "packages and the report-generation scripts are not listed.")
for rel, title in [
    ("configs/params.yaml", "central configuration"),
    ("dvc.yaml", "DVC pipeline definition"),
    ("run_pipeline.py", "pipeline runner"),
    ("src/utils.py", "shared helpers"),
    ("src/data_ingestion.py", "download, CSV conversion and schema check"),
    ("src/preprocess.py", "cleaning, resampling and outlier report"),
    ("src/features.py", "feature engineering"),
    ("src/train.py", "model training, tuning and registry"),
    ("src/classify.py", "peak-hour classifier"),
    ("src/monitor.py", "drift monitoring"),
    ("src/evaluate.py", "evaluation figures"),
    ("src/analysis.py", "extended analysis"),
    ("src/predict.py", "recursive forecasting"),
    ("src/export_web.py", "web data export"),
    ("serving/main.py", "Flask REST API"),
    ("app.py", "web entry point"),
    ("tests/test_pipeline.py", "automated tests"),
    ("dashboard/app.py", "Streamlit dashboard"),
    ("web/src/main.jsx", "React entry point"),
    ("web/src/App.jsx", "React application"),
    ("web/src/components/Plot.jsx", "Plotly wrapper component"),
    ("web/src/components/Section.jsx", "shared section component"),
    ("web/src/components/Forecast.jsx", "live forecast component"),
    ("web/src/components/ModelPerformance.jsx", "model performance component"),
    ("web/src/components/PeakAlerts.jsx", "classifier component"),
    ("web/src/components/ResponsibleAI.jsx", "responsible-AI component"),
    ("web/src/index.css", "styles"),
    ("Dockerfile", "container image"),
    ("docker-compose.yml", "multi-service stack"),
    ("monitoring/prometheus.yml", "Prometheus scrape configuration"),
    ("monitoring/grafana/provisioning/datasources/prometheus.yml", "Grafana data source"),
    ("k8s/deployment.yaml", "Kubernetes manifests"),
    (".github/workflows/ci.yml", "CI/CD workflow"),
    ("vercel.json", "Vercel build configuration"),
    ("requirements.txt", "serving dependencies"),
    ("requirements-dev.txt", "development dependencies"),
]:
    code_listing(rel, title)

# ================= USER MANUAL =================
unnumbered("USER MANUAL", "M")
para("**Hosted version (no installation):** open https://voltcast-forecast.vercel.app for the web application "
     "and live forecasts. The API is at https://voltcast-forecast.vercel.app/api (for example /api/health). "
     "Source code: https://github.com/Kunal039/voltcast.")
para("**Prerequisites:** Python 3.10 or newer, Node.js 20, about 1 GB free disk space, and internet access for "
     "the first run (dataset download, ~20 MB). Docker Desktop is needed for the monitoring stack.")
steps = [
    ("Get the project", "git clone https://github.com/Kunal039/voltcast.git   then open a terminal inside the folder."),
    ("Install dependencies", "pip install -r requirements-dev.txt"),
    ("Reproduce the pipeline", "dvc repro   (downloads the dataset as CSV, preprocesses, builds features, trains "
     "and tunes all models, trains the peak classifier, runs monitoring and creates all figures; about 4 minutes). "
     "dvc metrics show prints the tracked metrics."),
    ("Run the tests", "pytest -q   (11 tests should pass)"),
    ("Inspect experiments in MLflow", "mlflow ui --backend-store-uri sqlite:///mlflow.db   then open "
     "http://localhost:5000 to compare runs and see the registered model voltcast-forecaster."),
    ("Start the web app and Flask API", "cd web && npm install && npm run build && cd ..   then   python app.py   "
     f"and open http://localhost:8000 (Figure M.1). Choose a date, starting hour and horizon and click Run forecast. "
     "The chart shows history, forecast, actual values, the 95% interval and peak-hour alerts."),
    ("Call the API", "curl http://localhost:8000/api/sample-request > req.json   then   curl -X POST "
     "-H \"Content-Type: application/json\" -d @req.json http://localhost:8000/api/predict"),
    ("Run the full stack with monitoring", "docker compose up --build   starts the API and web app (8000), "
     "Streamlit (8501), MLflow (5000), Prometheus (9090) and Grafana (3000). Open Grafana and select the "
     "dashboard “VoltCast – Model Monitoring” (Figures M.4–M.5)."),
    ("Deploy on Kubernetes (optional)", "docker build -t voltcast:latest .   then load the image into the "
     "cluster (e.g. minikube image load voltcast:latest) and run   kubectl apply -f k8s/"),
    ("Retrain with new settings", "Edit configs/params.yaml (e.g. a hyper-parameter grid or split dates) "
     "and run dvc repro. Only affected stages re-run; a new MLflow run and model version are created."),
]
for i, (t, d) in enumerate(steps, 1):
    p = para(after=2)
    add_rich(p, f"**Step {i}: {t}.** ")
    p2 = para(d, 11, align=WD_ALIGN_PARAGRAPH.LEFT, spacing=1.15, after=8)
    p2.paragraph_format.left_indent = Cm(0.8)
figure(SHOT / "web_forecast.png", "Web application – live forecast", 15.5)
figure(SHOT / "web_models.png", "Web application – model performance, over-fitting and tuning", 15.5)
figure(SHOT / "web_peaks.png", "Web application – peak-hour classifier metrics", 15.5)
figure(SHOT / "grafana.png", "Grafana monitoring dashboard", 15.5)
figure(SHOT / "prometheus_targets.png", "Prometheus targets page", 14)
figure(SHOT / "dash_1.png", "Streamlit dashboard (Docker / local) – forecast tab", 15.5)

# ---------- page numbers ----------
for s in doc.sections:
    p = s.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    field(p, "PAGE", 11, "1")

doc.save(OUT)
print("Saved", OUT)
