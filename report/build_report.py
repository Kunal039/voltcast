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
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]
        c.text = ""
        set_font(c.paragraphs[0].add_run(str(h)), 10.5, bold=True)
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
    ["OpenAPI Specification 3.1", "OpenAPI Initiative", "Language-agnostic description of REST APIs; the "
     "prediction service publishes an OpenAPI 3.1 schema.", "Ch. 4.1"],
], "Standards referred to in the project", [3.2, 2.8, 7.0, 2.2])

# ---------- abstract etc ----------
unnumbered("ABSTRACT")
para(f"Residential buildings account for a large share of national electricity demand, yet most households "
     "receive only a monthly bill and have no forward-looking view of their consumption. Utilities, in turn, "
     "need short-term load forecasts at the household level to plan demand-response programmes as smart "
     "meters are rolled out. This project develops a **Household Electricity Consumption Prediction System** "
     "that forecasts a household's hourly active power for the next 1–48 hours and is engineered "
     "end to end following Machine Learning Operations (MLOps) practice.")
para(f"The system uses the public UCI “Individual Household Electric Power Consumption” dataset: "
     f"{stats['raw_rows']:,} one-minute readings recorded over almost four years in a house near Paris. "
     f"A modular, configuration-driven pipeline ingests and validates the data, cleans {stats['raw_missing_pct']}% "
     f"missing values, resamples to {stats['hourly_rows']:,} hourly records and engineers 31 leakage-free lag, "
     "rolling-window, calendar and sub-meter features. Five models (persistence and seasonal-naive baselines, "
     "Ridge regression, Random Forest and Histogram Gradient Boosting) are trained on a strict chronological "
     "split and tracked in MLflow. The best model, Histogram Gradient Boosting, reaches a test MAE of "
     f"{T['mae']:.3f} kW, RMSE of {T['rmse']:.3f} kW and R² of {T['r2']:.3f}, improving MAE by "
     f"{MAE_GAIN:.1f}% and RMSE by {RMSE_GAIN:.1f}% over the persistence baseline.")
para("The model is registered in the MLflow model registry, served through a containerised FastAPI REST "
     "service with input validation and prediction intervals, and presented in an interactive Streamlit "
     "dashboard. A GitHub Actions workflow re-runs the pipeline, enforces a quality gate, executes the "
     "automated test suite and builds the Docker image on every change and on a weekly schedule. A monitoring "
     "stage computes the Population Stability Index and rolling error; it correctly detected seasonal drift "
     "between the winter-heavy training data and the summer test period and raised a retraining "
     "recommendation, demonstrating the closed MLOps loop.")
para("Keywords: load forecasting, smart meter, gradient boosting, MLOps, MLflow, FastAPI, data drift.", italic=True)

unnumbered("GRAPHICAL ABSTRACT")
figure(FIG / "fig_architecture.png", "Graphical abstract: end-to-end MLOps architecture of the system", 16)

unnumbered("ABBREVIATIONS")
table(["Abbreviation", "Full form"], [
    ["AI", "Artificial Intelligence"], ["API", "Application Programming Interface"],
    ["ARIMA", "Auto-Regressive Integrated Moving Average"], ["CI/CD", "Continuous Integration / Continuous Delivery"],
    ["CT", "Continuous Training"], ["CNN", "Convolutional Neural Network"], ["DVC", "Data Version Control"],
    ["EDA", "Exploratory Data Analysis"], ["HGB", "Histogram-based Gradient Boosting"],
    ["IEA", "International Energy Agency"], ["kW / kWh", "Kilowatt / Kilowatt-hour"],
    ["LSTM", "Long Short-Term Memory"], ["MAE", "Mean Absolute Error"],
    ["MAPE", "Mean Absolute Percentage Error"], ["ML", "Machine Learning"],
    ["MLOps", "Machine Learning Operations"], ["NILM", "Non-Intrusive Load Monitoring"],
    ["PSI", "Population Stability Index"], ["REST", "Representational State Transfer"],
    ["RF", "Random Forest"], ["RMSE", "Root Mean Squared Error"], ["R²", "Coefficient of determination"],
    ["SVR", "Support Vector Regression"], ["UCI", "University of California, Irvine (ML Repository)"],
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
    ["σ_r", "Standard deviation of test residuals (used for prediction intervals)"],
    ["h", "Forecast horizon in hours"],
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
     "most failures come from data dependencies, configuration and lack of monitoring [4]. A consumption "
     "predictor that silently degrades when seasons or occupants change is a contemporary, documented issue.")

heading("1.2. Identification of Problem", 2)
para("Households and utilities lack a reliable, continuously maintained way to anticipate a single "
     "home's electricity demand over the next hours. Household load is highly volatile, depends on human "
     "routines, weekday, season and appliance usage, and drifts over time. Meter readings contain gaps and "
     "noise. Models built once and left unattended become stale, and there is no feedback that tells the "
     "operator when this happens.")

heading("1.3. Identification of Tasks", 2)
para("The work was divided into the following tasks, which also define the structure of this report:")
table(["#", "Task", "Output", "Chapter"], [
    ["T1", "Survey of the problem, existing forecasting methods and MLOps literature", "Problem definition, objectives", "1, 2"],
    ["T2", "Selection of a public, realistic smart-meter dataset and EDA", "Data understanding, figures", "3.1, 4.1"],
    ["T3", "Design of features, constraints and alternative system designs", "Selected design", "3"],
    ["T4", "Data pipeline: ingestion, validation, cleaning, resampling, feature engineering", "Versioned datasets", "3.6"],
    ["T5", "Model training, experiment tracking, selection and registry", "Registered model", "4.1"],
    ["T6", "Deployment: REST API, Docker, dashboard", "Running services", "4.1"],
    ["T7", "Automation and quality: CI/CT workflow, automated tests, monitoring", "CI pipeline, drift report", "4.1"],
    ["T8", "Validation, documentation, report and user manual", "This report", "4, 5, Manual"],
], "Identified tasks", [1.0, 7.3, 4.2, 2.0])

heading("1.4. Timeline", 2)
para("The project was executed over twelve weeks. Figure 1.1 shows the Gantt chart; tasks overlap where the "
     "pipeline allowed parallel work (for example, the dashboard was built while CI was configured).")
figure(FIG / "fig_gantt.png", "Gantt chart of the project timeline")

heading("1.5. Organization of the Report", 2)
bullets([
    "**Chapter 1** introduces the need, the problem, the tasks and the timeline.",
    "**Chapter 2** reviews the history of load forecasting, existing statistical, machine-learning and deep-"
    "learning solutions, compares them, and defines the problem and objectives.",
    "**Chapter 3** evaluates features and constraints, compares two alternative designs, selects one and "
    "presents the implementation methodology.",
    "**Chapter 4** presents the implementation and validates the results with metrics, plots, tests and "
    "monitoring outputs.",
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
    ["2023", "MLOps formally defined and its architecture surveyed in IEEE Access [5]."],
], "Timeline of the reported problem", [3.0, 12.0])

heading("2.2. Existing solutions", 2)
para("**Statistical time-series models.** ARIMA and exponential smoothing [6], [16] model linear "
     "autocorrelation and seasonality. They are interpretable and cheap but assume stationarity and struggle "
     "with the abrupt, appliance-driven spikes of a single household.")
para("**Classical machine learning.** Fan et al. combined data-mining techniques to predict next-day building "
     "consumption and peak demand [17]. Tree ensembles such as Random Forest [8] and gradient boosting [9], "
     "[18] learn non-linear interactions from engineered lag and calendar features and are robust to scale and "
     "outliers.")
para("**Deep learning.** Marino et al. applied sequence-to-sequence LSTMs to the same UCI dataset [11]; Kim and "
     "Cho combined CNN and LSTM layers and reported lower error than linear and LSTM baselines [12]. These "
     "models capture long dependencies but require more data, GPU training and careful tuning.")
para("**MLOps platforms.** Kreuzberger et al. describe the components of a production ML system: versioned "
     "data and code, experiment tracking, model registry, automated pipelines, serving and monitoring [5]. "
     "MLflow [14] and drift-detection methods [19] are widely used building blocks.")

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
     "work stops at an offline accuracy number, with no serving, automation or monitoring. This project "
     "therefore combines a strong, efficient tree-based model with a complete MLOps pipeline, and evaluates "
     "it honestly against naive baselines on a chronological hold-out.")

heading("2.5. Problem Definition", 2)
para("**What is to be done:** build a system that, given the most recent week of hourly smart-meter readings "
     "for a household, forecasts the household's mean active power (kW) for each of the next 1–48 hours "
     "with a prediction interval, and that can be retrained, tested, deployed and monitored automatically.")
para("**How it is to be done:** using the UCI household dataset, a reproducible configuration-driven "
     "pipeline, supervised regression models compared under MLflow tracking, a FastAPI service packaged with "
     "Docker, a Streamlit dashboard, a GitHub Actions CI/CT workflow and PSI-based drift monitoring.")
para("**What is not to be done:** appliance-level disaggregation (NILM), control of household devices, "
     "billing integration, or collection of new personal data. Forecasts are for a single household and are "
     "not intended for grid-level dispatch.")

heading("2.6. Goals/Objectives", 2)
bullets([
    f"Clean and resample the 2.07 M-row dataset into a gap-free hourly series with an auditable data report.",
    "Engineer at least 25 features that use only past information (verified by an automated leakage test).",
    "Train and compare at least four models under experiment tracking on a strict chronological split.",
    "Beat the persistence baseline on the held-out test period by at least 15% in MAE.",
    "Serve forecasts through a documented REST API with input validation and a response time under 1 s.",
    "Automate pipeline, quality gate, tests and container build in a CI/CT workflow.",
    "Detect data drift (PSI > 0.2) and performance degradation, and recommend retraining automatically.",
], numbered=True)

# ================= CHAPTER 3 =================
chapter(3, "DESIGN FLOW/PROCESS")
heading("3.1. Evaluation & Selection of Specifications/Features", 2)
para(f"**Dataset.** The UCI dataset [10] contains {stats['raw_rows']:,} one-minute measurements from "
     f"{stats['start'][:10]} to {stats['end'][:10]} for one house in Sceaux, France: global active and reactive "
     "power, voltage, current intensity and three sub-meters (kitchen; laundry room; electric water heater and "
     "air conditioner). It was chosen because it is public, long enough to cover four winters, contains real "
     "gaps, and mirrors the load-profile channels exposed by IEC 62056 smart meters. Figures 3.1–3.3 "
     "summarise the exploratory analysis.")
figure(FIG / "fig_eda_timeseries.png", "Daily mean active power with 30-day rolling mean (strong yearly seasonality)")
figure(FIG / "fig_eda_heatmap.png", "Average consumption by weekday and hour (morning and evening peaks)")
figure(FIG / "fig_eda_month_submeter.png", "Monthly pattern and share of energy per sub-meter")
para("The EDA shows a strong winter peak (consumption roughly doubles from August to December), two daily "
     "peaks around 07:00 and 20:00–21:00, higher daytime use at weekends, and an August holiday dip. "
     "These findings drove the feature specification in Table 3.1.")
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
     "(reliability, maintainability) drive testing and modularity; API documented to OpenAPI 3.1."],
    ["Economic / Cost", "Must run on a student laptop and free CI runners: no GPU, training under one minute, "
     "only open-source tools (scikit-learn, MLflow, FastAPI, Streamlit, Docker)."],
    ["Environmental", "Low-compute models reduce the energy footprint of retraining; the purpose of the system "
     "is itself to support energy saving (ISO 50001 spirit)."],
    ["Health & Safety", "Advisory system only: it never switches appliances, so incorrect forecasts cannot cause "
     "physical harm."],
    ["Professional / Ethical", "Honest evaluation against naive baselines on unseen future data; no "
     "data leakage; limitations are disclosed."],
    ["Social & Privacy", "Fine-grained meter data can reveal occupancy and routines. The system uses a "
     "public anonymised dataset, stores no personal identifiers and processes data on-premise."],
    ["Manufacturability / Deployability", "Must be reproducible on any machine: pinned configuration, "
     "Docker image, single-command pipeline."],
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
    "**Modified:** missing data handling. Gaps up to 6 h are time-interpolated; longer outages (e.g. multi-day "
    "gaps in 2010) are filled with the same hour of the previous week to preserve the daily profile.",
    "**Added:** prediction intervals (±1.96σ_r√h) so users can see forecast uncertainty.",
    "**Added:** automated leakage test and a baseline quality gate to satisfy the professional/ethical constraint.",
])

heading("3.4. Design Flow", 2)
para("Two alternative designs were developed to meet the specification (Figure 3.4).")
figure(FIG / "fig_design_alternatives.png", "Alternative designs considered")
para("**Design A – Notebook-centric deep learning.** All steps (cleaning, features, an LSTM model) live "
     "in a single Jupyter notebook. The trained network is pickled manually and predictions are produced "
     "with an ad-hoc script. This is the typical approach in the literature [11], [12].")
para("**Design B – Modular MLOps pipeline.** Each stage is an independent, tested Python module driven by "
     "one configuration file (params.yaml) and declared as a DVC pipeline. Several models are compared under "
     "MLflow tracking; the best is registered and served by a Dockerised FastAPI service and a Streamlit "
     "dashboard; GitHub Actions automates the pipeline and a monitoring stage closes the loop.")

heading("3.5. Design selection", 2)
table(["Criterion", "Design A", "Design B"], [
    ["Expected accuracy", "High (with tuning)", "High (gradient boosting)"],
    ["Training cost / hardware", "GPU preferred, minutes–hours", "CPU, < 10 s per model"],
    ["Reproducibility", "Low: hidden notebook state", "High: config + pipeline + Docker"],
    ["Experiment tracking", "Manual", "Automatic (MLflow)"],
    ["Deployment", "Manual script", "REST API + container + dashboard"],
    ["Testing / CI", "None", "8 automated tests, CI quality gate"],
    ["Monitoring & retraining", "None", "PSI drift + MAE checks, weekly CT"],
    ["Explainability", "Low", "Permutation importance"],
    ["Fit with constraints", "Violates cost & reproducibility", "Satisfies all"],
], "Comparison of alternative designs", [5.0, 5.0, 5.0])
para("**Design B is selected.** It meets every constraint in Section 3.2, its accuracy is competitive, and "
     "it delivers the operational capabilities (automation, deployment, monitoring) that are the core "
     "learning outcomes of an MLOps project and that the literature identifies as missing.")

heading("3.6. Implementation plan/methodology", 2)
para("Figure 3.5 shows the algorithm executed by the pipeline and Figure 3.6 the detailed block diagram.")
figure(FIG / "fig_flowchart.png", "Flowchart of the training and deployment algorithm", 10)
figure(FIG / "fig_architecture.png", "Detailed block diagram of the MLOps architecture")
para("**Data split.** To respect time ordering, data is split chronologically: training up to 31 Dec 2009 "
     f"(26,503 h), validation Jan–May 2010 (3,624 h) and test {meta['test_period'][0][:10]} to "
     f"{meta['test_period'][1][:10]} (4,294 h). Models are selected on validation, refit on "
     "train+validation, and scored once on test.")
para("**Models.** Persistence (ŷₜ = yₜ₋₁) and seasonal-naive (ŷₜ = "
     "yₜ₋₂₄) baselines; Ridge regression (α = 1, standardised inputs); Random Forest "
     "(200 trees, depth 16) [8]; Histogram Gradient Boosting (500 iterations, η = 0.05, 31 leaves) [9], [18].")
para("**Metrics.** MAE = (1/n)Σ|yₜ − ŷₜ|, RMSE = √((1/n)Σ(yₜ − "
     "ŷₜ)²), R² and MAPE. RMSE is used for model selection because peak errors matter most "
     "for demand response.")
para("**Monitoring.** PSI = Σ(qᵢ − pᵢ)·ln(qᵢ/pᵢ) over 10 quantile bins of the "
     "training distribution; PSI > 0.2 signals significant drift. Retraining is recommended when any key "
     "feature drifts or recent 30-day MAE degrades more than 20%.")
table(["Module", "Responsibility"], [
    ["src/data_ingestion.py", "Download dataset, verify schema"],
    ["src/preprocess.py", "Parse, clean, derive unmetered energy, resample hourly, fill gaps"],
    ["src/features.py", "Build 31 leakage-free features"],
    ["src/train.py", "Train/compare 5 models, MLflow logging, model registry, metadata"],
    ["src/monitor.py", "PSI drift and performance degradation, retraining decision"],
    ["src/evaluate.py", "EDA and diagnostic figures, permutation importance"],
    ["src/predict.py", "Recursive multi-step forecasting with intervals"],
    ["serving/main.py, app.py", "FastAPI service: /predict, /forecast, /health, /model-info, /metrics"],
    ["public/index.html, src/export_web.py", "Web dashboard hosted on Vercel and its data export"],
    ["dashboard/app.py", "Streamlit dashboard (5 tabs)"],
    ["tests/test_pipeline.py", "Unit, leakage, API and quality-gate tests"],
    [".github/workflows/ci.yml, Dockerfile, dvc.yaml", "Automation, packaging, reproducibility"],
], "Software modules of the implementation", [6.0, 9.0])

# ================= CHAPTER 4 =================
chapter(4, "RESULTS ANALYSIS AND VALIDATION")
heading("4.1. Implementation of solution", 2)
para("**Modern tools used.** Analysis: pandas, NumPy, Matplotlib, Seaborn, Plotly. Design drawings: "
     "Matplotlib-generated architecture, flow and Gantt diagrams. Modelling: scikit-learn. Experiment "
     "management: MLflow (SQLite backend and model registry). Project management and communication: Git, "
     "GitHub Actions, DVC pipeline definition. Testing and validation: pytest, FastAPI TestClient, PSI "
     "drift analysis. Report preparation: python-docx generated directly from pipeline outputs, so every "
     "number in this chapter is reproducible.")
para(f"**Data preparation results.** {stats['raw_missing_pct']}% of minute readings were missing ('?'). After "
     f"cleaning and hourly resampling the series has {stats['hourly_rows']:,} gap-free records with mean "
     "1.09 kW (median 0.80 kW, max 6.56 kW). Feature building drops the first 168 h (needed for the weekly "
     "lag), leaving 34,421 modelling rows. Figure 4.1 shows that current intensity is almost perfectly "
     "correlated with active power, while voltage is weakly negatively correlated.")
figure(FIG / "fig_eda_correlation.png", "Correlation matrix of the measured variables", 12)

para("**Model comparison.** Table 4.1 and Figure 4.2 show validation results tracked in MLflow.")
rows = [[n.replace("_", " ").title(), f"{m['mae']:.4f}", f"{m['rmse']:.4f}", f"{m['r2']:.4f}",
         f"{m['mape']:.1f}", f"{m['fit_seconds']:.2f}"] for n, m in V.items()]
best_i = list(V).index(meta["model_name"])
rows[best_i] = [f"**{c}**" for c in rows[best_i]]
table(["Model", "MAE (kW)", "RMSE (kW)", "R²", "MAPE (%)", "Fit time (s)"], rows,
      "Validation performance of candidate models (Jan–May 2010)", [4.6, 2.0, 2.1, 1.8, 2.0, 2.2])
figure(FIG / "fig_model_comparison.png", "Validation MAE and R² of candidate models (best in orange)")
para(f"Both ML ensembles clearly beat the baselines and the linear model. Histogram Gradient Boosting has the "
     f"lowest validation RMSE ({V['hist_gradient_boosting']['rmse']:.4f} kW) and trains six times faster than "
     "Random Forest, so it was selected. The seasonal-naive model has negative R²: yesterday's same hour "
     "is a worse predictor than the mean, confirming that household load is driven by irregular routines.")

para("**Test-set results.** The selected model was refit on train+validation and evaluated once on the "
     "untouched test period (Table 4.2).")
table(["Model", "MAE (kW)", "RMSE (kW)", "R²", "MAPE (%)"], [
    ["Persistence baseline", f"{B['mae']:.4f}", f"{B['rmse']:.4f}", f"{B['r2']:.4f}", f"{B['mape']:.1f}"],
    ["**Histogram Gradient Boosting**", f"**{T['mae']:.4f}**", f"**{T['rmse']:.4f}**", f"**{T['r2']:.4f}**",
     f"**{T['mape']:.1f}**"],
    ["Improvement", f"{MAE_GAIN:.1f}%", f"{RMSE_GAIN:.1f}%", f"+{T['r2'] - B['r2']:.3f}", "–"],
], "Held-out test performance (Jun–Nov 2010)", [5.4, 2.4, 2.4, 2.4, 2.4])
figure(FIG / "fig_actual_vs_pred.png", "Next-hour forecast vs actual consumption, last 14 days of the test set")
figure(FIG / "fig_residuals.png", "Predicted vs actual scatter and residual distribution")
para("The forecasts follow the daily cycle and the timing of evening peaks closely (Figure 4.3). Residuals are "
     "centred near zero but right-skewed (Figure 4.4): the largest errors are sudden short spikes, such as an "
     "oven or water heater switching on, which are inherently unpredictable from past meter data alone.")
figure(FIG / "fig_error_by_hour.png", "Mean absolute error by hour of day on the test set")
para("Error is lowest at night (stable base load) and highest during the 07:00 and 19:00–22:00 "
     "activity peaks (Figure 4.5), where occupant behaviour dominates.")

para("**Explainability.** Permutation importance (Figure 4.6) shows which inputs the model relies on.")
figure(FIG / "fig_feature_importance.png", "Top 15 features by permutation importance", 13)
top = ", ".join(f"{k} ({v:.3f})" for k, v in imp.head(4).items())
para(f"The most important inputs are {top}. The previous hour's current intensity and power carry the "
     "household's current state, the water-heater/AC sub-meter signals heavy thermal loads, and the hour "
     "encodings capture daily routines, which is physically consistent.")

para("**Deployment validation.** The FastAPI service and web dashboard are deployed on Vercel at https://household-power-forecast.vercel.app, "
     "auto-deployed from the GitHub repository. The service was also run locally. A forecast request "
     "with one week of history returned a 24-hour forecast with intervals in well under one second. "
     "Interactive OpenAPI documentation is served at /docs (see User Manual).")
para("**Automated testing.** The pytest suite (Table 4.3) passes in about 3 seconds and runs in CI on every push.")
table(["Test", "What it validates", "Result"], [
    ["test_resample_fills_gaps", "Missing hours are filled; output is gap-free", "Pass"],
    ["test_features_have_no_leakage", "lag_1 and rolling means use only past values", "Pass"],
    ["test_metrics_perfect_prediction", "Metric implementation correctness", "Pass"],
    ["test_psi_detects_shift", "PSI < 0.1 for same distribution, > 0.2 for shifted", "Pass"],
    ["test_health_endpoint", "Service liveness and model loading", "Pass"],
    ["test_predict_endpoint_returns_horizon", "Horizon length, interval ordering", "Pass"],
    ["test_predict_rejects_short_history", "Input validation returns HTTP 422", "Pass"],
    ["test_model_beats_baseline", "Quality gate: model MAE < baseline MAE", "Pass"],
], "Automated test suite results", [5.6, 7.4, 2.0])

para("**Monitoring validation.** Figure 4.7 shows the PSI between the training and test distributions.")
figure(FIG / "fig_drift_psi.png", "Population Stability Index of monitored features (train vs test)", 13)
drifted = ", ".join(mon["drifted_features"])
para(f"Significant drift was detected in {drifted} (PSI {mon['psi']['Sub_metering_3_lag1']:.2f} for the "
     "water-heater/AC sub-meter). This is genuine seasonal drift: the training data covers three winters, "
     "while the test period is June–November, when heating load is low. The 30-day rolling MAE "
     f"({mon['recent_30d_mae']:.3f} kW) was {mon['mae_degradation_pct']:.1f}% above the reference test MAE as "
     "consumption climbed again in November. The monitor therefore set retraining_recommended = true, which "
     "is exactly the behaviour a production MLOps loop requires.")
table(["Objective (Section 2.6)", "Target", "Achieved"], [
    ["Clean hourly series", "Gap-free", f"{stats['hourly_rows']:,} hourly rows, 0 gaps"],
    ["Feature engineering", "≥ 25 leakage-free", "31 features, leakage test passes"],
    ["Models compared", "≥ 4", "5 models tracked in MLflow"],
    ["Improvement over baseline", "≥ 15% MAE", f"{MAE_GAIN:.1f}% MAE, {RMSE_GAIN:.1f}% RMSE"],
    ["REST API latency", "< 1 s", "Achieved (24-h forecast)"],
    ["CI/CT automation", "Pipeline + tests + build", "GitHub Actions workflow"],
    ["Drift detection", "PSI > 0.2 flagged", f"{len(mon['drifted_features'])} features flagged, retrain advised"],
], "Achievement of objectives", [5.5, 4.0, 5.5])

# ================= CHAPTER 5 =================
chapter(5, "CONCLUSION AND FUTURE WORK")
heading("5.1. Conclusion", 2)
para("This project delivered a complete, working Household Electricity Consumption Prediction System built "
     "on MLOps principles. Starting from 2.07 million raw smart-meter readings, the automated pipeline "
     "produces a clean hourly series, 31 leakage-free features and five tracked models. The selected "
     f"Histogram Gradient Boosting model forecasts next-hour consumption with a test MAE of {T['mae']:.3f} kW "
     f"and R² of {T['r2']:.3f}, a {MAE_GAIN:.1f}% MAE reduction over the persistence baseline. The model "
     "is registered in MLflow, served through a Dockerised FastAPI service, visualised in a Streamlit "
     "dashboard, protected by automated tests and a CI quality gate, and watched by a drift monitor.")
para("**Expected vs obtained results.** The objective of a ≥ 15% improvement over the baseline was met. "
     f"However, the absolute R² (~{T['r2']:.2f}) is lower than the >0.9 values sometimes reported in the "
     "literature. The reasons are: (i) we forecast a single household at hourly resolution, where "
     "behaviour-driven spikes are inherently unpredictable; (ii) we use a strict chronological split and "
     "refuse features computed from the hour being predicted, whereas some studies shuffle data or use "
     "same-hour intensity, which inflates scores; and (iii) the test period has a different seasonal "
     "distribution from training, which the monitor correctly flagged. MAPE (~39%) is inflated by very low "
     "night-time loads, where small absolute errors become large percentages; MAE and RMSE are the more "
     "meaningful measures here.")
heading("5.2. Future work", 2)
bullets([
    "**Weather and occupancy inputs:** add outdoor temperature (heating/cooling degree hours) and holiday "
    "calendars from an API; expected to reduce winter peak error substantially.",
    "**Deep and probabilistic models:** evaluate LSTM/Temporal Fusion Transformer and quantile gradient "
    "boosting for calibrated prediction intervals [12], [15].",
    "**Multi-household, Indian data:** train a global model across many homes using data from IS 16444 "
    "smart meters, with federated learning to protect privacy.",
    "**Automated retraining:** connect the monitor's retraining flag to a workflow trigger and promote a new "
    "model only if it beats the current one (champion/challenger) in the registry.",
    "**Cloud deployment:** deploy the container to a managed service with autoscaling, and add Prometheus/"
    "Grafana for service metrics.",
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
    ["Measurable objectives defined", "✔"],
    ["Design constraints (standards, cost, ethics, privacy, safety) considered", "✔"],
    ["At least two alternative designs compared and one selected with reasons", "✔"],
    ["Flowchart / block diagram of implementation", "✔"],
    ["Public dataset, reproducible pipeline (params.yaml, dvc.yaml)", "✔"],
    ["Chronological train/validation/test split, baseline comparison", "✔"],
    ["Experiment tracking and model registry (MLflow)", "✔"],
    ["Model serving (FastAPI) and containerisation (Docker)", "✔"],
    ["Interactive dashboard (Streamlit)", "✔"],
    ["Automated tests and CI/CT workflow", "✔"],
    ["Data drift and performance monitoring", "✔"],
    ["User manual with step-by-step instructions", "✔"],
], "Design checklist", [12.5, 2.5])

# ================= USER MANUAL =================
unnumbered("USER MANUAL", "M")
para("**Hosted version (no installation):** open https://household-power-forecast.vercel.app for the web dashboard and live forecasts, "
     "and https://household-power-forecast.vercel.app/api/docs for the API. Source code: https://github.com/Kunal039/household-electricity-consumption-prediction.")
para("**Prerequisites:** Python 3.10 or newer, about 1 GB free disk space, internet access for the first "
     "run (dataset download, ~20 MB). Docker Desktop is optional.")
steps = [
    ("Get the project", "git clone https://github.com/Kunal039/household-electricity-consumption-prediction.git   then open a terminal inside the folder."),
    ("Install dependencies", "pip install -r requirements-dev.txt"),
    ("Run the complete pipeline", "python run_pipeline.py   (downloads data, preprocesses, builds features, "
     "trains and registers the model, runs monitoring and creates all figures; about 1 minute)"),
    ("Run the tests", "pytest -q   (8 tests should pass)"),
    ("Inspect experiments in MLflow", "mlflow ui --backend-store-uri sqlite:///mlflow.db   then open "
     "http://localhost:5000 to compare runs and see the registered model household-power-forecaster."),
    ("Start the prediction API", "uvicorn app:app --port 8000   then open http://localhost:8000/api/docs "
     "(Figure M.1). Use GET /api/sample-request to obtain a ready-made input, then paste it into POST /api/predict "
     "and click Execute."),
    ("Start the dashboard", "streamlit run dashboard/app.py   then open http://localhost:8501 (Figures M.2–M.6). "
     "In the Forecast tab choose a date, start hour and horizon; the chart shows history, forecast, actual "
     "values and the 95% interval, with predicted energy, cost and peak hour. Other tabs show the data "
     "explorer, model performance, monitoring and pipeline diagram."),
    ("Run everything with Docker (optional)", "docker compose up --build   starts the API (port 8000), "
     "dashboard (8501) and MLflow UI (5000)."),
    ("Retrain with new settings", "Edit configs/params.yaml (e.g. model hyper-parameters or split dates) "
     "and re-run python run_pipeline.py (or dvc repro). A new MLflow run and model version are created."),
]
for i, (t, d) in enumerate(steps, 1):
    p = para(after=2)
    add_rich(p, f"**Step {i}: {t}.** ")
    p2 = para(d, 11, align=WD_ALIGN_PARAGRAPH.LEFT, spacing=1.15, after=8)
    p2.paragraph_format.left_indent = Cm(0.8)
figure(SHOT / "api_docs.png", "Interactive API documentation (Swagger UI) at /api/docs", 14)
for i, name in enumerate(["Forecast", "Data explorer", "Model performance", "Monitoring", "MLOps pipeline"], 1):
    figure(SHOT / f"dash_{i}.png", f"Streamlit dashboard – {name} tab", 15.5)

# ---------- page numbers ----------
for s in doc.sections:
    p = s.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    field(p, "PAGE", 11, "1")

doc.save(OUT)
print("Saved", OUT)
