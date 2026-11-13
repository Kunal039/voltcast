"""Additional report diagrams: DFDs, use cases, sequence, deployment, CI/CD, DVC DAG and stage flowcharts."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon

from make_diagrams import BLUE, GREEN, GREY, ORANGE, OUT, arrow, box

YELLOW = "#fff3c4"


def canvas(w, h, xmax=None, ymax=None):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, xmax or w * 1.6)
    ax.set_ylim(0, ymax or h * 1.6)
    ax.axis("off")
    return fig, ax


def save(name):
    plt.savefig(OUT / name, bbox_inches="tight")
    plt.close()


def diamond(ax, cx, cy, w, h, text, color=YELLOW, size=7.5):
    ax.add_patch(Polygon([(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)],
                         closed=True, fc=color, ec="#333", lw=0.9))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=size)


def oval(ax, cx, cy, w, h, text, color=GREY, size=8):
    ax.add_patch(Ellipse((cx, cy), w, h, fc=color, ec="#333", lw=0.9))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=size)


def store(ax, x, y, w, h, text):
    """Data store in DFD notation (open-ended rectangle)."""
    ax.plot([x, x + w], [y + h, y + h], color="#333", lw=0.9)
    ax.plot([x, x + w], [y, y], color="#333", lw=0.9)
    ax.plot([x, x], [y, y + h], color="#333", lw=0.9)
    ax.text(x + 0.15, y + h / 2, text, va="center", fontsize=7.5)


def flowchart(name, steps, title, decisions=None, width=6.2, exits=None, down_labels=None):
    """Vertical flowchart. steps: list of (text, kind) with kind in box/decision/terminal."""
    n = len(steps)
    fig, ax = plt.subplots(figsize=(width, 0.75 * n + 0.6))
    ax.set_xlim(0, 12.2 if exits else 10 + 0.7 * max(len(decisions or []) - 1, 0))
    ax.set_ylim(0, n * 1.3 + 0.4)
    ax.axis("off")
    ys = []
    for i, (text, kind) in enumerate(steps):
        y = n * 1.3 - i * 1.3 - 0.4
        ys.append(y)
        if kind == "terminal":
            oval(ax, 5, y, 4.6, 0.85, text, size=9)
        elif kind == "decision":
            diamond(ax, 5, y, 6.6, 1.1, text, size=8.5)
        else:
            box(ax, 1.4, y - 0.4, 7.2, 0.8, text, ORANGE if kind == "model" else BLUE, size=8.5)
        if i:
            arrow(ax, (5, ys[i - 1] - (0.53 if steps[i - 1][1] == "decision" else 0.4)),
                  (5, y + (0.53 if kind == "decision" else 0.4)))
    for k, (src, dst, label) in enumerate(decisions or []):
        x = 9.3 + 0.7 * k
        arrow(ax, (8.2, ys[src]), (x, ys[src]), style="-")
        ax.plot([x, x], [ys[src], ys[dst]], color="#333", lw=0.9)
        arrow(ax, (x, ys[dst]), (8.4, ys[dst]))
        ax.text(x + 0.12, (ys[src] + ys[dst]) / 2, label, fontsize=8.5, rotation=90, va="center", style="italic")
    for idx, label, text in exits or []:
        arrow(ax, (8.3, ys[idx]), (9.4, ys[idx]))
        ax.text(8.45, ys[idx] + 0.12, label, fontsize=8.5, style="italic")
        oval(ax, 10.8, ys[idx], 2.9, 1.0, text, "#f6d6d3", 7.5)
    down = {i: "yes" for i, (_, k) in enumerate(steps) if k == "decision"}
    down.update(down_labels or {})
    for idx, label in down.items():
        ax.text(5.15, ys[idx] - 0.75, label, fontsize=8.5, style="italic")
    ax.set_title(title, fontsize=11, weight="bold")
    save(name)


def dfd_level0():
    fig, ax = canvas(9, 4.2, 15, 7)
    oval(ax, 7.5, 3.5, 4.2, 2.0, "0\nVoltCast\nprediction system", BLUE, 9)
    box(ax, 0.2, 5.2, 2.8, 1.1, "UCI repository\n(data source)", GREY)
    box(ax, 0.2, 0.7, 2.8, 1.1, "Household /\nsmart meter", GREY)
    box(ax, 12.0, 5.2, 2.8, 1.1, "User\n(web browser)", GREY)
    box(ax, 12.0, 0.7, 2.8, 1.1, "Operator /\nML engineer", GREY)
    arrow(ax, (3.0, 5.75), (5.6, 4.2), "raw readings (zip)")
    arrow(ax, (3.0, 1.25), (5.6, 2.8), "hourly history (JSON)")
    arrow(ax, (12.0, 6.1), (9.3, 4.4))
    ax.text(10.9, 5.75, "forecast request", fontsize=7, style="italic", rotation=31)
    arrow(ax, (9.7, 3.8), (12.0, 5.25))
    ax.text(10.95, 4.25, "forecast, alerts", fontsize=7, style="italic", rotation=31)
    arrow(ax, (9.6, 2.9), (12.0, 1.5))
    ax.text(10.5, 2.6, "metrics, drift, logs", fontsize=7, style="italic", rotation=-30)
    arrow(ax, (12.0, 0.85), (9.2, 2.6))
    ax.text(10.0, 1.25, "parameters", fontsize=7, style="italic", rotation=-32)
    ax.set_title("Data-flow diagram, level 0 (context diagram)", fontsize=10, weight="bold")
    save("dfd_level0.png")


def dfd_level1():
    fig, ax = canvas(10, 6.4, 16, 10)
    procs = [(2.0, 8.3, "1.0\nIngest &\nvalidate"), (6.0, 8.3, "2.0\nClean &\nresample"),
             (10.0, 8.3, "3.0\nEngineer\nfeatures"), (14.0, 8.3, "4.0\nTrain, tune\n& select"),
             (14.0, 4.6, "5.0\nServe\npredictions"), (8.8, 4.6, "6.0\nMonitor\n& log"),
             (3.0, 4.6, "7.0\nVisualise\n(web app)")]
    for x, y, t in procs:
        oval(ax, x, y, 3.1, 1.7, t, BLUE, 7.5)
    store(ax, 0.5, 6.0, 3.0, 0.7, "D1  dataset CSV (DVC)")
    store(ax, 4.6, 6.0, 3.0, 0.7, "D2  hourly.parquet")
    store(ax, 8.6, 6.0, 3.0, 0.7, "D3  features.parquet")
    store(ax, 12.4, 6.0, 3.4, 0.7, "D4  models + MLflow registry")
    store(ax, 6.2, 1.6, 3.6, 0.7, "D5  Prometheus TSDB / logs")
    for a, b in [((3.55, 8.3), (4.45, 8.3)), ((7.55, 8.3), (8.45, 8.3)), ((11.55, 8.3), (12.45, 8.3))]:
        arrow(ax, a, b)
    arrow(ax, (2.0, 7.45), (2.0, 6.7))
    arrow(ax, (6.0, 7.45), (6.0, 6.7))
    arrow(ax, (10.0, 7.45), (10.0, 6.7))
    arrow(ax, (14.0, 7.45), (14.0, 6.7), "model")
    arrow(ax, (14.0, 6.0), (14.0, 5.45))
    arrow(ax, (12.45, 4.6), (10.35, 4.6), "metrics")
    arrow(ax, (8.8, 3.75), (8.0, 2.3))
    arrow(ax, (13.0, 3.85), (4.1, 3.85))
    ax.text(11.6, 3.45, "forecast JSON", fontsize=7, style="italic")
    box(ax, 0.2, 0.3, 2.6, 1.0, "User", GREY)
    arrow(ax, (1.5, 1.3), (2.4, 3.8), "request")
    ax.set_title("Data-flow diagram, level 1", fontsize=10, weight="bold")
    save("dfd_level1.png")


def use_case():
    fig, ax = canvas(9, 5.6, 15, 9.5)
    ax.add_patch(FancyBboxPatch((3.6, 0.3), 7.8, 8.6, boxstyle="round,pad=0.05", fc="#fafafa", ec="#333"))
    ax.text(7.5, 8.55, "VoltCast system", ha="center", fontsize=9, weight="bold")
    left = ["View 24-h forecast", "See peak-hour alerts", "Explore consumption data", "Call REST API"]
    right = ["Run / reproduce pipeline", "Compare & tune models", "Promote model (registry)",
             "Monitor metrics & drift", "Retrain on schedule (CI)"]
    for i, t in enumerate(left):
        oval(ax, 6.0, 7.4 - i * 1.6, 3.8, 1.0, t, "#e8f0fb", 7.5)
    for i, t in enumerate(right):
        oval(ax, 9.2, 7.4 - i * 1.6 + 0.4, 3.8, 0.95, t, "#fdeedd", 7.5)
    for x, label, ys in [(1.3, "Household user", left), (13.8, "ML engineer", right)]:
        ax.add_patch(Ellipse((x, 5.2), 0.6, 0.6, fc="white", ec="#333"))
        ax.plot([x, x], [4.9, 3.9], color="#333")
        ax.plot([x - 0.5, x + 0.5], [4.6, 4.6], color="#333")
        ax.plot([x, x - 0.4], [3.9, 3.3], color="#333")
        ax.plot([x, x + 0.4], [3.9, 3.3], color="#333")
        ax.text(x, 2.9, label, ha="center", fontsize=8)
    for i in range(len(left)):
        ax.plot([1.7, 4.1], [4.5, 7.4 - i * 1.6], color="#666", lw=0.7)
    for i in range(len(right)):
        ax.plot([13.4, 11.1], [4.5, 7.8 - i * 1.6], color="#666", lw=0.7)
    box(ax, 12.3, 0.4, 2.6, 0.9, "GitHub Actions\n(scheduler)", GREY, 7)
    ax.plot([13.6, 11.1], [1.3, 1.4], color="#666", lw=0.7, ls="--")
    ax.set_title("Use-case diagram", fontsize=10, weight="bold")
    save("use_case.png")


def sequence():
    actors = ["User\n(browser)", "React app", "Flask API\n(/api/forecast)", "Predictor\n(src/predict.py)", "Models\n(joblib)", "Prometheus"]
    fig, ax = canvas(10, 6, 16, 10)
    xs = [1.2 + i * 2.7 for i in range(len(actors))]
    for x, a in zip(xs, actors):
        box(ax, x - 1.1, 8.8, 2.2, 0.9, a, BLUE, 7.5)
        ax.plot([x, x], [0.3, 8.8], color="#999", ls="--", lw=0.8)
    msgs = [(0, 1, "choose date, hour, horizon"), (1, 2, "GET /api/forecast?origin&horizon"),
            (2, 2, "validate query (422 if invalid)"), (2, 3, "forecast(history, horizon)"),
            (3, 4, "load once (lru_cache)"), (3, 3, "loop h = 1..H: features -> predict"),
            (4, 3, "kW, peak probability"), (3, 2, "DataFrame (kW, interval, alerts)"),
            (2, 2, "log JSON line, update counters"), (2, 1, "200 JSON response"),
            (1, 0, "chart + statistics"), (5, 2, "every 15 s: GET /api/metrics")]
    y = 8.2
    for a, b, t in msgs:
        if a == b:
            ax.annotate("", xy=(xs[a] + 0.05, y - 0.35), xytext=(xs[a] + 0.05, y),
                        arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3,rad=-1.2", lw=0.8))
            ax.text(xs[a] + 0.45, y - 0.15, t, fontsize=6.8, style="italic")
        else:
            arrow(ax, (xs[a], y), (xs[b], y), ls="--" if b < a else "-")
            ax.text((xs[a] + xs[b]) / 2, y + 0.1, t, ha="center", fontsize=6.8)
        y -= 0.66
    ax.set_title("Sequence diagram: serving a forecast request", fontsize=10, weight="bold")
    save("sequence_forecast.png")


def deployment():
    fig, ax = canvas(10, 5.6, 16, 9)
    ax.add_patch(FancyBboxPatch((0.2, 0.3), 9.6, 8.2, boxstyle="round,pad=0.05", fc="#f4f7fb", ec="#333"))
    ax.text(5.0, 8.15, "On-premise: Docker Compose (docker-compose.yml)", ha="center", fontsize=8.5, weight="bold")
    svcs = [(0.6, 6.0, "api : 8000\nFlask + gunicorn\nReact build", ORANGE), (3.6, 6.0, "dashboard : 8501\nStreamlit", ORANGE),
            (6.6, 6.0, "mlflow : 5000\nMLflow UI", BLUE), (0.6, 2.6, "prometheus : 9090\nscrapes /api/metrics", GREEN),
            (3.6, 2.6, "grafana : 3000\nprovisioned dashboard", GREEN), (6.6, 2.6, "volumes\nlogs/, mlflow.db,\nmlruns/", GREY)]
    for x, y, t, c in svcs:
        box(ax, x, y, 2.8, 1.6, t, c, 7.5)
    arrow(ax, (2.0, 4.2), (2.0, 6.0))
    ax.text(2.1, 5.0, "scrape /api/metrics\nevery 15 s", fontsize=7, style="italic")
    arrow(ax, (3.6, 3.4), (3.4, 3.4))
    ax.text(3.0, 3.55, "PromQL", fontsize=6.5, style="italic")
    arrow(ax, (7.6, 4.2), (7.6, 6.0))
    ax.text(7.7, 5.0, "mounted\nvolumes", fontsize=7, style="italic")
    ax.add_patch(FancyBboxPatch((10.4, 4.8), 5.3, 3.7, boxstyle="round,pad=0.05", fc="#fdf3e8", ec="#333"))
    ax.text(13.05, 8.15, "Cloud: Vercel", ha="center", fontsize=8.5, weight="bold")
    box(ax, 10.7, 6.2, 4.7, 1.4, "Edge CDN: React static files\nPython function: Flask app.py", ORANGE, 7.5)
    box(ax, 10.7, 5.0, 4.7, 0.9, "voltcast-forecast.vercel.app", GREY, 7.5)
    ax.add_patch(FancyBboxPatch((10.4, 0.3), 5.3, 4.0, boxstyle="round,pad=0.05", fc="#eef6ef", ec="#333"))
    ax.text(13.05, 3.95, "Kubernetes (k8s/deployment.yaml)", ha="center", fontsize=8.5, weight="bold")
    box(ax, 10.7, 2.3, 2.2, 1.2, "Pod 1\nvoltcast-api", ORANGE, 7)
    box(ax, 13.2, 2.3, 2.2, 1.2, "Pod 2\nvoltcast-api", ORANGE, 7)
    box(ax, 10.7, 0.6, 4.7, 1.2, "Service (NodePort 30080) + HPA 2-5 pods", GREY, 7)
    ax.set_title("Deployment diagram", fontsize=10, weight="bold")
    save("deployment.png")


def cicd():
    fig, ax = canvas(10.5, 3.6, 17, 5.8)
    steps = ["git push /\nweekly cron", "checkout +\npip install", "dvc repro\n(full pipeline)", "quality gates\nMAE, AUC",
             "pytest\n11 tests", "npm build\n(React)", "validate\ncompose/k8s", "docker\nbuild", "upload\nartifacts"]
    for i, s in enumerate(steps):
        x = 0.2 + i * 1.86
        box(ax, x, 3.0, 1.6, 1.3, s, GREEN if i in (3, 4) else BLUE, 7)
        if i:
            arrow(ax, (x - 0.26, 3.65), (x, 3.65))
    ax.text(8.5, 4.9, "GitHub Actions workflow  (.github/workflows/ci.yml)", ha="center", fontsize=8.5, weight="bold")
    box(ax, 0.2, 0.3, 3.4, 1.3, "git push to main", GREY, 7.5)
    box(ax, 4.6, 0.3, 4.6, 1.3, "Vercel: npm ci + vite build,\ninstall requirements.txt", ORANGE, 7.5)
    box(ax, 10.2, 0.3, 3.4, 1.3, "Production deploy\nvoltcast-forecast", ORANGE, 7.5)
    arrow(ax, (3.6, 0.95), (4.6, 0.95))
    arrow(ax, (9.2, 0.95), (10.2, 0.95))
    ax.text(8.5, 2.1, "Continuous deployment (Vercel Git integration)", ha="center", fontsize=8.5, weight="bold")
    ax.set_title("CI/CD pipeline", fontsize=10, weight="bold")
    save("cicd.png")


def dvc_dag():
    fig, ax = canvas(10, 5.2, 16, 8.6)
    W, H = 2.6, 0.95
    nodes = {"ingest": (0.4, 7.2), "preprocess": (4.2, 7.2), "features": (8.0, 7.2), "train": (11.8, 7.2),
             "classify": (11.8, 4.9), "monitor": (8.0, 4.9), "evaluate": (4.2, 2.6), "export_web": (8.0, 2.6),
             "analyze": (11.8, 0.3)}
    for n, (x, y) in nodes.items():
        box(ax, x, y, W, H, n, ORANGE if n in ("train", "classify") else BLUE, 8.5)
    c = lambda n, side: {"r": (nodes[n][0] + W, nodes[n][1] + H / 2), "l": (nodes[n][0], nodes[n][1] + H / 2),
                         "b": (nodes[n][0] + W / 2, nodes[n][1]), "t": (nodes[n][0] + W / 2, nodes[n][1] + H)}[side]
    edges = [("ingest", "r", "preprocess", "l"), ("preprocess", "r", "features", "l"), ("features", "r", "train", "l"),
             ("train", "b", "classify", "t"), ("features", "b", "classify", "l"), ("train", "b", "monitor", "r"),
             ("monitor", "b", "evaluate", "t"), ("classify", "b", "evaluate", "r"), ("monitor", "b", "export_web", "t"),
             ("classify", "b", "export_web", "r"), ("classify", "b", "analyze", "t"), ("export_web", "b", "analyze", "l")]
    for a, sa, b, sb in edges:
        arrow(ax, c(a, sa), c(b, sb))
    ax.text(0.4, 4.0, "Cached by DVC: dataset CSV,\nhourly.parquet, features.parquet\n\nGit-tracked (cache: false):\n"
            "models, metrics JSON, figures,\nserving CSV, web data", fontsize=7.5, va="center",
            bbox=dict(boxstyle="round", fc="#f6f6f6", ec="#bbb"))
    ax.set_title("DVC pipeline graph (dvc.yaml): stages and dependencies", fontsize=10, weight="bold")
    save("dvc_dag.png")


def entity(ax, x, y, name, fields, w=3.4):
    """ER entity: title bar plus attribute list (PK/FK marked)."""
    h = 0.42 * (len(fields) + 1)
    ax.add_patch(FancyBboxPatch((x, y - h), w, h, boxstyle="square,pad=0", fc="white", ec="#333", lw=1))
    ax.add_patch(FancyBboxPatch((x, y - 0.42), w, 0.42, boxstyle="square,pad=0", fc=BLUE, ec="#333", lw=1))
    ax.text(x + w / 2, y - 0.21, name, ha="center", va="center", fontsize=8, weight="bold")
    for i, f in enumerate(fields):
        ax.text(x + 0.1, y - 0.42 * (i + 1.5), f, va="center", fontsize=6.8)
    return (x, y, w, h)


def er_diagram():
    fig, ax = canvas(11, 7.2, 17.5, 11.5)
    e = {}
    e["reading"] = entity(ax, 0.2, 11.2, "MINUTE_READING (CSV)", ["PK Datetime", "Global_active_power (kW)",
                          "Global_reactive_power (kW)", "Voltage (V)", "Global_intensity (A)",
                          "Sub_metering_1..3 (Wh)"])
    e["hourly"] = entity(ax, 4.6, 11.2, "HOURLY_RECORD", ["PK datetime (hour)", "mean power, voltage, current",
                         "sum Sub_metering_1..3", "Sub_metering_other (Wh)"])
    e["feature"] = entity(ax, 9.0, 11.2, "FEATURE_ROW", ["PK datetime", "FK -> HOURLY_RECORD", "31 features (lags,",
                          "  rolling, calendar, t-1)", "target (kW)"])
    e["pred"] = entity(ax, 13.4, 11.2, "PREDICTION_LOG (JSONL)", ["time", "source (predict/forecast)",
                       "FK model -> REGISTERED_MODEL", "horizon, first_hour", "predicted_kw[ ]", "peak_hours"], w=3.9)
    e["exp"] = entity(ax, 0.2, 5.6, "EXPERIMENT", ["PK experiment_id", "name"])
    e["run"] = entity(ax, 4.6, 5.6, "RUN", ["PK run_id", "FK experiment_id", "run_name, status", "start/end time"])
    e["param"] = entity(ax, 9.0, 7.0, "PARAM", ["FK run_id", "key", "value"])
    e["metric"] = entity(ax, 9.0, 4.4, "METRIC", ["FK run_id", "key (val_mae, test_r2 ...)", "value, step"])
    e["rm"] = entity(ax, 0.2, 2.4, "REGISTERED_MODEL", ["PK name", "(voltcast-forecaster)"])
    e["mv"] = entity(ax, 4.6, 2.4, "MODEL_VERSION", ["PK name + version", "FK run_id", "status, source"])
    e["dvc"] = entity(ax, 13.4, 5.6, "DVC_STAGE (dvc.lock)", ["PK stage name", "cmd", "deps: path + md5",
                      "params", "outs: path + md5"], w=3.9)

    def link(a, b, label, side_a="r", side_b="l"):
        xa, ya, wa, ha = e[a]
        xb, yb, wb, hb = e[b]
        pa = {"r": (xa + wa, ya - ha / 2), "b": (xa + wa / 2, ya - ha), "l": (xa, ya - ha / 2), "t": (xa + wa / 2, ya)}[side_a]
        pb = {"r": (xb + wb, yb - hb / 2), "b": (xb + wb / 2, yb - hb), "l": (xb, yb - hb / 2), "t": (xb + wb / 2, yb)}[side_b]
        ax.add_patch(FancyArrowPatch(pa, pb, arrowstyle="-", lw=0.9, color="#333"))
        ax.text((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2 + 0.12, label, ha="center", fontsize=6.8, style="italic")

    link("reading", "hourly", "60 : 1")
    link("hourly", "feature", "1 : 1")
    link("exp", "run", "1 : N")
    link("run", "param", "1 : N")
    link("run", "metric", "1 : N")
    link("rm", "mv", "1 : N")
    link("mv", "run", "N : 1", "t", "b")
    link("feature", "run", "trains (N : M)", "b", "t")
    ax.text(0.2, 0.25, "Data files: CSV, Parquet and JSON Lines. Experiment metadata: MLflow's SQLite database (mlflow.db). "
            "Pipeline lineage: dvc.lock.", fontsize=7, style="italic")
    ax.set_title("Entity-relationship diagram of the data stored by VoltCast", fontsize=10, weight="bold")
    save("er_diagram.png")


def stage_flowcharts():
    flowchart("flow_preprocess.png", [
        ("Start", "terminal"), ("Read CSV, parse Datetime, cast float32", "box"),
        ("Compute unmetered energy (Sub_metering_other)", "box"), ("Resample to 1 h (mean / sum)", "box"),
        ("Interpolate gaps of up to 6 h (time-based)", "box"), ("Fill longer gaps with same hour last week", "box"),
        ("IQR outlier report (values kept)", "box"), ("Save hourly.parquet + data_stats.json", "box"), ("End", "terminal")],
        "Flowchart: preprocessing stage")
    flowchart("flow_training.png", [
        ("Start", "terminal"), ("Load features, chronological split", "box"), ("For each of 8 models", "box"),
        ("Fit on train; score train + validation", "model"), ("Log params & metrics to MLflow", "box"),
        ("Grid defined for model?", "decision"), ("GridSearchCV, TimeSeriesSplit(5)", "model"),
        ("More algorithms?", "decision"), ("Pick lowest validation RMSE", "box"),
        ("Refit on train + val, score test", "model"), ("Register in MLflow, save model.joblib", "box"), ("End", "terminal")],
        "Flowchart: training and tuning stage", decisions=[(5, 7, "no"), (7, 2, "yes (next algorithm)")],
        down_labels={7: "no"})
    flowchart("flow_classifier.png", [
        ("Start", "terminal"), ("Label: consumption > 75th percentile", "box"), ("Train default decision tree", "model"),
        ("GridSearchCV tree (F1, TimeSeriesSplit)", "model"), ("Train random forest classifier", "model"),
        ("Select best validation F1", "box"), ("Refit on train + val", "model"),
        ("Accuracy, precision, recall, F1, AUC, confusion matrix", "box"), ("Save classifier + threshold", "box"), ("End", "terminal")],
        "Flowchart: peak-hour classifier stage")
    flowchart("flow_prediction.png", [
        ("API request", "terminal"), ("Valid input?", "decision"),
        ("Build hourly history (169+ h)", "box"),
        ("h = 1: features -> forecast + classifier", "model"), ("h > 1: append forecast, recompute features", "model"),
        ("Alert = forecast > threshold (h > 1)", "box"), ("Add 95% interval, log JSON, update metrics", "box"),
        ("Return JSON forecast", "terminal")],
        "Flowchart: serving a prediction", exits=[(1, "no", "HTTP 422 +\nerror counter")])
    flowchart("flow_monitoring.png", [
        ("Start (pipeline run)", "terminal"), ("PSI of key features: train vs recent", "box"),
        ("Recent 30-day MAE vs reference test MAE", "box"), ("PSI > 0.2 or MAE +20%?", "decision"),
        ("retraining_recommended = true", "box"), ("Write monitoring.json, PSI chart", "box"), ("End", "terminal")],
        "Flowchart: drift monitoring and retraining decision", exits=[(3, "no", "flag = false,\nwrite JSON, end")])


if __name__ == "__main__":
    dfd_level0()
    dfd_level1()
    use_case()
    sequence()
    deployment()
    cicd()
    dvc_dag()
    stage_flowcharts()
    er_diagram()
    print("diagrams written")
