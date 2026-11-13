"""Draws the report-only diagrams: architecture, Gantt chart, design alternatives, flowchart."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "diagrams"  # kept apart from pipeline-generated figures
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 9, "figure.dpi": 150})
BLUE, ORANGE, GREEN, GREY = "#dce8f7", "#fbe3cc", "#dcefdc", "#eeeeee"


def box(ax, x, y, w, h, text, color=BLUE, size=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color, ec="#333", lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size)


def arrow(ax, p, q, text=None, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=11, lw=0.9, ls=ls, color="#333"))
    if text:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + 0.12, text, ha="center", fontsize=7, style="italic")


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w * 1.6)
    ax.set_ylim(0, h * 1.6)
    ax.axis("off")
    return fig, ax


def architecture():
    fig, ax = canvas(9, 4.2)
    # Row 1: data & training
    xs = [0.2, 2.6, 5.0, 7.4, 9.8, 12.2]
    labels = ["UCI dataset\n2.07 M minute\nreadings", "Data ingestion\n& schema\nvalidation",
              "Cleaning &\nhourly\nresampling", "Feature\nengineering\n(31 features)",
              "Train 8 models\nGridSearchCV\n+ MLflow", "MLflow model\nregistry\n(best model)"]
    for x, t in zip(xs, labels):
        box(ax, x, 4.6, 2.0, 1.25, t)
    for a, b in zip(xs, xs[1:]):
        arrow(ax, (a + 2.0, 5.22), (b, 5.22))
    # Row 2: serving & monitoring
    box(ax, 12.2, 2.3, 2.0, 1.25, "Flask REST API\n(Docker / K8s\n/ Vercel)", ORANGE)
    box(ax, 9.8, 2.3, 2.0, 1.25, "React web app\n+ Streamlit", ORANGE)
    box(ax, 7.4, 2.3, 2.0, 1.25, "End users /\nsmart-home\napps", GREY)
    box(ax, 12.2, 0.2, 2.0, 1.25, "Prometheus +\nGrafana, PSI\ndrift, logs", GREEN)
    box(ax, 0.2, 0.2, 8.6, 1.4, "CI / CT (GitHub Actions): dvc repro -> quality gate (beat baseline)\n"
                             "-> pytest -> Docker build -> upload artifacts\n"
                             "Triggered on push, pull request and a weekly schedule", GREEN, size=8)
    arrow(ax, (13.2, 4.6), (13.2, 3.55))
    arrow(ax, (12.2, 2.92), (11.8, 2.92))
    arrow(ax, (9.8, 2.92), (9.4, 2.92))
    arrow(ax, (13.2, 2.3), (13.2, 1.45))
    arrow(ax, (12.2, 0.82), (8.8, 0.82), "retraining trigger", ls="--")
    arrow(ax, (1.2, 1.6), (1.2, 4.6), "re-runs\npipeline", ls="--")
    ax.text(0.2, 6.25, "Data & training layer", fontsize=9, weight="bold")
    ax.text(7.4, 3.75, "Serving & operations layer", fontsize=9, weight="bold")
    plt.savefig(OUT / "fig_architecture.png", bbox_inches="tight")
    plt.close()


def gantt():
    tasks = [
        ("Problem identification & literature survey", 0, 2),
        ("Dataset selection & exploratory analysis", 1, 2),
        ("Data pipeline: ingestion, cleaning, features", 2, 2),
        ("Model training & MLflow experiment tracking", 4, 2),
        ("Model evaluation & selection", 5, 1.5),
        ("Flask API, Docker & Kubernetes", 6, 2),
        ("React web app & Streamlit dashboard", 7, 1.5),
        ("CI/CD (GitHub Actions) + tests", 8, 1.5),
        ("Prometheus, Grafana & drift detection", 9, 1),
        ("Report writing & presentation", 9, 3),
    ]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for i, (t, s, d) in enumerate(tasks):
        ax.barh(i, d, left=s, color="#1f5fa8" if i < 5 else "#e07b24", height=0.55)
    ax.set_yticks(range(len(tasks)), [t[0] for t in tasks])
    ax.invert_yaxis()
    ax.set_xticks(range(13), [f"W{i}" for i in range(1, 14)])
    ax.set_xlim(0, 12)
    ax.grid(axis="x", ls=":", alpha=0.6)
    ax.set_title("Project timeline (12 weeks)")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUT / "fig_gantt.png", bbox_inches="tight")
    plt.close()


def alternatives():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax in axes:
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 7)
        ax.axis("off")
    a = ["Raw CSV", "Notebook: clean +\nfeatures + LSTM", "Pickle file\nsaved manually", "Ad-hoc script\npredictions"]
    b = ["Versioned raw data", "Modular pipeline stages\n(params.yaml / dvc.yaml)",
         "8 models, GridSearchCV\n+ MLflow registry", "Flask API + Docker\n+ React", "CI/CT + drift\nmonitoring"]
    for ax, steps, title, col in [(axes[0], a, "Design A: Notebook-centric deep learning", GREY),
                                  (axes[1], b, "Design B: Modular MLOps pipeline (selected)", BLUE)]:
        ax.set_title(title, fontsize=9.5, weight="bold")
        h = 6.4 / len(steps)
        for i, s in enumerate(steps):
            y = 6.6 - (i + 1) * h
            box(ax, 1.0, y + 0.1, 4.0, h - 0.35, s, col)
            if i:
                arrow(ax, (3.0, y + h + 0.1), (3.0, y + h - 0.25))
    plt.tight_layout()
    plt.savefig(OUT / "fig_design_alternatives.png", bbox_inches="tight")
    plt.close()


def flowchart():
    fig, ax = plt.subplots(figsize=(5.2, 7.4))
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 13.6)
    ax.axis("off")
    steps = [("Start", GREY), ("Download & validate dataset", BLUE),
             ("Parse timestamps, mark '?' as missing", BLUE),
             ("Resample to hourly; interpolate gaps <= 6 h", BLUE),
             ("Build lag, rolling & calendar features", BLUE),
             ("Chronological split: train / validation / test", BLUE),
             ("Train 8 models + GridSearchCV; log to MLflow", ORANGE),
             ("Select lowest validation RMSE; refit on train+val", ORANGE),
             ("Beats persistence baseline on test?", GREEN),
             ("Register model, deploy Flask API + web app", ORANGE),
             ("Monitor PSI drift & rolling MAE", GREEN), ("End / retrain loop", GREY)]
    for i, (s, c) in enumerate(steps):
        y = 12.7 - i * 1.12
        box(ax, 0.6, y, 4.8, 0.72, s, c, size=8)
        if i:
            arrow(ax, (3.0, y + 1.12), (3.0, y + 0.72))
    ax.text(5.5, 12.7 - 8 * 1.12 + 0.2, "No: tune\nfeatures /\nparams", fontsize=7, style="italic")
    arrow(ax, (5.4, 12.7 - 8 * 1.12 + 0.36), (5.4, 12.7 - 6 * 1.12 + 0.36), style="-|>", ls="--")
    plt.savefig(OUT / "fig_flowchart.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    architecture()
    gantt()
    alternatives()
    flowchart()
