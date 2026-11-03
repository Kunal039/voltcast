"""Compact view of the DVC-tracked metrics:  dvc metrics show --json | python scripts/show_metrics.py"""
import json
import sys

data = json.load(sys.stdin)
files = {fn.replace("\\", "/"): m["data"] for rev in data.values() for fn, m in rev["data"].items()}


def row(*cells, widths):
    print("  ".join(str(c).ljust(w) if i == 0 else str(c).rjust(w) for i, (c, w) in enumerate(zip(cells, widths))))


d = files.get("reports/data_stats.json", {})
if d:
    print(f"reports/data_stats.json: {d['raw_rows']:,} raw rows, {d['raw_missing_pct']}% missing, "
          f"{d['hourly_rows']:,} hourly rows ({d['start']} .. {d['end']})")

m = files.get("reports/metrics.json", {})
if m:
    print("\nreports/metrics.json - validation (default hyper-parameters)")
    w = (24, 10, 9, 9, 9, 8)
    row("model", "train_mae", "val_mae", "val_rmse", "val_r2", "fit_s", widths=w)
    for n, v in m["validation"].items():
        row(n, f"{v['train_mae']:.4f}", f"{v['mae']:.4f}", f"{v['rmse']:.4f}", f"{v['r2']:.4f}", f"{v['fit_seconds']:.1f}", widths=w)
    print("\nreports/metrics.json - GridSearchCV tuning")
    w = (24, 9, 12, 10, 42)
    row("model", "cv_mae", "default_mae", "tuned_mae", "best_params", widths=w)
    for n, t in m["tuning"].items():
        row(n, f"{t['cv_mae']:.4f}", f"{t['default_val']['mae']:.4f}", f"{t['tuned_val']['mae']:.4f}",
            ", ".join(f"{k}={v}" for k, v in t["best_params"].items()), widths=w)
    print("\nreports/metrics.json - test (selected model vs baseline)")
    w = (24, 9, 9, 9, 9, 8)
    row("model", "mae", "mse", "rmse", "r2", "mape", widths=w)
    for n, t in m["test"].items():
        row(n, f"{t['mae']:.4f}", f"{t['mse']:.4f}", f"{t['rmse']:.4f}", f"{t['r2']:.4f}", f"{t['mape']:.1f}", widths=w)

c = files.get("reports/classification.json", {})
if c:
    print(f"\nreports/classification.json - peak threshold {c['threshold_kw']:.3f} kW, selected {c['selected']}")
    w = (32, 9, 10, 8, 8, 8)
    row("classifier / split", "accuracy", "precision", "recall", "f1", "roc_auc", widths=w)
    for n, s in c["validation"].items():
        row(f"{n} (val)", f"{s['accuracy']:.4f}", f"{s['precision']:.4f}", f"{s['recall']:.4f}", f"{s['f1']:.4f}", f"{s['roc_auc']:.4f}", widths=w)
    t = c["test"]
    row(f"{c['selected']} (test)", f"{t['accuracy']:.4f}", f"{t['precision']:.4f}", f"{t['recall']:.4f}", f"{t['f1']:.4f}", f"{t['roc_auc']:.4f}", widths=w)

mo = files.get("reports/monitoring.json", {})
if mo:
    drifted = [k for k, v in mo["psi"].items() if v > 0.2]  # DVC only exposes numeric values, so re-derive the list
    print(f"\nreports/monitoring.json - drifted (PSI > 0.2): {', '.join(drifted)}; recent MAE {mo['recent_30d_mae']} "
          f"({mo['mae_degradation_pct']:+.1f}%); retraining_recommended = {mo['retraining_recommended']}")
    print("  PSI: " + ", ".join(f"{k}={v:.3f}" for k, v in mo["psi"].items()))

f = files.get("reports/fairness.json", {})
if f:
    print(f"\nreports/fairness.json - overall test MAE {f['overall_mae']:.4f}")
    for name, groups in f["slices"].items():
        print(f"  {name:12s} " + ", ".join(f"{g}={v['mae']:.3f}" for g, v in groups.items()))

a = files.get("reports/analysis.json", {})
if a:
    print("\nreports/analysis.json - every model on the test set")
    w = (30, 8, 8, 8, 12, 10)
    row("model", "mae", "rmse", "r2", "acc(1-WAPE)", "±0.5kW %", widths=w)
    for n, t in a["test_all_models"].items():
        row(n, f"{t['mae']:.4f}", f"{t['rmse']:.4f}", f"{t['r2']:.4f}", f"{t['accuracy_1_minus_wape']:.4f}", f"{t['within_0_5kw_pct']:.1f}", widths=w)
