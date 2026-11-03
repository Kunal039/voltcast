"""Run the project's key commands in PowerShell and save their real output as terminal images for the report.

For each command: the exact command line and its combined stdout/stderr are saved to
reports/terminal/<name>.txt and rendered to reports/terminal/<name>[_partN].png in a terminal style.
Nothing is edited: long outputs are split over several numbered images instead of being cut.

Usage (from the project root, Docker Desktop running):  python report/capture_terminal.py [name ...]
"""
import json
import subprocess
import sys
import textwrap
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "terminal"
FONT = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 15)
FONT_B = ImageFont.truetype("C:/Windows/Fonts/consolab.ttf", 15)
COLS, LINES_PER_IMAGE = 118, 58
BG, FG, PROMPT, TITLE_BG = (12, 12, 12), (204, 204, 204), (97, 214, 214), (32, 32, 32)
PS_PROMPT = r"PS D:\01 Projects\VoltCast> "


def ps(cmd: str, timeout: int = 1800, cwd: Path = ROOT) -> str:
    env_prefix = "$env:MLFLOW_DISABLE_AGENT_HINT='1'; $env:PYTHONIOENCODING='utf-8'; $ProgressPreference='SilentlyContinue'; "
    r = subprocess.run(["powershell.exe", "-NoProfile", "-Command", env_prefix + cmd], cwd=cwd, capture_output=True,
                       timeout=timeout, encoding="utf-8", errors="replace")
    return (r.stdout + r.stderr).replace("\r\n", "\n").rstrip()


def render(name: str, command: str, output: str, stamp: str) -> list[str]:
    lines = []
    for raw in output.split("\n"):
        lines += textwrap.wrap(raw, COLS, replace_whitespace=False, drop_whitespace=False) or [""]
    parts = [lines[i:i + LINES_PER_IMAGE] for i in range(0, max(len(lines), 1), LINES_PER_IMAGE)] or [[]]
    files = []
    for k, chunk in enumerate(parts, 1):
        cmd_lines = textwrap.wrap(PS_PROMPT + command, COLS) if k == 1 else []
        body = cmd_lines + chunk
        w, lh = 9 + int(FONT.getlength("M") * COLS) + 18, 19
        h = 38 + 14 + lh * len(body) + 14
        img = Image.new("RGB", (w, h), BG)
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, w, 32], fill=TITLE_BG)
        title = f"Windows PowerShell - VoltCast    {stamp}" + (f"    (part {k}/{len(parts)})" if len(parts) > 1 else "")
        d.text((14, 8), title, font=FONT, fill=(170, 170, 170))
        # Windows caption buttons: minimise, maximise, close.
        gray = (170, 170, 170)
        d.line([w - 128, 17, w - 116, 17], fill=gray, width=1)
        d.rectangle([w - 82, 11, w - 71, 22], outline=gray)
        d.line([w - 34, 11, w - 23, 22], fill=gray, width=1)
        d.line([w - 34, 22, w - 23, 11], fill=gray, width=1)
        y = 38 + 8
        for i, line in enumerate(body):
            if i < len(cmd_lines):
                if i == 0:
                    d.text((9, y), PS_PROMPT, font=FONT_B, fill=PROMPT)
                    d.text((9 + FONT.getlength(PS_PROMPT), y), line[len(PS_PROMPT):], font=FONT_B, fill=(255, 255, 255))
                else:
                    d.text((9, y), line, font=FONT_B, fill=(255, 255, 255))
            else:
                d.text((9, y), line, font=FONT, fill=FG)
            y += lh
        fn = f"{name}.png" if len(parts) == 1 else f"{name}_part{k}.png"
        img.save(OUT / fn)
        files.append(fn)
    return files


def wait_http(url: str, seconds: int = 120) -> None:
    for _ in range(seconds):
        try:
            urllib.request.urlopen(url, timeout=3)
            return
        except Exception:
            time.sleep(1)


def traffic() -> None:
    """Send forecasts (and two deliberately invalid requests) so the monitoring commands have data to show."""
    for i in range(40):
        urllib.request.urlopen(f"http://localhost:8000/api/forecast?origin=2010-10-{i % 25 + 1:02d}T{i % 24:02d}:00&horizon={i % 24 + 1}")
    for bad in ["origin=bad", "origin=2010-10-01T08:00&horizon=99"]:
        try:
            urllib.request.urlopen(f"http://localhost:8000/api/forecast?{bad}")
        except Exception:
            pass
    time.sleep(20)  # let Prometheus scrape at least once


STEPS = [
    ("env_versions", "python --version; pip list 2>$null | Select-String -Pattern '^(scikit-learn|pandas|numpy|mlflow|dvc|Flask|prometheus.client|pydantic|pytest|streamlit) '"),
    ("git_log", "git log --oneline -n 15; git remote -v"),
    ("dvc_dag", "dvc dag"),
    ("dvc_repro", "dvc repro --force"),
    ("dvc_status", "dvc status; dvc params diff --all HEAD~3 2>$null | Select-Object -First 12"),
    ("dvc_metrics", "dvc metrics show --json | python scripts/show_metrics.py"),
    ("pytest", "python -m pytest -v"),
    ("mlflow_runs", "mlflow runs list --experiment-id 1"),
    ("mlflow_registry", "python scripts/list_registry.py"),
    ("npm_build", "cd web; npm run build; cd .."),
    ("docker_up", "docker compose up -d --build 2>&1 | Select-Object -Last 12; docker compose ps"),
    ("docker_images", "docker images voltcast; docker compose config --services"),
    ("api_health", "curl.exe -s http://localhost:8000/api/health; ''; curl.exe -s http://localhost:8000/api/model-info | python -m json.tool"),
    ("api_forecast", "curl.exe -s 'http://localhost:8000/api/forecast?origin=2010-11-20T08:00&horizon=4' | python -c \"import sys,json; d=json.load(sys.stdin); print(json.dumps(d['forecast'], indent=2)); print('predicted_energy_kwh =', d['predicted_energy_kwh'])\""),
    ("api_predict", "curl.exe -s http://localhost:8000/api/sample-request -o req.json; (Get-Content req.json | ConvertFrom-Json).history.Count; curl.exe -s -X POST -H 'Content-Type: application/json' --data-binary '@req.json' http://localhost:8000/api/predict | python -c \"import sys,json; d=json.load(sys.stdin); print('model:', d['model'], '| horizon:', d['horizon'], '| energy kWh:', d['predicted_energy_kwh']); [print(p['timestamp'], round(p['predicted_kw'],3), p['alert_method'], p['peak_alert']) for p in d['forecast'][:8]]\"; Remove-Item req.json"),
    ("api_invalid", "curl.exe -s -w '  -> HTTP %{http_code}' 'http://localhost:8000/api/forecast?origin=2010-10-01T08:00&horizon=99'; ''; curl.exe -s -w '  -> HTTP %{http_code}' -X POST -H 'Content-Type: application/json' -d '{\\\"history\\\": [], \\\"horizon\\\": 5}' http://localhost:8000/api/predict"),
    ("prometheus_metrics", "curl.exe -s http://localhost:8000/api/metrics | Select-String -Pattern '^voltcast_' | Select-Object -First 45"),
    ("prometheus_query", "curl.exe -s 'http://localhost:9090/api/v1/targets' | python -c \"import sys,json; [print('target', t['scrapeUrl'], 'health', t['health']) for t in json.load(sys.stdin)['data']['activeTargets']]\"; curl.exe -s 'http://localhost:9090/api/v1/query?query=sum(voltcast_predictions_total)' | python -m json.tool"),
    ("prediction_log", "Get-Content logs\\predictions.jsonl -Tail 5"),
    ("grafana_api", "curl.exe -s 'http://localhost:3000/api/search?query=VoltCast' | python -m json.tool; curl.exe -s http://localhost:3000/api/datasources | python -m json.tool"),
    ("ci_runs", "gh run list --limit 8"),
    ("ci_run_view", "gh run view (gh run list --limit 1 --json databaseId --jq '.[0].databaseId')"),
    ("vercel_ls", "npx --yes vercel ls voltcast 2>&1 | Select-Object -First 16"),
    ("live_site", "curl.exe -s https://voltcast-forecast.vercel.app/api/health; ''; curl.exe -s -o NUL -w 'web app: HTTP %{http_code}, %{size_download} bytes' https://voltcast-forecast.vercel.app/"),
    ("k8s_validate", "python -c \"import yaml; docs=list(yaml.safe_load_all(open('k8s/deployment.yaml'))); [print(d['kind'], d['metadata']['name']) for d in docs]\"; docker compose config -q; 'docker compose config: OK'"),
]


def main(selected: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for name, cmd in STEPS:
        if selected and name not in selected:
            continue
        if name == "api_health":
            wait_http("http://localhost:8000/api/health")
        if name == "prometheus_metrics":
            traffic()
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        t0 = time.perf_counter()
        output = ps(cmd)
        secs = time.perf_counter() - t0
        (OUT / f"{name}.txt").write_text(f"{PS_PROMPT}{cmd}\n{output}\n", encoding="utf-8")
        files = render(name, cmd, output, stamp)
        manifest[name] = {"command": cmd, "time": stamp, "seconds": round(secs, 1), "images": files,
                          "lines": output.count("\n") + 1}
        print(f"{name:20s} {secs:7.1f}s  {len(files)} image(s)")
        manifest_path.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
