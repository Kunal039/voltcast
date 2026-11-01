"""Captures dashboard screenshots (one per tab) for the user manual via Edge/Chrome DevTools protocol.

Requires the dashboard running at http://localhost:8501 and a Chromium browser.
Usage: python report/capture_screenshots.py "<path to msedge.exe or chrome.exe>"
"""
import asyncio
import base64
import json
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import websockets

OUT = Path(__file__).resolve().parents[1] / "reports" / "screenshots"
TABS = ["Forecast", "Data explorer", "Model performance", "Monitoring", "MLOps pipeline"]


async def main(browser: str) -> None:
    profile = tempfile.mkdtemp()
    proc = subprocess.Popen([browser, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                             "--remote-debugging-port=9333", f"--user-data-dir={profile}",
                             "--window-size=1440,1000", "about:blank"])
    try:
        for _ in range(50):
            try:
                targets = json.load(urllib.request.urlopen("http://127.0.0.1:9333/json"))
                break
            except OSError:
                time.sleep(0.2)
        ws_url = next(t for t in targets if t["type"] == "page")["webSocketDebuggerUrl"]
        async with websockets.connect(ws_url, max_size=50_000_000) as ws:
            msg_id = 0

            async def cmd(method, **params):
                nonlocal msg_id
                msg_id += 1
                await ws.send(json.dumps({"id": msg_id, "method": method, "params": params}))
                while True:
                    r = json.loads(await ws.recv())
                    if r.get("id") == msg_id:
                        return r.get("result", {})

            async def wait_for(js, timeout=60):
                for _ in range(timeout * 2):
                    r = await cmd("Runtime.evaluate", expression=js, returnByValue=True)
                    if r.get("result", {}).get("value"):
                        return
                    await asyncio.sleep(0.5)

            await cmd("Page.enable")
            await cmd("Emulation.setDeviceMetricsOverride", width=1440, height=1000, deviceScaleFactor=1, mobile=False)
            await cmd("Page.navigate", url="http://localhost:8501")
            await wait_for("document.querySelectorAll('.js-plotly-plot').length >= 4 && "
                           "!document.querySelector('[data-testid=\"stStatusWidget\"]')")
            await asyncio.sleep(3)
            for i, tab in enumerate(TABS):
                await cmd("Runtime.evaluate", expression=(
                    "[...document.querySelectorAll('[role=tab]')]"
                    f".find(b => b.innerText.includes({json.dumps(tab)})).click()"))
                await asyncio.sleep(3)
                shot = await cmd("Page.captureScreenshot", format="png", captureBeyondViewport=False)
                if "data" not in shot:
                    await asyncio.sleep(3)
                    shot = await cmd("Page.captureScreenshot", format="png")
                (OUT / f"dash_{i + 1}.png").write_bytes(base64.b64decode(shot["data"]))
                print("saved", tab)
    finally:
        proc.terminate()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
