"""Screenshot any page with a headless Chromium browser (Edge/Chrome) for the report.

Usage: python report/capture_page.py <browser.exe> <url> <out.png> [wait_seconds] [width] [height]
"""
import asyncio
import base64
import json
import subprocess
import sys
import tempfile
import time
import urllib.request

import websockets


async def capture(browser: str, url: str, out: str, wait: float, width: int, height: int) -> None:
    proc = subprocess.Popen([browser, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                             "--remote-debugging-port=9334", f"--user-data-dir={tempfile.mkdtemp()}", "about:blank"])
    try:
        for _ in range(50):
            try:
                targets = json.load(urllib.request.urlopen("http://127.0.0.1:9334/json"))
                break
            except OSError:
                time.sleep(0.2)
        ws_url = next(t for t in targets if t["type"] == "page")["webSocketDebuggerUrl"]
        async with websockets.connect(ws_url, max_size=50_000_000) as ws:
            n = 0

            async def cmd(method, **params):
                nonlocal n
                n += 1
                await ws.send(json.dumps({"id": n, "method": method, "params": params}))
                while True:
                    r = json.loads(await ws.recv())
                    if r.get("id") == n:
                        return r.get("result", {})

            await cmd("Emulation.setDeviceMetricsOverride", width=width, height=height, deviceScaleFactor=1, mobile=False)
            await cmd("Page.navigate", url=url)
            await asyncio.sleep(wait)
            shot = await cmd("Page.captureScreenshot", format="png")
            with open(out, "wb") as f:
                f.write(base64.b64decode(shot["data"]))
    finally:
        proc.terminate()


if __name__ == "__main__":
    a = sys.argv
    asyncio.run(capture(a[1], a[2], a[3], float(a[4]) if len(a) > 4 else 8,
                        int(a[5]) if len(a) > 5 else 1440, int(a[6]) if len(a) > 6 else 1000))
