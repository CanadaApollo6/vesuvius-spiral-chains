"""Headless end-to-end test for check_viewer.py: drives the real UI in headless chromium over the DevTools protocol
(keyboard only, like the user), times item loads, takes screenshots and checks the jsonl log and resume.

  uv run python tools/check_viewer.py shots --queue data/scratch/hitl/check_queue_standin_band1.json
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import aiohttp

KEYS = {"Enter": ("Enter", 13), "Backspace": ("Backspace", 8), "ArrowLeft": ("ArrowLeft", 37),
        "ArrowRight": ("ArrowRight", 39)}


class Page:
    def __init__(self, ws):
        self.ws = ws
        self.i = 0

    async def cmd(self, method, **params):
        self.i += 1
        my = self.i
        await self.ws.send_json(dict(id=my, method=method, params=params))
        while True:
            m = await self.ws.receive_json()
            if m.get("id") == my:
                if "error" in m:
                    raise RuntimeError(m["error"])
                return m.get("result", {})

    async def js(self, expr):
        r = await self.cmd("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)
        return r.get("result", {}).get("value")

    async def key(self, k):
        if k in KEYS:
            name, vk = KEYS[k]
            p = dict(key=name, code=name, windowsVirtualKeyCode=vk)
            await self.cmd("Input.dispatchKeyEvent", type="rawKeyDown", **p)
            await self.cmd("Input.dispatchKeyEvent", type="keyUp", **p)
        else:
            p = dict(key=k, text=k, code="Key" + k.upper() if k.isalpha() else "Digit" + k,
                     windowsVirtualKeyCode=ord(k.upper()))
            await self.cmd("Input.dispatchKeyEvent", type="keyDown", **p)
            await self.cmd("Input.dispatchKeyEvent", type="keyUp", **p)

    async def wait(self, expr, timeout=30.0):
        t = time.time()
        while time.time() - t < timeout:
            if await self.js(expr):
                return time.time() - t
            await asyncio.sleep(0.05)
        raise TimeoutError(expr)

    async def shot(self, path: Path):
        r = await self.cmd("Page.captureScreenshot", format="png")
        path.write_bytes(base64.b64decode(r["data"]))


READY = "(() => { try { return !st.busy && document.getElementById('loading').style.display !== 'flex' " \
        "&& document.getElementById('imAx').complete && document.getElementById('imAx').naturalWidth > 0 } " \
        "catch (e) { return false } })()"


async def drive(url: str, screens: Path, port_dbg: int) -> dict:
    out = {}
    async with aiohttp.ClientSession() as s:
        for _ in range(100):
            try:
                async with s.get(f"http://127.0.0.1:{port_dbg}/json") as r:
                    tabs = await r.json()
                    tabs = [t for t in tabs if t.get("type") == "page"]
                    if tabs:
                        break
            except aiohttp.ClientError:
                pass
            await asyncio.sleep(0.1)
        async with s.ws_connect(tabs[0]["webSocketDebuggerUrl"], max_msg_size=0) as ws:
            p = Page(ws)
            await p.cmd("Page.enable")
            await p.cmd("Emulation.setDeviceMetricsOverride", width=1920, height=1080, deviceScaleFactor=1, mobile=False)
            await p.cmd("Page.navigate", url=url)
            await p.wait("typeof st !== 'undefined' && st.mode === 'intro'")
            await p.key("Enter")
            ntut = await p.js("st.ntut")
            tut_times = []
            for i in range(ntut):
                await p.wait(f"st.mode === 'tutorial' && st.tutIdx === {i} && " + READY)
                await asyncio.sleep(0.3)
                await p.key("1")
                await p.wait("st.tutAnswered")
                if i == 3:  # the touching-sheets example, revealed
                    await asyncio.sleep(0.3)
                    await p.shot(screens / "1_tutorial_touching_reveal.png")
                tut_times.append(await p.js("(performance.now() - st.t0) / 1000"))
                await p.key("Enter")
            await p.wait("st.mode === 'ready'")
            await p.key("Enter")
            loads = []
            plan = ["0", "1", "h1", "u", "2", "0", "back", "1", "0", "1"]
            for step in plan:
                await p.wait("st.mode === 'main' && " + READY)
                loads.append(await p.js("window.__lastLoadMs"))
                await asyncio.sleep(0.25)
                if step == "back":
                    await p.key("Backspace")
                    await p.wait("st.mode === 'main' && st.revisit && " + READY)
                    continue
                if step.startswith("h"):
                    await p.key("h")
                    await asyncio.sleep(0.6)
                    await p.shot(screens / "3_item_2p4um.png")
                    step = step[1:]
                if len(loads) == 2:
                    await p.shot(screens / "2_item_9p4um.png")
                await p.key(step)
                if step != "back":
                    await p.wait("st.busy === false")
            await p.wait("st.mode === 'main' && " + READY)
            out["tutorial_answer_secs"] = tut_times
            out["client_load_ms"] = loads
            out["answers_in_ui"] = await p.js("Object.keys(st.answers).length")
            out["pos"] = await p.js("st.pos")
    return out


def run_shots(a) -> None:
    repo = Path(__file__).resolve().parents[1]
    screens = Path(a.screens)
    screens.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="hitl_e2e_"))
    log = tmp / "answers_test.jsonl"
    srv = subprocess.Popen([sys.executable, str(repo / "tools" / "check_viewer.py"), "--queue", a.queue, "--out",
                            str(log), "--port", str(a.port), "--no-browser"], stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, start_new_session=True)
    dbg = a.port + 1
    prof = tmp / "chrome"
    chrome = subprocess.Popen([shutil.which("chromium") or "chromium", "--headless=new", "--disable-gpu",
                               f"--remote-debugging-port={dbg}", f"--user-data-dir={prof}", "--window-size=1920,1080",
                               "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                              start_new_session=True)
    try:
        url = f"http://127.0.0.1:{a.port}/"
        for _ in range(100):
            try:
                urllib.request.urlopen(url + "api/state", timeout=1)
                break
            except OSError:
                time.sleep(0.1)
        res = asyncio.run(drive(url, screens, dbg))
        recs = [json.loads(x) for x in log.read_text().splitlines() if x.strip()]
        tut = [json.loads(x) for x in log.with_suffix(".tutorial.jsonl").read_text().splitlines() if x.strip()]
        st = json.loads(urllib.request.urlopen(url + "api/state").read())
        res.update(n_log=len(recs), n_tutorial_log=len(tut), log_sample=recs[2] if len(recs) > 2 else recs,
                   hires_flags=[r["hires_opened"] for r in recs], revisits=[r["revisit"] for r in recs],
                   answers=[r["answer"] for r in recs], seconds=[r["seconds"] for r in recs],
                   resume_first=st["first"], resume_answered=len(st["answers"]), tutorial_done=st["tutorial_done"])
        print(json.dumps(res, indent=1))
        assert st["tutorial_done"] and st["first"] == len(st["answers"]) == 8, st["first"]
        assert res["hires_flags"][2] and not res["hires_flags"][0]
        assert any(res["revisits"])
        print("E2E OK; screenshots in", screens)
    finally:
        for pr in (chrome, srv):
            try:
                os.killpg(pr.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
