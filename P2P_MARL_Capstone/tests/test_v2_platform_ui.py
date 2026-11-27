"""End-to-end browser tests of the 3D platform (platform/index.html).

Skipped when Playwright or its browser is not installed. The page loads three.js
from a CDN; set P2P_THREE_DIR to an unpacked ``three@0.128.0`` npm package to
serve it locally instead (no internet needed).

    pip install playwright && python -m playwright install chromium
    python -m pytest tests/test_v2_platform_ui.py -q
"""
from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

sync_api = pytest.importorskip("playwright.sync_api")
ROOT = Path(__file__).resolve().parents[1]
THREE = os.environ.get("P2P_THREE_DIR")
ARGS = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as p:
        try:
            b = p.chromium.launch(args=ARGS)
        except Exception as e:                      # browser binaries not installed
            pytest.skip(f"chromium not available: {e}")
        yield b
        b.close()


def _page(browser, w=1440, h=900):
    page = browser.new_page(viewport={"width": w, "height": h})
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" and "fonts" not in m.text
            and "ERR_FAILED" not in m.text and "net::" not in m.text else None)
    if THREE:
        base = Path(THREE)
        page.route(re.compile(r"cdnjs\.cloudflare\.com/ajax/libs/three\.js/r128/three\.min\.js"),
                   lambda r: r.fulfill(body=(base / "build/three.min.js").read_bytes(), content_type="text/javascript"))
        page.route(re.compile(r"cdn\.jsdelivr\.net/npm/three@0\.128\.0/(.*)"),
                   lambda r: r.fulfill(body=(base / r.request.url.split("three@0.128.0/")[1]).read_bytes(),
                                       content_type="text/javascript"))
    page.route(re.compile(r"fonts\.(googleapis|gstatic)\.com"), lambda r: r.abort())
    return page, errors


def _state(page, expr):
    return page.evaluate(f"() => {{ const S = window.__S; return {expr}; }}")


def test_replay_page_plays_compares_and_switches_variants(browser):
    page, errors = _page(browser)
    page.goto((ROOT / "platform/index.html").as_uri(), wait_until="domcontentloaded", timeout=90000)
    page.wait_for_function("() => window.__S && window.__S.frames.length === 96", timeout=60000)
    assert _state(page, "S.run.ctrl") == "hybrid" and _state(page, "S.households.length") == 25
    page.click("#openCompare")
    page.wait_for_selector("#cmpBody tbody tr")
    assert page.locator("#cmpBody tbody tr").count() == 6
    assert "MAT + MAPPO fallback" in page.locator("#cmpBody tbody tr").first.inner_text()
    page.click('[data-watch="rule"]')
    assert _state(page, "S.run.ctrl") == "rule" and page.locator("#compareWrap").is_hidden()
    page.click("#openSetup"); page.click("#tabCommunity")
    assert page.locator("#replayOnly").is_visible() and page.locator("#nHomes").is_disabled()
    page.click("#tabRun"); page.check('input[name="scen"][value="evening"]'); page.check('input[name="ctrl"][value="mat_stale"]')
    page.click("#runDay")
    assert _state(page, "S.run.scen") == "evening" and _state(page, "S.run.ctrl") == "mat_stale"
    page.evaluate("() => { const s = document.getElementById('scrub'); s.value = 95.9; s.dispatchEvent(new Event('input')); }")
    page.click("#openBills")
    assert page.locator("#ledger tr").count() == 26 and "better off" in page.inner_text("#billsBadge").lower()
    assert errors == []
    page.close()


def test_phone_width_has_no_horizontal_scroll(browser):
    page, errors = _page(browser, 390, 844)
    page.goto((ROOT / "platform/index.html").as_uri(), wait_until="domcontentloaded", timeout=90000)
    page.wait_for_function("() => window.__S && window.__S.frames.length === 96", timeout=60000)
    assert page.evaluate("document.documentElement.scrollWidth") <= 390
    assert errors == []
    page.close()


@pytest.fixture(scope="module")
def server():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
    proc = subprocess.Popen([sys.executable, str(ROOT / "platform/server.py"), "--port", str(port)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=str(ROOT))
    base = f"http://127.0.0.1:{port}"
    for _ in range(240):
        try:
            urllib.request.urlopen(base + "/api/health", timeout=1); break
        except Exception:
            time.sleep(0.5)
    yield base
    proc.terminate(); proc.wait(10)


def test_live_custom_community_runs_cuts_and_compares(browser, server):
    page, errors = _page(browser)
    page.goto(server + "/", wait_until="domcontentloaded", timeout=90000)
    page.wait_for_function("() => window.__S && window.__S.server", timeout=60000)
    assert page.locator("#setup").is_visible() and page.inner_text("#modeBadge").strip().upper() == "LIVE"
    # build an 8-home community and give home 1 a large solar array
    page.click("#tabCommunity")
    page.fill("#nHomesNum", "8"); page.dispatch_event("#nHomesNum", "change")
    page.fill("#drawSeed", "3"); page.click("#drawHomes")
    page.wait_for_function("() => window.__S.spec.n_households === 8")
    page.fill('[data-i="0"][data-k="pv_kw"]', "9"); page.dispatch_event('[data-i="0"][data-k="pv_kw"]', "change")
    assert "1 change" in page.inner_text("#changedCount").lower()
    page.click("#tabRun"); page.check('input[name="ctrl"][value="hybrid"]')
    page.check('input[name="scen"][value="normal"]')
    page.select_option("#speed", "8")
    page.click("#runDay")
    page.wait_for_function("() => window.__S.frames.length >= 6", timeout=60000)
    assert _state(page, "S.households.length") == 8 and _state(page, "S.households[0].pv_kw") == 9
    page.click("#cut")
    page.wait_for_function("() => window.__S.summary !== null", timeout=120000)
    fr = _state(page, "S.frames.map(f => f.online)")
    assert len(fr) == 96 and fr[0] == 1.0 and fr[-1] == 0.0
    assert _state(page, "S.summary.unserved_kwh") < 1e-3
    # compare every controller on the same community
    page.click("#openCompare")
    page.wait_for_selector("#cmpBody tbody tr", timeout=180000)
    assert page.locator("#cmpBody tbody tr").count() == 6
    page.click('[data-watch="mappo"]')
    assert _state(page, "S.run.ctrl") == "mappo" and _state(page, "S.frames.length") == 96
    assert errors == []
    page.close()


# ---------------------------------------------------------------- 2D page (platform/classic.html)
def test_2d_replay_has_every_variant_compare_and_bills(browser):
    page, errors = _page(browser, 1280, 900)
    page.goto((ROOT / "platform/classic.html").as_uri(), wait_until="domcontentloaded", timeout=90000)
    page.wait_for_function("() => window.__S && window.__S.frames.length === 96", timeout=60000)
    assert page.locator("#board .hh").count() == 25
    page.click("#openSetup")
    assert page.locator('input[name="ctrl"]').count() == 6 and page.locator('input[name="scen"]').count() == 5
    page.check('input[name="scen"][value="partial"]'); page.check('input[name="ctrl"][value="mat_rule"]'); page.click("#runDay")
    assert _state(page, "S.run.ctrl") == "mat_rule" and _state(page, "S.run.scen") == "partial"
    page.evaluate("() => { const s = document.getElementById('scrub'); s.value = 50; s.dispatchEvent(new Event('input')); }")
    assert page.locator('#board .hh[data-mode="2"]').count() == 8           # 30% cut off -> rule fallback
    page.click("#board .hh >> nth=3")
    assert page.locator("#inspect").is_visible() and "home 4" in page.inner_text("#iTitle").lower()
    page.click("#openCompare"); page.wait_for_selector("#cmpBody tbody tr")
    assert page.locator("#cmpBody tbody tr").count() == 6
    page.click('[data-watch="hybrid"]'); assert _state(page, "S.run.ctrl") == "hybrid"
    page.evaluate("() => { const s = document.getElementById('scrub'); s.value = 95.9; s.dispatchEvent(new Event('input')); }")
    page.click("#openBills"); assert page.locator("#ledger tr").count() == 26
    assert page.evaluate("document.documentElement.scrollWidth") <= 1280
    assert errors == []
    page.close()


def test_2d_live_custom_community(browser, server):
    page, errors = _page(browser)
    page.goto(server + "/classic.html", wait_until="domcontentloaded", timeout=90000)
    page.wait_for_function("() => window.__S && window.__S.server", timeout=60000)
    page.click("#tabCommunity"); page.fill("#nHomesNum", "6"); page.dispatch_event("#nHomesNum", "change")
    page.click("#drawHomes"); page.wait_for_function("() => window.__S.spec.n_households === 6")
    page.fill("#bBat", "12"); page.click("#bApply")
    assert all(h["battery_kwh"] == 12 for h in _state(page, "S.spec.houses"))
    page.click("#tabRun"); page.check('input[name="scen"][value="full"]'); page.select_option("#speed", "8"); page.click("#runDay")
    page.wait_for_function("() => window.__S.summary !== null", timeout=120000)
    assert page.locator("#board .hh").count() == 6 and _state(page, "S.frames.length") == 96
    assert all(h["battery_kwh"] == 12 for h in _state(page, "S.households"))
    assert _state(page, "S.summary.unserved_kwh") < 1e-3
    assert errors == []
    page.close()
