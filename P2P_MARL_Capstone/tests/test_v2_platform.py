"""Stakeholder platform and final-model selection.

* the "final MAT" rule picks the RL-fine-tuned model only when it wins on validation;
* the degrade switch routes every household to the local policy;
* the recorded replay file is complete and self-consistent;
* the live server streams a full day and reacts to a cable cut.
"""
from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from src.control import models
from src.control.hybrid import HybridController, OutageSchedule, MODE_CENTRAL, MODE_LOCAL

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- model selection
def _fake_run(tmp_path, start_val, best_val):
    d = tmp_path / "dagger_mat/seed0"; d.mkdir(parents=True)
    json.dump({"selected_iter_on_val": 3}, open(d / "summary.json", "w"))
    f = tmp_path / "ft_mat/seed0"; f.mkdir(parents=True)
    json.dump({"start": {"val": {"net_cost_inv_neutral_eur": start_val}},
               "best": {"update": 40, "val": {"net_cost_inv_neutral_eur": best_val}}}, open(f / "summary.json", "w"))


def test_final_mat_uses_fine_tuned_model_only_if_it_wins_on_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "R", tmp_path)
    _fake_run(tmp_path, start_val=115.0, best_val=114.0)
    p, label = models.final_mat_ckpt(0)
    assert p == tmp_path / "ft_mat/seed0/best.pt" and "fine-tuning" in label


def test_final_mat_keeps_dagger_model_when_fine_tuning_does_not_help(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "R", tmp_path)
    _fake_run(tmp_path, start_val=113.0, best_val=113.5)
    p, label = models.final_mat_ckpt(0)
    assert p == tmp_path / "dagger_mat/seed0/iter3.pt" and "rejected" in label


def test_final_mat_selection_on_the_shipped_runs():
    for s in range(3):
        p, _ = models.final_mat_ckpt(s)
        assert p.exists(), p


# ---------------------------------------------------------------- degrade switch
class _Const:
    def __init__(self, v): self.v = v
    def act(self, obs, state, deterministic=True):
        return (np.full(obs.shape[:2] + (5,), self.v, np.float32),)


class _Env:
    def __init__(self, t): self.t, self.comms_online = t, None


@pytest.mark.parametrize("frac,expect_all_local", [(0.12, False), (0.3, True)])
def test_degrade_switch_routes_everyone_to_local(frac, expect_all_local):
    sched = OutageSchedule.random_households(25, frac, seed=1)
    ctrl = HybridController(_Const(0.5), _Const(-0.5), sched, "local", degrade_frac=0.2)
    a = ctrl.act(np.zeros((1, 25, 4), np.float32), None, _Env(0))
    on = sched.online(0)
    if expect_all_local:
        assert (a == -0.5).all() and (ctrl.modes[-1] == MODE_LOCAL).all()
        assert ctrl.modes[-1].shape == (25,)
    else:
        assert (a[0, on] == 0.5).all() and (a[0, ~on] == -0.5).all()
        assert (ctrl.modes[-1][on] == MODE_CENTRAL).all()


# ---------------------------------------------------------------- replay data
CTRL_KEYS = {"hybrid", "mat_mixed", "mat_rule", "mat_stale", "mappo", "rule"}


def _traces():
    p = ROOT / "platform/data/traces.js"
    s = p.read_text()
    return json.loads(s[s.index("=") + 1:].strip().rstrip(";"))


def test_replay_file_is_complete_and_consistent():
    d = _traces()
    n = len(d["meta"]["households"])
    assert n == 25 and set(d["scenarios"]) == {"normal", "evening", "one", "partial", "full"}
    assert [c["key"] for c in d["meta"]["controllers"]][0] == "hybrid"
    for key, sc in d["scenarios"].items():
        assert set(sc["runs"]) == CTRL_KEYS
        for name, run in sc["runs"].items():
            fr, sm = run["frames"], run["summary"]
            assert len(fr) == 96 and [f["t"] for f in fr] == list(range(96))
            assert all(len(f["hh"]) == n and len(f["hh"][0]) == 7 for f in fr)
            assert all(len(f["pv"]) == n and min(f["pv"]) >= 0 for f in fr)
            assert abs(fr[-1]["cost"] - sm["cost"]) < 0.02
            assert sm["unserved_kwh"] < 1e-3 and sm["ev_shortfall_kwh"] < 1e-3
            assert abs(sum(sm["settled_bills"]) - sm["cost"]) < 0.05      # budget-balanced settlement
            assert all(f["feeder_kw"] <= f["limit_kw"] + 1e-3 for f in fr)
        rows = {r["key"]: r for r in sc["compare"]["rows"]}
        assert set(rows) == CTRL_KEYS and sorted(r["rank"] for r in rows.values()) == list(range(1, 7))
        assert rows["hybrid"]["cost_battery_neutral"] == sc["runs"]["hybrid"]["summary"]["cost_battery_neutral"]
    ev = d["scenarios"]["evening"]["runs"]["hybrid"]["frames"]
    assert ev[67]["online"] == 1.0 and ev[68]["online"] == 0.0 and ev[84]["online"] == 1.0
    # headline result visible to stakeholders: the proposed controller wins every scenario
    for sc in d["scenarios"].values():
        assert min(sc["compare"]["rows"], key=lambda r: r["cost_battery_neutral"])["key"] in ("hybrid", "mat_mixed")


# ---------------------------------------------------------------- live server
def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); return s.getsockname()[1]


@pytest.fixture(scope="module")
def server():
    port = _free_port()
    proc = subprocess.Popen([sys.executable, str(ROOT / "platform/server.py"), "--port", str(port)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=str(ROOT))
    base = f"http://127.0.0.1:{port}"
    for _ in range(240):
        try:
            urllib.request.urlopen(base + "/api/health", timeout=1); break
        except Exception:
            if proc.poll() is not None:
                raise RuntimeError(proc.stdout.read().decode())
            time.sleep(0.5)
    yield base
    proc.terminate(); proc.wait(10)


def _get(url):
    return json.load(urllib.request.urlopen(url, timeout=120))


def _post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        return 200, json.load(urllib.request.urlopen(req, timeout=300))
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def _stream(url, on_frame=None):
    r = urllib.request.urlopen(url, timeout=300)
    frames, summary, hello, event = [], None, None, None
    for raw in r:
        line = raw.decode().rstrip("\n")
        if line.startswith("event: "):
            event = line[7:]
        elif line.startswith("data: "):
            data = json.loads(line[6:])
            if event == "hello":
                hello = data
            elif event == "frame":
                frames.append(data)
                if on_frame:
                    on_frame(hello, data)
            elif event == "summary":
                summary = data
                break
    return hello, frames, summary


def test_server_serves_pages_and_health(server):
    h = _get(server + "/api/health")
    assert h["live"] and len(h["default"]["houses"]) == 25 and h["feeder_limit_kw"] == 200.0
    assert [c["key"] for c in h["controllers"]][0] == "hybrid" and "evening" in h["scenarios"]
    assert h["limits"]["n_households"] == [4, 60]
    assert "three.min.js" in urllib.request.urlopen(server + "/").read().decode()
    assert "Feeder Control Room" in urllib.request.urlopen(server + "/classic.html").read().decode()


def test_defaults_endpoint_draws_a_community(server):
    d = _get(server + "/api/defaults?n=12&seed=2")
    assert d["n_households"] == 12 and len(d["houses"]) == 12 and d["seed"] == 2


def test_custom_session_streams_a_day_and_reacts_to_a_cable_cut(server):
    spec = _get(server + "/api/defaults?n=9&seed=1")
    spec["houses"][0]["pv_kw"] = 7.0
    code, s = _post(server + "/api/session", {"spec": spec, "scenario": "normal", "controller": "hybrid"})
    assert code == 200 and len(s["households"]) == 9 and s["households"][0]["pv_kw"] == 7.0
    assert s["feeder_limit_kw"] == 72.0 and "trained on" in s["note"]

    def cut_at_10(hello, f):
        if f["t"] == 9:
            assert _get(f"{server}/api/control?session={hello['session']}&cut=1")["cut"]
    hello, frames, summary = _stream(f"{server}/api/stream?session={s['session']}&speed=32", cut_at_10)
    assert len(frames) == 96 and len(frames[0]["hh"]) == 9 and summary is not None
    assert frames[0]["online"] == 1.0 and all(f["online"] == 0.0 for f in frames[15:])
    assert all({h[0] for h in f["hh"]} == {1} for f in frames[15:])            # everyone on local MAPPO
    assert summary["unserved_kwh"] < 1e-3 and summary["ev_shortfall_kwh"] < 1e-3


def test_preset_stream_without_a_session_still_works(server):
    _, frames, summary = _stream(f"{server}/api/stream?scenario=evening&controller=rule&speed=32")
    assert len(frames) == 96 and summary["unserved_kwh"] < 1e-3


def test_compare_endpoint(server):
    spec = _get(server + "/api/defaults?n=8&seed=4")
    code, res = _post(server + "/api/compare", {"spec": spec, "scenario": {"kind": "window", "start": 68, "end": 84},
                                                 "controllers": ["hybrid", "mat_stale", "rule"]})
    assert code == 200 and {r["key"] for r in res["rows"]} == {"hybrid", "mat_stale", "rule"}
    assert set(res["runs"]) == {"hybrid", "mat_stale", "rule"} and len(res["runs"]["rule"]["frames"]) == 96
    assert all(r["unserved_kwh"] < 1e-3 for r in res["rows"])


@pytest.mark.parametrize("body,msg", [
    ({"spec": {"n_households": 200}}, "n_households"),
    ({"spec": None, "scenario": {"kind": "moon"}}, "scenario"),
    ({"spec": None, "controller": "magic"}, "controller"),
])
def test_bad_requests_get_a_clear_400(server, body, msg):
    code, res = _post(server + "/api/session", body)
    assert code == 400 and msg in res["error"]
