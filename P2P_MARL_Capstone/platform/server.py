"""Live stakeholder platform: simulate any community in the browser, in real time.

    python platform/server.py            then open http://localhost:8765
    python platform/server.py --port 9000

The page (platform/index.html) is the same one used for recorded replays; when
it is served from here it switches to LIVE mode: viewers can build their own
community (number of homes, solar, batteries, cars, demand, feeder size), pick
an outage scenario and a controller, watch the day run, cut the operator's
communication link at any moment, and compare every controller side by side.
Standard library only.

Endpoints
  GET  /api/health                         controllers, scenarios, limits, default community
  GET  /api/defaults?n=25&seed=0           a community specification drawn from the dataset
  POST /api/session  {spec, scenario, controller}  -> {session, households, feeder_limit_kw, note}
  GET  /api/stream?session=ID&speed=2      Server-Sent Events: hello, frame x96, summary
  GET  /api/stream?scenario=evening&controller=hybrid   (study community, no session needed)
  GET  /api/control?session=ID&cut=1|0&speed=N          change a running day
  POST /api/compare  {spec, scenario, controllers?}     -> ranked table + every run's frames
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
import uuid
from collections import OrderedDict
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

torch.set_num_threads(1)

from src.platform import sim  # noqa: E402
from src.platform.custom import LIMITS, SpecError, build_community, default_spec  # noqa: E402

NOTE_CUSTOM = ("The controllers were trained on the 25-home study community. On this community they run "
               "unchanged and the safety layer still holds, but costs are a generalisation test.")


class State:
    """Models and built communities, shared by every request."""

    def __init__(self, model_seed: int = 0):
        self.models = sim.load_models(model_seed)
        self.default = default_spec()
        self.communities: "OrderedDict[str, tuple]" = OrderedDict()
        self.sessions: dict = {}
        self.lock = threading.Lock()          # one simulation step at a time (small CPU models)
        self.community(self.default)

    def community(self, spec):
        key = hashlib.md5(json.dumps(spec, sort_keys=True).encode()).hexdigest()
        if key not in self.communities:
            ct, cfg, meta = build_community(spec)
            self.communities[key] = (ct, cfg, meta, sim.settlement_weights(ct, cfg))
            while len(self.communities) > 8:
                self.communities.popitem(last=False)
        return self.communities[key]


STATE: State | None = None


def _clean_spec(spec):
    if spec is None:
        return STATE.default
    if not isinstance(spec, dict):
        raise SpecError("spec must be an object")
    out = dict(STATE.default)
    out.update({k: v for k, v in spec.items() if k in ("n_households", "seed", "feeder_kw_per_home",
                                                       "export_kw_per_home", "houses")})
    if "houses" not in spec and out["n_households"] != STATE.default["n_households"]:
        out["houses"] = []
    return out


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(ROOT / "platform"), **kw)

    def log_message(self, fmt, *args):
        pass

    def end_headers(self):
        if self.path.endswith((".html", ".js", "/")):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _json(self, obj, code=200):
        body = json.dumps(obj, separators=(",", ":")).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > 2_000_000:
            raise SpecError("request too large")
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            raise SpecError("the request body is not valid JSON") from None

    # ------------------------------------------------------------ GET
    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path == "/api/health":
                ct, cfg, meta, _ = STATE.community(STATE.default)
                return self._json({"live": True, "controllers": sim.controllers(), "scenarios": sim.scenarios(),
                                   "limits": {k: list(v) for k, v in LIMITS.items()}, "default": STATE.default,
                                   "households": meta, "feeder_limit_kw": cfg.feeder_import_kw_per_hh * ct.n_agents})
            if u.path == "/api/defaults":
                n, seed = int(q.get("n", 25)), int(q.get("seed", 0))
                lo, hi = LIMITS["n_households"]
                if not lo <= n <= hi:
                    raise SpecError(f"n_households must be between {lo} and {hi}")
                return self._json(default_spec(n, seed))
            if u.path == "/api/control":
                s = STATE.sessions.get(q.get("session", ""))
                if s is None:
                    return self._json({"error": "unknown or finished session"}, 404)
                if "cut" in q:
                    s["cut"] = q["cut"] == "1"
                if "speed" in q:
                    s["speed"] = max(0.25, min(32.0, float(q["speed"])))
                return self._json({"ok": True, "cut": s["cut"], "speed": s["speed"]})
            if u.path == "/api/stream":
                return self._stream(q)
        except (SpecError, ValueError) as e:
            return self._json({"error": str(e)}, 400)
        return super().do_GET()

    # ------------------------------------------------------------ POST
    def do_POST(self):
        u = urlparse(self.path)
        try:
            body = self._body()
            if u.path == "/api/session":
                spec = _clean_spec(body.get("spec"))
                scn = body.get("scenario", "normal")
                ctrl = body.get("controller", "hybrid")
                ct, cfg, meta, _ = STATE.community(spec)
                sim.schedule(scn, ct.n_agents)
                if ctrl not in {c["key"] for c in sim.controllers()}:
                    raise SpecError(f"unknown controller '{ctrl}'")
                sid = uuid.uuid4().hex[:12]
                STATE.sessions[sid] = {"spec": spec, "scenario": scn, "controller": ctrl, "cut": False,
                                       "speed": 2.0, "started": False, "created": time.time()}
                custom = spec != STATE.default
                return self._json({"session": sid, "households": meta,
                                   "feeder_limit_kw": cfg.feeder_import_kw_per_hh * ct.n_agents,
                                   "note": NOTE_CUSTOM if custom else ""})
            if u.path == "/api/compare":
                spec = _clean_spec(body.get("spec"))
                scn = body.get("scenario", "normal")
                keys = body.get("controllers") or [c["key"] for c in sim.controllers()]
                valid = {c["key"] for c in sim.controllers()}
                if not isinstance(keys, list) or not set(keys) <= valid:
                    raise SpecError("controllers must be a list of known controller keys")
                ct, cfg, meta, x = STATE.community(spec)
                sim.schedule(scn, ct.n_agents)
                with STATE.lock:
                    res = sim.compare(ct, cfg, scn, STATE.models, keys, x_core=x, with_frames=True)
                res["households"] = meta
                res["feeder_limit_kw"] = cfg.feeder_import_kw_per_hh * ct.n_agents
                return self._json(res)
        except SpecError as e:
            return self._json({"error": str(e)}, 400)
        return self._json({"error": "not found"}, 404)

    # ------------------------------------------------------------ SSE
    def _send(self, event, data):
        self.wfile.write(f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n".encode())
        self.wfile.flush()

    def _stream(self, q):
        sid = q.get("session")
        if sid:
            session = STATE.sessions.get(sid)
            if session is None or session["started"]:
                raise SpecError("unknown or already used session")
        else:                                   # preset day on the study community
            sid = uuid.uuid4().hex[:12]
            session = {"spec": STATE.default, "scenario": q.get("scenario", "normal"),
                       "controller": {"proposed": "hybrid"}.get(q.get("controller"), q.get("controller", "hybrid")),
                       "cut": False, "speed": 2.0}
            STATE.sessions[sid] = session
        if "speed" in q:
            session["speed"] = max(0.25, min(32.0, float(q["speed"])))
        session["started"] = True
        ct, cfg, meta, x = STATE.community(session["spec"])
        gen = sim.simulate(ct, cfg, session["controller"], session["scenario"], STATE.models, x_core=x,
                           cut=lambda: session["cut"])
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            self._send("hello", {"session": sid, "scenario": session["scenario"], "controller": session["controller"]})
            while True:
                t0 = time.time()
                with STATE.lock:
                    kind, payload = next(gen)
                self._send(kind, payload)
                if kind == "summary":
                    break
                time.sleep(max(0.0, 1.0 / session["speed"] - (time.time() - t0)))
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            STATE.sessions.pop(sid, None)


def main():
    global STATE
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--model-seed", type=int, default=0, choices=[0, 1, 2])
    a = ap.parse_args()
    print("loading models ...", flush=True)
    STATE = State(a.model_seed)
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    srv.daemon_threads = True
    print(f"Feeder Control Room (LIVE) -> http://{a.host}:{a.port}   (Ctrl+C to stop)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
