"""Controller variants and outage scenarios for the stakeholder platform.

    models = load_models(seed=0)
    frames, summary = run_day(ct, env_cfg, "hybrid", "evening", models)
    table = compare(ct, env_cfg, {"kind": "homes", "share": 0.3}, models)

``simulate`` is the generator behind both: it yields ("frame", dict) for each
15-minute step and finally ("summary", dict).  The live server iterates it and
can cut the operator link at any moment through the ``cut`` callback.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np

from ..control.hybrid import HybridController, OutageSchedule, RuleAgent, proposed_controller
from ..control.models import final_mappo_ckpt, final_mat_ckpt
from ..env.community_env import P2PCommunityEnv
from ..game.core import owen_allocation
from ..marl.mat import load_any
from .custom import SpecError
from .recorder import day_summary, frame

CONTROLLERS = [
    ("hybrid", "MAT + MAPPO fallback (proposed)",
     "Central MAT while every home is reachable; if any home is cut off, every home switches to its local "
     "MAPPO student. Cut-off homes stay inside the 8 kW envelope."),
    ("mat_mixed", "MAT + MAPPO for cut-off homes only",
     "MAT keeps controlling the reachable homes on a stale picture of the others; only cut-off homes run MAPPO."),
    ("mat_rule", "MAT + rule fallback",
     "Central MAT; cut-off homes fall back to the simple tariff rule controller."),
    ("mat_stale", "MAT, no fallback",
     "Central MAT; a cut-off home keeps repeating the last setpoint it received (a naive central system)."),
    ("mappo", "MAPPO everywhere (decentralised)",
     "Every home always runs its own MAPPO student on its own data; no central controller."),
    ("rule", "Rule controller",
     "Every home follows a tariff rule: charge when power is cheap, use storage when it is dear."),
]
SCENARIOS = {
    "normal": ("Normal day", {"kind": "none"}),
    "evening": ("Evening outage 17:00-21:00", {"kind": "window", "start": 68, "end": 84}),
    "one": ("One home cut off all day", {"kind": "homes", "share": 0.0, "count": 1, "seed": 0}),
    "partial": ("30% of homes cut off all day", {"kind": "homes", "share": 0.3, "seed": 0}),
    "full": ("Operator down all day", {"kind": "full"}),
}


def controllers() -> List[Dict]:
    return [{"key": k, "label": l, "description": d} for k, l, d in CONTROLLERS]


def scenarios() -> Dict[str, str]:
    return {k: v[0] for k, v in SCENARIOS.items()}


def load_models(seed: int = 0):
    return {"mat": load_any(final_mat_ckpt(seed)[0])[0], "mappo": load_any(final_mappo_ckpt(seed))[0]}


def _step(v, name):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= 96:
        raise SpecError(f"{name} must be a step between 0 and 96 (15-minute steps)")
    return int(v)


def schedule(scn, n: int) -> OutageSchedule:
    """An OutageSchedule from a preset name or a dict {kind, start, end, share, seed}."""
    if isinstance(scn, str):
        if scn not in SCENARIOS:
            raise SpecError(f"unknown scenario '{scn}'")
        label, scn = SCENARIOS[scn]
    if not isinstance(scn, dict):
        raise SpecError("the scenario must be a preset name or an object")
    kind = scn.get("kind")
    t0, t1 = _step(scn.get("start", 0), "start"), _step(scn.get("end", 96), "end")
    if t1 <= t0 and kind != "none":
        raise SpecError("the outage must end after it starts")
    if kind == "none":
        return OutageSchedule.none(n)
    if kind == "window":
        return OutageSchedule.window(n, t0, t1)
    if kind == "full":
        return OutageSchedule(n, "operator down", [(t0, t1, None)])
    if kind == "homes":
        share = scn.get("share", 0.3)
        if isinstance(share, bool) or not isinstance(share, (int, float)) or not 0 <= share <= 1:
            raise SpecError("share must be between 0 and 1")
        k = int(scn.get("count") or max(1, round(share * n)))
        k = min(max(k, 1), n)
        hh = np.random.default_rng(int(scn.get("seed", 0))).choice(n, k, replace=False).tolist()
        return OutageSchedule(n, f"{k} homes cut off", [(t0, t1, hh)])
    raise SpecError(f"unknown scenario kind '{kind}'")


class _Cuttable:
    """Wraps a schedule so a viewer can cut every link at any moment."""

    def __init__(self, base: OutageSchedule, cut: Optional[Callable[[], bool]]):
        self.base, self.cut, self.n_agents = base, cut, base.n_agents

    def online(self, t):
        on = self.base.online(t)
        return np.zeros_like(on) if (self.cut and self.cut()) else on


def make_controller(key: str, models: Dict, sched) -> HybridController:
    mat, mappo = models["mat"], models["mappo"]
    if key == "hybrid":
        return proposed_controller(mat, mappo, sched)
    if key == "mat_mixed":
        return HybridController(mat, mappo, sched, "local")
    if key == "mat_rule":
        return HybridController(mat, None, sched, "rule")
    if key == "mat_stale":
        return HybridController(mat, None, sched, "stale")
    if key == "mappo":
        return HybridController(mappo, mappo, sched, "local")
    if key == "rule":
        return HybridController(RuleAgent(), None, sched, "rule")
    raise SpecError(f"unknown controller '{key}'")


def settlement_weights(ct, env_cfg):
    return owen_allocation(ct, env_cfg)["x"]


def simulate(ct, env_cfg, key: str, scn, models: Dict, x_core=None,
             cut: Optional[Callable[[], bool]] = None) -> Iterator[Tuple[str, Dict]]:
    sched = _Cuttable(schedule(scn, ct.n_agents), cut)
    ctrl = make_controller(key, models, sched)
    env = P2PCommunityEnv(ct, replace(env_cfg, n_envs=1, noise_std=0.0, residual=False, action_mode="power"),
                          seed=12345)
    obs, st = env.reset(seed=12345, deterministic=True)
    done, t, info = False, 0, None
    while not done:
        obs, st, _, done, info = env.step(ctrl.act(obs, st, env))
        fr = frame(env, ctrl, t)
        if key == "rule":
            for h in fr["hh"]:
                h[0] = 2
        fr["cut"] = bool(cut and cut())
        yield "frame", fr
        t += 1
    yield "summary", day_summary(env, info, x_core)


def run_day(ct, env_cfg, key, scn, models, x_core=None):
    frames, summary = [], None
    for kind, payload in simulate(ct, env_cfg, key, scn, models, x_core):
        if kind == "frame":
            frames.append(payload)
        else:
            summary = payload
    return frames, summary


def compare(ct, env_cfg, scn, models, keys: Optional[List[str]] = None, x_core=None,
            with_frames: bool = False) -> Dict:
    """Run several variants on the same community and scenario; rank by battery-neutral cost."""
    keys = keys or [k for k, _, _ in CONTROLLERS]
    labels = {k: l for k, l, _ in CONTROLLERS}
    rows, frames = [], {}
    grid_only = None
    for k in keys:
        fr, sm = run_day(ct, env_cfg, k, scn, models, x_core)
        grid_only = sm["grid_only"]
        rows.append({"key": k, "label": labels[k], "cost": sm["cost"],
                     "cost_battery_neutral": sm["cost_battery_neutral"],
                     "unserved_kwh": sm["unserved_kwh"], "ev_shortfall_kwh": sm["ev_shortfall_kwh"],
                     "p2p_kwh": sm["p2p_kwh"], "peak_kw": round(max(f["feeder_kw"] for f in fr), 1),
                     "saving_pct": round(max(0.0, 100 * (sm["grid_only"] - sm["cost_battery_neutral"]) / sm["grid_only"]), 1),
                     "all_better_off": sm.get("all_better_off")})
        if with_frames:
            frames[k] = {"frames": fr, "summary": sm}
    for r, row in enumerate(sorted(rows, key=lambda r: r["cost_battery_neutral"]), 1):
        row["rank"] = r
    out = {"grid_only": grid_only, "rows": rows}
    if with_frames:
        out["runs"] = frames
    return out
