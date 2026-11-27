"""Turn a simulated day into frames for the stakeholder platform.

One frame per 15-minute step, the same format for recorded replays
(platform/data/traces.js) and the live server (platform/server.py):

    {"t": step, "clock": "HH:MM", "feeder_kw": net community import (kW),
     "limit_kw": feeder limit, "price": import tariff (EUR/kWh, mean),
     "p2p_kwh": energy traded peer-to-peer this step,
     "cost": cumulative community cost so far (EUR),
     "grid_only": cumulative cost of the grid-only counterfactual (EUR),
     "unserved": cumulative unserved energy (kWh),
     "online": share of households the operator can reach,
     "hh": [[mode, battery_soc, ev1_soc, ev2_soc, ev1_home, ev2_home, net_kw], ...],
     "pv": [solar output of each household this step (kW), ...]}

``mode``: 0 central (MAT), 1 local fallback (MAPPO student), 2 local rule
fallback, 3 holding last setpoint, -1 not applicable (policy without modes).
SoC values are fractions 0..1 (-1 for a household without that device).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np


def frame(env, ctrl, t: int) -> Dict:
    ct = env.ct
    modes = ctrl.modes[-1] if getattr(ctrl, "modes", None) else np.full(env.N, -1)
    has_b = np.asarray(ct.has_bess, bool)
    soc_b = np.where(has_b, env.bess_soc[0] / np.maximum(ct.bess_capacity, 1e-9), -1)
    evs = []
    for j in range(2):
        has = ct.ev_charge_max[j] > 0
        evs.append(np.where(has, env.ev_soc[j, 0] / np.maximum(ct.ev_capacity[j], 1e-9), -1))
    home = [(ct.ev_at_home[j, t] > 0.5) & (ct.ev_charge_max[j] > 0) for j in range(2)]
    net = env.last_residual_kw[0]
    online = 1.0 if env.comms_online is None else float(env.comms_online[0].mean())
    hh = [[int(modes[i]), round(float(soc_b[i]), 3), round(float(evs[0][i]), 3), round(float(evs[1][i]), 3),
           int(home[0][i]), int(home[1][i]), round(float(net[i]), 2)] for i in range(env.N)]
    return {"t": t, "clock": f"{t // 4:02d}:{(t % 4) * 15:02d}",
            "feeder_kw": round(float(env.last_net_community[0]), 2), "limit_kw": float(env.feeder_import_kw),
            "price": round(float(ct.buy_price[t].mean()), 4),
            "p2p_kwh": round(float(env.last_volume[0]), 3),
            "cost": round(float(env.cum_cost[0].sum()), 3),
            "grid_only": round(float(env.baseline_cost_step[:t + 1].sum()), 3),
            "unserved": round(float(env.metrics["unserved_kwh"][0]), 3),
            "online": round(online, 3), "hh": hh,
            "pv": [round(float(v), 2) for v in np.asarray(ct.pv_kw[t]).reshape(-1)[:env.N]]}


def day_summary(env, info, x_core=None) -> Dict:
    ep = {k: float(np.atleast_1d(v)[0]) for k, v in info["episode"].items()}
    out = {"cost": round(ep["net_cost_eur"], 2), "cost_battery_neutral": round(ep["net_cost_inv_neutral_eur"], 2),
           "grid_only": round(float(env.baseline_cost_total.sum()), 2),
           "unserved_kwh": round(ep["unserved_kwh"], 3), "ev_shortfall_kwh": round(ep["ev_shortfall_kwh"], 3),
           "p2p_kwh": round(ep["p2p_volume_kwh"], 1),
           "raw_bills": [round(float(b), 2) for b in env.cum_cost[0]],
           "grid_only_bills": [round(float(b), 2) for b in env.baseline_cost_total]}
    if x_core is not None:
        from ..game.core import settle
        s = settle(env.baseline_cost_total, float(env.cum_cost[0].sum()), x_core)
        out["settled_bills"] = [round(float(b), 2) for b in s["pay"]]
        out["all_better_off"] = bool(s["ir"].all()) and s["savings"] >= 0
    return out


def households(ct) -> List[Dict]:
    """Static description of each household for the platform."""
    out = []
    for i in range(ct.n_agents):
        out.append({"id": i + 1,
                    "pv_kw": round(float(ct.pv_kw[:, i].max()), 1) if ct.has_pv[i] else 0.0,
                    "battery_kwh": round(float(ct.bess_capacity[i]), 1) if ct.has_bess[i] else 0.0,
                    "evs": int((ct.ev_charge_max[:, i] > 0).sum())})
    return out
