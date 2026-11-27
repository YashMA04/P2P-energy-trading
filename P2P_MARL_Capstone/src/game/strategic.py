"""Strategic robustness: can one self-interested household profit by deviating?

Two attack surfaces are measured.

1. **Market manipulation** -- a household rewrites its own P2P quote and
   offered quantity (bid shading, withholding) while everyone else follows
   the cooperative policy.  Its best response is found by exhaustive search
   over a grid of constant (quote, quantity) deviations, separately for each
   household.  Payoff is evaluated two ways:
     * raw:     the household's own auction bill;
     * settled: its bill under the core-guided settlement
                (:func:`src.game.core.settle`), where a household's payment
                depends on the *community* saving, not on its own trades.
   Exploitability = max over households of (deviation payoff gain).

2. **Data misreporting to the settlement** -- the settlement's shares come
   from an LP over households' *reported* load/PV/EV needs.  A household may
   misreport them to inflate its share; we scale its reported load and EV
   requirement and recompute its settled bill against the truthful day.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from typing import Callable, Dict, Optional

import numpy as np

from ..env.community_env import A_PRICE, A_QTY, EnvConfig, P2PCommunityEnv
from .core import owen_allocation, settle


def rollout_bills(ct, env_cfg: EnvConfig, policy, is_agent: bool,
                  hook: Optional[Callable] = None, multipliers=None) -> Dict[str, np.ndarray]:
    env = P2PCommunityEnv(ct, replace(env_cfg, n_envs=1, noise_std=0.0), seed=12345)
    if multipliers:
        env.set_multipliers(multipliers)
    env.action_hook = hook
    obs, st = env.reset(seed=12345, deterministic=True)
    done = False
    while not done:
        a = policy.act(obs, st, True)[0] if is_agent else policy.act(obs, st, env)
        obs, st, _, done, info = env.step(a)
    return {"bill": env.cum_cost[0].copy(), "total": float(env.cum_cost[0].sum()),
            "grid_only": env.baseline_cost_total.copy(),
            "unserved": float(info["episode"]["unserved_kwh"][0])}


def quote_hook(h: int, price: float, qty: float):
    def hook(a, env):
        a = a.copy()
        a[:, h, A_PRICE] = price
        a[:, h, A_QTY] = a[:, h, A_QTY] * qty
        return a
    return hook


def market_exploitability(ct, env_cfg, policy, is_agent: bool, x_core: np.ndarray,
                          prices=(0.0, 0.25, 0.5, 0.75, 1.0), qtys=(0.0, 0.5, 1.0),
                          households=None, multipliers=None) -> Dict[str, object]:
    base = rollout_bills(ct, env_cfg, policy, is_agent, multipliers=multipliers)
    base_set = settle(base["grid_only"], base["total"], x_core)["pay"]
    H = range(ct.n_agents) if households is None else households
    rows = []
    for h in H:
        best_raw, best_set, arg = 0.0, 0.0, None
        for p in prices:
            for q in qtys:
                r = rollout_bills(ct, env_cfg, policy, is_agent, quote_hook(h, p, q), multipliers)
                gain_raw = base["bill"][h] - r["bill"][h]
                gain_set = base_set[h] - settle(r["grid_only"], r["total"], x_core)["pay"][h]
                if gain_raw > best_raw:
                    best_raw, arg = gain_raw, (p, q, r["total"] - base["total"])
                best_set = max(best_set, gain_set)
        rows.append({"household": int(h), "best_gain_raw_eur": best_raw,
                     "best_gain_settled_eur": best_set,
                     "best_raw_deviation": arg})
    return {"base_total": base["total"], "rows": rows,
            "exploitability_raw_eur": max(r["best_gain_raw_eur"] for r in rows),
            "exploitability_settled_eur": max(r["best_gain_settled_eur"] for r in rows)}


def misreport_gain(ct, env_cfg, realised_total: float, grid_only: np.ndarray, h: int,
                   load_scale: float = 1.0, ev_scale: float = 1.0) -> float:
    """Gain (EUR) to household h from misreporting its data to the settlement LP."""
    truthful = owen_allocation(ct, env_cfg)["x"]
    pay_true = settle(grid_only, realised_total, truthful)["pay"][h]
    load = ct.load_kw.copy(); load[:, h] *= load_scale
    ct_lie = replace(ct, load_kw=load,
                     ev_departure_target=_scale_col(ct.ev_departure_target, h, ev_scale),
                     ev_final_target=_scale_col(ct.ev_final_target, h, ev_scale))
    lied = owen_allocation(ct_lie, env_cfg)["x"]
    # the settlement compares against the TRUE grid-only bills and the TRUE realised total
    pay_lie = settle(grid_only, realised_total, lied)["pay"][h]
    return float(pay_true - pay_lie)


def _scale_col(arr, h, s):
    out = np.array(arr, copy=True)
    out[:, h] = out[:, h] * s
    return out
