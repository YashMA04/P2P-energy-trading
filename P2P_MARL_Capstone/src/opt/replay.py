"""Turn an optimiser schedule into simulator actions and settle it there.

This is the check that keeps the optimisation layer honest: a schedule is only
worth its LP cost if the *simulator* -- with its own physics, feeder logic,
market and shield -- settles it at (close to) that cost.  It is also how the
privileged teacher's decisions become action labels for the student policy.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from typing import Dict

import numpy as np

from ..env.community_env import (A_BESS, A_EV1, A_EV2, A_PRICE, A_QTY, ACTION_DIM,
                                 EnvConfig, P2PCommunityEnv)


def schedule_to_actions(ct, x: Dict[str, np.ndarray], k: int,
                        households=None) -> np.ndarray:
    """(M, 5) raw (non-residual) action for schedule step ``k``."""
    idx = np.arange(ct.n_agents) if households is None else np.asarray(households)
    M = len(idx)
    a = np.zeros((M, ACTION_DIM), np.float32)
    bnet = x["bc"][k] - x["bd"][k]
    a[:, A_BESS] = np.where(bnet >= 0,
                            bnet / np.maximum(ct.bess_charge_max[idx], 1e-9),
                            bnet / np.maximum(ct.bess_discharge_max[idx], 1e-9))
    t = None
    for j, slot in enumerate((A_EV1, A_EV2)):
        ch, dis = x["ec"][j, k], x["ed"][j, k]
        a[:, slot] = np.where(ch >= dis,
                              (ch - dis) / np.maximum(ct.ev_charge_max[j, idx], 1e-9),
                              -(dis - ch) / np.maximum(ct.ev_discharge_max[j, idx], 1e-9))
    # Offer the whole imbalance; buyers bid the import tariff and sellers ask
    # the export tariff, so every feasible match clears (welfare-maximising
    # under uniform pricing; the clearing price is a transfer).
    a[:, A_QTY] = 1.0
    net_buy = (x["imp"][k] + x["pin"][k]) > (x["exp"][k] + x["pout"][k])
    a[:, A_PRICE] = np.where(net_buy, 1.0, 0.0)
    return np.clip(a, [-1, -1, -1, -1, 0], [1, 1, 1, 1, 1]).astype(np.float32)


def v2g_scale_fix(ct, a: np.ndarray, t: int) -> np.ndarray:
    """The simulator scales negative EV commands by min(discharge_max, v2g_max[t])."""
    out = a.copy()
    for j, slot in enumerate((A_EV1, A_EV2)):
        cap = np.minimum(ct.ev_discharge_max[j], ct.ev_v2g_max[j, t])
        neg = out[:, slot] < 0
        ratio = np.where(cap > 1e-9, ct.ev_discharge_max[j] / np.maximum(cap, 1e-9), 0.0)
        out[:, slot] = np.where(neg, np.clip(out[:, slot] * ratio, -1, 0), out[:, slot])
    return out


def replay_in_simulator(ct, env_cfg: EnvConfig, x: Dict[str, np.ndarray],
                        shield: bool = True) -> Dict[str, float]:
    """Settle a full-day schedule in the simulator (deterministic day)."""
    cfg = replace(env_cfg, n_envs=1, noise_std=0.0, residual=False,
                  ev_safety_projection=shield)
    env = P2PCommunityEnv(ct, cfg, seed=0)
    env.reset(seed=0, deterministic=True)
    done = False
    while not done:
        a = v2g_scale_fix(ct, schedule_to_actions(ct, x, env.t), env.t)
        _, _, _, done, info = env.step(a[None])
    ep = {k: float(np.atleast_1d(v)[0]) for k, v in info["episode"].items()}
    ep["bess_end_minus_start_kwh"] = float(env.bess_soc.sum() - ct.bess_soc0.sum())
    return ep
