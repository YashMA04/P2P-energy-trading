"""Privileged teacher: exact dispatch on perturbed days -> (observation, action) labels.

The teacher sees the whole perturbed day (perfect foresight) and solves the
exact dispatch program for it.  Its schedule is replayed through the simulator
on *that same* perturbed day, recording what a household would actually
observe at each step and the action the teacher took.  The student policy is
then trained to map observation -> teacher action, i.e. to imitate the
optimum using only causal information.  This is privileged-teacher / student
distillation (as used for sim-to-real control), applied here per household.

Labels are in the simulator's *absolute* action convention (``residual=False``)
because the optimum is frequently outside the reach of a residual correction
around the rule controller (e.g. V2G discharge where the rule never discharges).
"""
from __future__ import annotations

import time
from dataclasses import replace
from typing import Dict, List

import numpy as np

from ..env.community_env import EnvConfig, P2PCommunityEnv
from .lp import solve_dispatch
from .replay import schedule_to_actions, v2g_scale_fix


def soc_target_labels(ct, data: Dict[str, np.ndarray], dt: float = 0.25) -> np.ndarray:
    """Convert stored teacher power labels to end-of-step SoC-target labels.

    The SoC after step t is read from the observation at t+1 (obs 3, 7, 12
    are battery/EV state-of-charge fractions); the last step is integrated
    from the power label.  Steps where an EV is away carry its current SoC
    (the converter ignores them).  EV trips are deducted at the departure
    step while the car is away, so they never enter a label.
    """
    obs, act = data["obs"], data["act"]
    lab = act.copy()
    soc_idx = (3, 7, 12)
    for k, oi in enumerate(soc_idx):
        nxt = np.concatenate([obs[:, 1:, :, oi], obs[:, -1:, :, oi]], axis=1)
        lab[..., k] = 2.0 * nxt - 1.0
    # last step: battery integrated from the power label
    cap, eta = ct.bess_capacity, np.sqrt(ct.bess_eta)
    soc = obs[:, -1, :, 3] * cap
    cmd = act[:, -1, :, 0]
    p = np.where(cmd >= 0, cmd * ct.bess_charge_max, cmd * ct.bess_discharge_max)
    soc_end = soc + np.where(p >= 0, p * dt * eta, p * dt / eta)
    lab[:, -1, :, 0] = 2.0 * soc_end / cap - 1.0
    for j in range(2):
        capj, ee = ct.ev_capacity[j], np.sqrt(ct.ev_eta[j])
        socj = obs[:, -1, :, 7 + 5 * j] * capj
        c = act[:, -1, :, 1 + j]
        pj = np.where(c >= 0, c * ct.ev_charge_max[j], c * ct.ev_discharge_max[j])
        lab[:, -1, :, 1 + j] = 2.0 * (socj + np.where(pj >= 0, pj * dt * ee, pj * dt / ee)) / capj - 1.0
    return np.clip(lab, -1.0, 1.0).astype(np.float32)


def generate_teacher_data(ct, env_cfg: EnvConfig, n_days: int, seed: int = 0,
                          noise_std: float = 0.02, include_nominal: bool = True,
                          time_limit: float = 60.0, verbose: bool = True) -> Dict[str, np.ndarray]:
    """Return obs (D,T,N,O), state (D,T,S), act (D,T,N,5), plus per-day costs."""
    cfg = replace(env_cfg, n_envs=1, noise_std=noise_std, residual=False)
    env = P2PCommunityEnv(ct, cfg, seed=seed)
    O, S = env.observation_dim, env.state_dim
    obs_l, st_l, act_l, info_l = [], [], [], []
    rng = np.random.default_rng(seed)
    t_start = time.time()
    for d in range(n_days):
        nominal = include_nominal and d == 0
        obs, st = env.reset(seed=int(rng.integers(1 << 31)), deterministic=nominal)
        r = solve_dispatch(ct, env_cfg, load=env.load[0], pv=env.pv[0], exact=True,
                           terminal_bess=True, time_limit=time_limit)
        if r is None or not np.isfinite(r.cost):
            continue
        O_d, S_d, A_d = [], [], []
        done = False
        while not done:
            a = v2g_scale_fix(ct, schedule_to_actions(ct, r.x, env.t), env.t)
            O_d.append(obs[0]); S_d.append(st[0]); A_d.append(a)
            obs, st, _, done, info = env.step(a[None])
        ep = info["episode"]
        obs_l.append(np.stack(O_d)); st_l.append(np.stack(S_d)); act_l.append(np.stack(A_d))
        info_l.append([r.cost, float(ep["net_cost_eur"][0]), float(ep["net_cost_inv_neutral_eur"][0]),
                       float(ep["unserved_kwh"][0]), float(ep["ev_shortfall_kwh"][0]), float(nominal)])
        if verbose:
            print(f"  teacher day {d + 1}/{n_days}: LP {r.cost:.2f} -> sim {info_l[-1][1]:.2f} "
                  f"(unserved {info_l[-1][3]:.2f}) [{time.time() - t_start:.0f}s]", flush=True)
    return {"obs": np.stack(obs_l).astype(np.float32), "state": np.stack(st_l).astype(np.float32),
            "act": np.stack(act_l).astype(np.float32),
            "day_info": np.asarray(info_l, np.float64),
            "day_info_cols": np.array(["lp_cost", "sim_cost", "sim_inv_neutral", "unserved",
                                       "ev_shortfall", "is_nominal"])}
