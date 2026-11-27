"""DAgger (Ross, Gordon & Bagnell, 2011) with the exact dispatch program as expert.

Behaviour cloning trains only on states the *teacher* visits.  The student's
small errors take it to states the teacher never visits (e.g. an EV already
charged above target at 14:00), where its actions are unconstrained by any
label -- and errors compound over 96 steps.  Measured here: a student with
validation MSE 0.012 costs EUR 194/day in closed loop against the teacher's 100.

DAgger rolls out the *student*, asks the expert what it would do from each
visited state (the exact MILP re-solved from that state to the end of the
day), and adds those labels.  Relabelling every ``every`` steps with a time
limit keeps the cost to a few minutes per day; a time-limited MILP returns its
best incumbent, a near-optimal feasible plan, which is a valid label.
"""
from __future__ import annotations

import multiprocessing as mp
from dataclasses import replace
from typing import Dict, List

import numpy as np

from ..env.community_env import EnvConfig, P2PCommunityEnv
from .lp import solve_dispatch
from .replay import schedule_to_actions, v2g_scale_fix


def _one_day(args):
    ct, env_cfg, ckpt, day_seed, every, time_limit, beta = args
    import torch
    torch.set_num_threads(1)
    from ..marl.mat import load_any
    agent, _ = load_any(ckpt)
    env = P2PCommunityEnv(ct, replace(env_cfg, n_envs=1, noise_std=0.02, residual=False,
                                      action_mode="power"), seed=day_seed)
    obs, st = env.reset(seed=day_seed)
    rng = np.random.default_rng(day_seed)
    O, S, A, info = [], [], [], None
    fails = 0
    done = False
    while not done:
        t = env.t
        a_s = agent.act(obs, st, True)[0]
        a = a_s
        if t % every == 0:
            r = solve_dispatch(ct, env_cfg, load=env.load[0], pv=env.pv[0], t0=t,
                               bess_soc=env.bess_soc[0], ev_soc=env.ev_soc[:, 0], exact=True,
                               terminal_bess=True, time_limit=time_limit)
            if r is not None and np.isfinite(r.cost) and r.meta.get("proven", False):
                a_t = v2g_scale_fix(ct, schedule_to_actions(ct, r.x, 0), t)
                O.append(obs[0].copy()); S.append(st[0].copy()); A.append(a_t)
                if rng.random() < beta:
                    a = a_t[None]
            else:
                fails += 1
        obs, st, _, done, info = env.step(a)
    cost = float(info["episode"]["net_cost_eur"][0])
    return np.stack(O), np.stack(S), np.stack(A), cost, fails


def dagger_collect(ct, env_cfg: EnvConfig, ckpt: str, n_days: int, seed: int,
                   every: int = 4, time_limit: float = 10.0, beta: float = 0.0,
                   workers: int = 2) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    jobs = [(ct, env_cfg, ckpt, int(rng.integers(1 << 31)), every, time_limit, beta)
            for _ in range(n_days)]
    if workers > 1:
        # "fork" is unavailable on Windows (and unsafe on macOS); fall back to "spawn".
        method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
        with mp.get_context(method).Pool(workers) as pool:
            out = pool.map(_one_day, jobs)
    else:
        out = [_one_day(j) for j in jobs]
    # Days can carry different numbers of labels (a relabel solve may find no
    # incumbent within its time limit), so each day is its own dataset.
    days = [{"obs": o[0][None].astype(np.float32), "state": o[1][None].astype(np.float32),
             "act": o[2][None].astype(np.float32)} for o in out]
    return {"days": days, "rollout_cost": np.array([o[3] for o in out]),
            "solve_failures": int(sum(o[4] for o in out))}
