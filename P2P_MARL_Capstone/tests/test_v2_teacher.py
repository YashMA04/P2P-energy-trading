"""Teacher labels must be faithful: replaying them reproduces the optimum."""
from __future__ import annotations

import numpy as np

from tests.test_all import community
from src.env.community_env import EnvConfig, P2PCommunityEnv
from src.opt.teacher import generate_teacher_data

V2 = dict(community_shield=True, firm_minimum_only=True, ir_episodic=True)


def test_teacher_labels_replay_to_teacher_cost():
    ct = community()
    cfg = EnvConfig(**V2)
    d = generate_teacher_data(ct, cfg, n_days=1, seed=0, include_nominal=True, verbose=False)
    lp, sim, _, unserved, short, _ = d["day_info"][0]
    assert unserved < 1e-3 and short < 1e-3
    assert lp - 1e-6 <= sim <= lp + 2 * cfg.p2p_fee * 0.3 * 2000 * ct.dt  # fee-bounded gap
    # replay stored labels in a fresh env: must reproduce the same cost exactly
    env = P2PCommunityEnv(ct, EnvConfig(**{**V2, "n_envs": 1, "noise_std": 0.0, "residual": False}))
    obs, st = env.reset(seed=0, deterministic=True)
    for t in range(env.n_steps):
        assert np.allclose(obs[0], d["obs"][0, t], atol=1e-5), f"obs diverged at t={t}"
        obs, st, _, done, info = env.step(d["act"][0, t][None])
    assert abs(float(info["episode"]["net_cost_eur"][0]) - sim) < 1e-4


def test_soc_target_labels_replay_to_teacher_cost():
    """The SoC-target action mode, fed the teacher's own targets, reproduces it."""
    from src.opt.teacher import soc_target_labels
    ct = community()
    cfg = EnvConfig(**V2)
    d = generate_teacher_data(ct, cfg, n_days=1, seed=0, include_nominal=True, verbose=False)
    lab = soc_target_labels(ct, d)
    env = P2PCommunityEnv(ct, EnvConfig(**{**V2, "n_envs": 1, "noise_std": 0.0, "residual": False,
                                           "action_mode": "soc_target"}))
    obs, st = env.reset(seed=0, deterministic=True)
    done = False
    t = 0
    while not done:
        obs, st, _, done, info = env.step(lab[0, t][None]); t += 1
    sim = d["day_info"][0, 1]
    got = float(info["episode"]["net_cost_eur"][0])
    assert abs(got - sim) < 0.05 * abs(sim) + 0.5, (got, sim)
    assert info["episode"]["unserved_kwh"][0] < 1e-3
