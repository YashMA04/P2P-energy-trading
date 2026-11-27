"""Cooperative-game layer: budget balance, core stability, IR-guaranteed settlement."""
from __future__ import annotations

import numpy as np

from tests.test_all import community
from src.env.community_env import EnvConfig, P2PCommunityEnv
from src.game.core import coalition_value, owen_allocation, settle

_A = {}


def alloc():
    if "a" not in _A:
        ct = community()
        _A["ct"], _A["cfg"] = ct, EnvConfig()
        _A["a"] = owen_allocation(ct, _A["cfg"])
        _A["vN"] = coalition_value(ct, _A["cfg"], range(ct.n_agents))
    return _A


def test_owen_allocation_is_budget_balanced():
    a = alloc()
    assert abs(a["a"]["x"].sum() - a["vN"]) < 1e-4 * max(1.0, abs(a["vN"])), (a["a"]["x"].sum(), a["vN"])


def test_owen_allocation_is_in_the_core_on_sampled_coalitions():
    a = alloc(); ct, cfg, x = a["ct"], a["cfg"], a["a"]["x"]
    N = ct.n_agents
    rng = np.random.default_rng(0)
    coalitions = [[i] for i in range(N)]                                  # every singleton
    coalitions += [sorted(rng.choice(N, k, replace=False).tolist())
                   for k in (2, 3, 4, 6, 8, 10) for _ in range(2)]
    worst = -np.inf
    for S in coalitions:
        excess = x[S].sum() - coalition_value(ct, cfg, S)                 # must be <= 0
        worst = max(worst, excess)
    assert worst <= 1e-3, f"a coalition gains {worst:.4f} EUR by leaving"


def test_settlement_is_ir_and_budget_balanced_whenever_community_saves():
    rng = np.random.default_rng(1)
    for _ in range(200):
        g = rng.uniform(1, 15, 25)
        x = g - rng.normal(1, 2, 25)                       # some households have negative core savings
        C = g.sum() * rng.uniform(0.5, 1.0)
        s = settle(g, C, x)
        assert abs(s["budget_gap"]) < 1e-9
        assert s["ir"].all()


def test_settlement_on_a_real_rollout():
    a = alloc(); ct = a["ct"]
    from src.env.baselines import TariffAwareP2P
    env = P2PCommunityEnv(ct, EnvConfig(n_envs=1, community_shield=True, firm_minimum_only=True))
    obs, st = env.reset(seed=0, deterministic=True); done = False
    pol = TariffAwareP2P()
    while not done:
        obs, st, _, done, info = env.step(pol.act(obs, st, env))
    s = settle(env.baseline_cost_total, float(info["episode"]["net_cost_eur"][0]), a["a"]["x"])
    assert s["ir"].all() and abs(s["budget_gap"]) < 1e-9
