"""Tests for the v2 optimisation layer (exact dispatch MILP, replay, accounting).

Every number the v2 study reports against "the optimum" relies on these.
The key invariant: the optimiser's schedule, replayed through the simulator,
is settled at (almost) the optimiser's own cost -- so the bound is attainable,
not an artefact of a softer model.
"""
from __future__ import annotations

import numpy as np

from tests.test_all import community, rollout
from src.env.baselines import BASELINES
from src.env.community_env import EnvConfig
from src.opt.lp import solve_dispatch
from src.opt.replay import replay_in_simulator

_LP = {}


def exact(term=True):
    if term not in _LP:
        _LP[term] = solve_dispatch(community(), EnvConfig(), exact=True, terminal_bess=term)
    return _LP[term]


def test_exact_milp_solves_and_has_no_grid_to_peer_arbitrage():
    r = exact()
    assert np.isfinite(r.cost)
    x = r.x
    assert np.minimum(x["imp"], x["pout"]).max() < 1e-6, "household imports and sells in one step"
    assert np.minimum(x["exp"], x["pin"]).max() < 1e-6, "household exports and buys in one step"


def test_exact_schedule_is_attainable_in_the_simulator():
    """Replay gap must be small and explained by the unmodelled P2P fee."""
    r = exact()
    ep = replay_in_simulator(community(), EnvConfig(), r.x)
    fee_upper = 2 * EnvConfig().p2p_fee * 0.30 * ep["p2p_volume_kwh"]   # price <= EUR 0.30
    assert ep["unserved_kwh"] < 1e-3
    assert ep["ev_shortfall_kwh"] < 1e-3
    assert r.cost - 1e-6 <= ep["net_cost_eur"] <= r.cost + fee_upper + 0.05, (r.cost, ep["net_cost_eur"])


def test_terminal_condition_is_respected_and_costs_more():
    with_t, without = exact(True), exact(False)
    assert with_t.meta["bess_end_minus_start_kwh"] > -1e-6
    assert with_t.cost >= without.cost - 1e-6


def test_exact_bound_is_below_every_baseline_inventory_neutral():
    lp = exact(True).cost
    for name, cls in BASELINES.items():
        _, info = rollout(cls())
        c = float(np.mean(info["episode"]["net_cost_inv_neutral_eur"]))
        assert lp <= c + 1e-6, f"{name}: {c:.2f} below the bound {lp:.2f}"


def test_relaxed_lp_is_below_exact():
    """The convex relaxation must bound the exact problem from below."""
    relaxed = solve_dispatch(community(), EnvConfig(), exact=False, terminal_bess=True)
    assert relaxed.cost <= exact(True).cost + 1e-6


def test_inventory_neutral_metric_is_zero_adjustment_without_storage_use():
    env, info = rollout(BASELINES["no_flexibility"]())
    ep = info["episode"]
    assert np.allclose(ep["net_cost_inv_neutral_eur"], ep["net_cost_eur"])
    assert np.allclose(ep["bess_drawdown_kwh"], 0.0)


def test_time_limited_incumbent_is_not_marked_proven():
    """Regression: a time-limited MILP incumbent must never be used as optimal."""
    r = solve_dispatch(community(), EnvConfig(), exact=True, terminal_bess=True, time_limit=0.5)
    assert r.meta["proven"] is False or r.meta["mip_gap"] <= 1e-3
    assert exact(True).meta["proven"]
