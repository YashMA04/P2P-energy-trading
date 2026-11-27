"""Tests for the v2 community feasibility shield.

The property that matters is stated directly: with the v2 shield on, *no*
action sequence -- adversarial deferral, uniformly random actions -- produces
unserved energy or EV shortfall on the deterministic day or on noisy days.
A negative control shows the v1 shield does fail on the same adversary, so
the test can tell the two apart.
"""
from __future__ import annotations

import numpy as np

from tests.test_all import community
from src.env.baselines import DeadlineCharging, NoFlexibility, TariffAwareP2P
from src.env.community_env import ACTION_DIM, EnvConfig, P2PCommunityEnv

V2 = dict(community_shield=True, firm_minimum_only=True)


class Uniform:
    residual = False

    def __init__(self, seed):
        self.r = np.random.default_rng(seed)

    def act(self, obs, state, env=None):
        a = self.r.uniform(-1, 1, obs.shape[:2] + (ACTION_DIM,)).astype(np.float32)
        a[..., 4] = self.r.uniform(0, 1, obs.shape[:2])
        return a


def run(policy, n=25, E=2, noise=0.0, seed=5, **flags):
    env = P2PCommunityEnv(community(n), EnvConfig(n_envs=E, noise_std=noise,
                                                  residual=policy.residual, **flags), seed=seed)
    obs, st = env.reset(seed=seed, deterministic=noise == 0.0)
    done = False
    while not done:
        obs, st, _, done, info = env.step(policy.act(obs, st, env))
    return env, info["episode"]


def test_negative_control_v1_shield_fails_under_deferral():
    _, ep = run(DeadlineCharging())
    assert ep["unserved_kwh"].max() > 10.0, "adversary too weak to test anything"


def test_v2_shield_survives_deferral_deterministic_and_noisy():
    for noise in (0.0, 0.02):
        _, ep = run(DeadlineCharging(), noise=noise, E=4, **V2)
        assert ep["unserved_kwh"].max() < 1e-3, ep["unserved_kwh"]
        assert ep["ev_shortfall_kwh"].max() < 1e-3


def test_v2_shield_survives_uniformly_random_actions():
    for seed in range(4):
        for noise in (0.0, 0.02):
            _, ep = run(Uniform(seed), noise=noise, E=2, seed=seed, **V2)
            assert ep["unserved_kwh"].max() < 1e-3
            assert ep["ev_shortfall_kwh"].max() < 1e-3


def test_v2_shield_on_smaller_community():
    _, ep = run(DeadlineCharging(), n=12, **V2)
    assert ep["unserved_kwh"].max() < 1e-3


def test_counterfactual_is_independent_of_v2_flags():
    a = P2PCommunityEnv(community(), EnvConfig(n_envs=1))
    b = P2PCommunityEnv(community(), EnvConfig(n_envs=1, **V2))
    assert np.allclose(a.baseline_cost_step, b.baseline_cost_step)


def test_no_flexibility_still_reproduces_counterfactual_under_v2():
    env, ep = run(NoFlexibility(), E=1, **V2)
    assert abs(ep["net_cost_eur"][0] - env.baseline_cost_total.sum()) < 1e-3


def test_v2_heuristic_no_worse_than_v1_heuristic():
    _, a = run(TariffAwareP2P(), E=1)
    _, b = run(TariffAwareP2P(), E=1, **V2)
    assert b["unserved_kwh"][0] < 1e-3
    assert b["net_cost_eur"][0] <= a["net_cost_eur"][0] + 1e-6
