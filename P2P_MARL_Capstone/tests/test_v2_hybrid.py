"""Hybrid execution: central MAT, local fallback on communication loss, fallback envelope.

The property that matters: with the fallback envelope, no outage pattern and no
fallback behaviour -- including an adversarial one that defers every car to
its deadline -- causes unserved energy or missed EV charging.  A negative
control shows the same adversary breaks the system without the envelope.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tests.test_all import community
from src.control.hybrid import (HybridController, OutageSchedule, RuleAgent,
                                MODE_CENTRAL, MODE_LOCAL, MODE_STALE)
from src.env.community_env import EnvConfig, P2PCommunityEnv
from src.marl.mat import load_any

ROOT = Path(__file__).resolve().parents[1]
V2 = dict(community_shield=True, firm_minimum_only=True, ir_episodic=True,
          residual=False, action_mode="power", n_envs=1, noise_std=0.0)


class Deferrer:
    """Adversarial fallback: commands nothing; only the shields make cars charge."""
    def act(self, obs, state, deterministic=True):
        a = np.zeros(obs.shape[:2] + (5,), np.float32)
        a[..., 4] = 0.5
        return (a,)


def run(ctrl, cap=8.0, n=25):
    env = P2PCommunityEnv(community(n), EnvConfig(**V2, offline_import_cap_kw=cap), seed=0)
    obs, st = env.reset(seed=0, deterministic=True)
    done = False
    while not done:
        obs, st, _, done, info = env.step(ctrl.act(obs, st, env))
    return info["episode"], env


def students():
    R = ROOT / "results/v2"
    it_mat = json.load(open(R / "dagger_mat/seed0/summary.json"))["selected_iter_on_val"]
    it_map = json.load(open(R / "dagger_mappo/seed0/summary.json"))["selected_iter_on_val"]
    return load_any(R / f"dagger_mat/seed0/iter{it_mat}.pt")[0], \
        load_any(R / f"dagger_mappo/seed0/iter{it_map}.pt")[0]


def test_outage_schedule_windows():
    s = OutageSchedule.window(25, 68, 84)
    assert s.online(67).all() and not s.online(68).any() and not s.online(83).any() and s.online(84).all()
    p = OutageSchedule.random_households(25, 0.3, seed=0)
    assert (~p.online(10)).sum() == 8
    assert not OutageSchedule.full(25).online(50).any() and OutageSchedule.none(25).online(50).all()


@pytest.mark.parametrize("sched", [OutageSchedule.full(25), OutageSchedule.window(25, 68, 84),
                                   OutageSchedule.random_households(25, 0.3, seed=1),
                                   OutageSchedule.window(25, 0, 96, households=range(0, 25, 2))])
def test_envelope_keeps_the_feeder_safe_under_an_adversarial_fallback(sched):
    ep, _ = run(HybridController(RuleAgent(), Deferrer(), sched, "local"), cap=8.0)
    assert ep["unserved_kwh"][0] < 1e-3, ep["unserved_kwh"]
    assert ep["ev_shortfall_kwh"][0] < 1e-3


def test_negative_control_without_envelope_the_adversary_breaks_the_feeder():
    ep, _ = run(HybridController(RuleAgent(), Deferrer(), OutageSchedule.full(25), "local"), cap=None)
    assert ep["unserved_kwh"][0] > 10.0


def test_no_outage_hybrid_is_exactly_the_central_policy():
    mat, mappo = students()
    a, _ = run(HybridController(mat, mappo, OutageSchedule.none(25), "local"))
    b, _ = run(HybridController(mat, None, None, "rule"))
    assert abs(a["net_cost_eur"][0] - b["net_cost_eur"][0]) < 1e-6


def test_full_outage_runs_every_household_on_its_local_fallback():
    mat, mappo = students()
    h = HybridController(mat, mappo, OutageSchedule.full(25), "local")
    ep, _ = run(h)
    modes = np.array(h.modes)
    assert (modes == MODE_LOCAL).all()
    ref, _ = run(HybridController(mappo, mappo, OutageSchedule.full(25), "local"))
    assert abs(ep["net_cost_eur"][0] - ref["net_cost_eur"][0]) < 1e-6   # MAT plays no part
    assert ep["unserved_kwh"][0] < 1e-3


def test_stale_mode_holds_the_last_received_setpoint():
    mat, _ = students()
    sched = OutageSchedule.window(25, 40, 50, households=[3])
    h = HybridController(mat, None, sched, "stale")
    env = P2PCommunityEnv(community(25), EnvConfig(**V2, offline_import_cap_kw=8.0), seed=0)
    obs, st = env.reset(seed=0, deterministic=True)
    acts = []
    for t in range(52):
        a = h.act(obs, st, env); acts.append(a[0, 3].copy())
        obs, st, _, _, _ = env.step(a)
    for t in range(40, 50):
        assert np.allclose(acts[t], acts[39])
        assert h.modes[t][3] == MODE_STALE and h.modes[t][0] == MODE_CENTRAL


def test_outage_state_never_leaks_into_the_next_episode():
    env = P2PCommunityEnv(community(25), EnvConfig(**V2), seed=0)
    env.comms_online = np.zeros((1, 25), bool)
    env.reset(seed=0)
    assert env.comms_online is None
