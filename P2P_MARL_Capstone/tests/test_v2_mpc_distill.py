"""MPC baseline and distillation plumbing."""
from __future__ import annotations

import numpy as np

from tests.test_all import community
from src.env.community_env import EnvConfig, P2PCommunityEnv
from src.marl.distill import BCRegulariser, fit_student
from src.marl.mappo import MAPPO, MAPPOConfig
from src.opt.lp import solve_dispatch
from src.opt.mpc import MPCPolicy
from src.opt.teacher import generate_teacher_data

V2 = dict(community_shield=True, firm_minimum_only=True, ir_episodic=True)


def test_mpc_is_safe_and_above_the_bound():
    ct = community()
    cfg = EnvConfig(**V2)
    pol = MPCPolicy(ct, cfg, horizon=4, time_limit=10.0)
    env = P2PCommunityEnv(ct, EnvConfig(**{**V2, "n_envs": 1, "noise_std": 0.0, "residual": False}))
    obs, st = env.reset(seed=0, deterministic=True)
    done = False
    while not done:
        obs, st, _, done, info = env.step(pol.act(obs, st, env))
    ep = info["episode"]
    bound = solve_dispatch(ct, cfg, exact=True, terminal_bess=True).cost
    assert pol.solve_failures == 0
    assert ep["unserved_kwh"][0] < 1e-3 and ep["ev_shortfall_kwh"][0] < 1e-3
    assert ep["net_cost_inv_neutral_eur"][0] >= bound - 1e-6


def test_student_fits_teacher_and_bc_hook_is_finite():
    ct = community()
    d = generate_teacher_data(ct, EnvConfig(**V2), n_days=2, seed=1, verbose=False)
    ag = MAPPO(d["obs"].shape[-1], d["state"].shape[-1], 5, MAPPOConfig())
    hist = fit_student(ag, d, epochs=15, batch=128, verbose=False)
    assert hist["train"][-1] < 0.5 * hist["train"][0]
    bc = BCRegulariser(d, batch=256)
    assert np.isfinite(bc.loss(ag).item())
    assert bc.coef(0.0) == 1.0 and abs(bc.coef(1.0) - 0.05) < 1e-9


def test_fine_tuning_hooks_are_inert_by_default():
    ag = MAPPO(45, 20, 5, MAPPOConfig())
    assert ag.bc is None and ag.freeze_actor is False


def test_gate_reference_is_the_rule_controller_even_for_absolute_learners():
    """Regression: an absolute-action learner must not weaken the gate."""
    from dataclasses import replace
    from src.marl.trainer import Trainer, TrainConfig
    ct = community()
    a = Trainer(ct, EnvConfig(**V2), MAPPOConfig(), TrainConfig(updates=1, eval_episodes=1),
                out_dir="/tmp/_t_ref_a")
    b = Trainer(ct, EnvConfig(**{**V2, "residual": False}), MAPPOConfig(),
                TrainConfig(updates=1, eval_episodes=1), out_dir="/tmp/_t_ref_b")
    ra, rb = a.reference_baseline(), b.reference_baseline()
    assert abs(ra["net_cost_eur"][0] - rb["net_cost_eur"][0]) < 1e-6
