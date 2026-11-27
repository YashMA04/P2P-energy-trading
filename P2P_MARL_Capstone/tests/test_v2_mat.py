"""Multi-Agent Transformer: exactness, causality, and serialisation of the order."""
from __future__ import annotations

import numpy as np
import torch

from tests.test_all import community
from src.env.community_env import EnvConfig, P2PCommunityEnv
from src.marl.mat import MAT, MATConfig, decoding_order, load_any
from src.marl.mappo import RolloutBuffer
from src.marl.trainer import collect_rollout


def env_and_obs(n=12, E=2):
    env = P2PCommunityEnv(community(n), EnvConfig(n_envs=E, community_shield=True,
                                                  firm_minimum_only=True))
    obs, st = env.reset(seed=1)
    return env, obs, st


def test_rollout_logp_equals_teacher_forced_logp():
    for order in ("laxity", "random", "index"):
        env, obs, st = env_and_obs()
        m = MAT(env.observation_dim, env.state_dim, 5, MATConfig(order=order, seed=3), n_agents=env.N)
        with torch.no_grad():
            m.net.mu.weight.normal_(0, 0.3)            # non-trivial conditioning
        a, z, lp, v, _ = m.act(obs, st)
        o = torch.as_tensor(m._prep_obs(obs)); s = torch.as_tensor(st)
        lp2, _ = m.joint_logp(o, s, torch.as_tensor(z))
        assert np.allclose(lp, lp2.detach().numpy(), atol=1e-4), order


def test_decoder_is_causal_in_the_decoding_order():
    env, obs, st = env_and_obs(E=1)
    m = MAT(env.observation_dim, env.state_dim, 5, MATConfig(order="index"), n_agents=env.N)
    with torch.no_grad():
        m.net.mu.weight.normal_(0, 0.3)
    o = torch.as_tensor(m._prep_obs(obs)); s = torch.as_tensor(st)
    a = torch.rand(1, env.N, 5)
    base = m.bc_mean(o, s, a)
    a2 = a.clone(); a2[0, 5] = -a2[0, 5]               # change household 5's action
    out = m.bc_mean(o, s, a2)
    assert torch.allclose(base[0, :6], out[0, :6], atol=1e-6), "earlier/own outputs changed"
    assert not torch.allclose(base[0, 6:], out[0, 6:], atol=1e-6), "later outputs ignore it"


def test_laxity_order_puts_most_urgent_first():
    obs = torch.zeros(1, 4, 45)
    obs[0, :, 11] = torch.tensor([0.1, 0.9, 0.0, 0.5])
    assert decoding_order(obs, "laxity", torch.arange(4)).tolist() == [[1, 3, 0, 2]]


def test_order_and_outputs_survive_save_load(tmp_path):
    env, obs, st = env_and_obs()
    m = MAT(env.observation_dim, env.state_dim, 5, MATConfig(order="random", order_seed=7), n_agents=env.N)
    m.save(tmp_path / "m.pt", extra={"residual_scale": 1.0})
    m2, ex = load_any(tmp_path / "m.pt")
    assert isinstance(m2, MAT) and m2.cfg.order == "random"
    assert torch.equal(m.fixed_perm, m2.fixed_perm)
    assert np.allclose(m.act(obs, st, True)[0], m2.act(obs, st, True)[0], atol=1e-6)


def test_mat_update_runs_and_changes_policy():
    env, obs, st = env_and_obs(E=2)
    m = MAT(env.observation_dim, env.state_dim, 5, MATConfig(ppo_epochs=1), n_agents=env.N)
    buf = RolloutBuffer(env.n_steps, env.E, env.N, env.observation_dim, env.state_dim, 5, MAT.COMPONENTS)
    last_v, _, _ = collect_rollout(env, m, buf)
    before = [p.detach().clone() for p in m.net.parameters()]
    rew = m.compose_reward({k: buf.components[k] for k in MAT.COMPONENTS}, env.cfg)
    stats = m.update(buf, rew, last_v)
    assert np.isfinite(stats["policy_loss"]) and np.isfinite(stats["value_loss"])
    assert any(not torch.equal(b, p) for b, p in zip(before, m.net.parameters()))
