"""Multi-Agent Transformer (Wen et al., NeurIPS 2022) with a domain decoding order.

Joint policy factorised autoregressively over households::

    pi(a | o) = prod_k pi(a_{s_k} | o_1..o_N, a_{s_1}..a_{s_{k-1}})

where ``s`` is the decoding order.  An encoder attends over all household
observations; a causal decoder generates each household's action conditioned
on the encoded community and on the actions already chosen by households
earlier in the order.  Training uses teacher forcing on the stored actions,
so the update is one parallel pass; only the rollout decodes sequentially.

Decoding order (``order``), a policy quantity serialised with the checkpoint:
    "laxity"  most urgent first: descending max EV urgency (obs 11/16), then
              descending |net load|.  Urgent cars commit first, flexible
              households adapt around them -- least-laxity-first scheduling.
    "random"  one fixed random permutation drawn at construction (stored).
    "index"   household index order.

Deviations from the paper, each for a reason measured in this project:
* separate critic encoder and optimiser: MAT shares the encoder between policy
  and value, which re-couples actor and critic gradients -- the v1 defect that
  produced an effective actor learning rate of 5.9e-7;
* per-household values and advantages (credit assignment), as in v2 MAPPO;
* Gaussian over the pre-squash variable with the stored-z log-prob, as in v2.
Inference is centralised (all observations, preceding actions): this is a
community-operator-executed policy, not a decentralised one.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from .mappo import MAPPO, MAPPOConfig
from .networks import Actor, AttentionCritic, N_TANH

URG1, URG2, NET = 11, 16, 2          # observation indices used by the "laxity" order


def decoding_order(obs: torch.Tensor, mode: str, fixed_perm: torch.Tensor) -> torch.Tensor:
    """(B, N) permutation: position k -> household index."""
    B, N, _ = obs.shape
    if mode == "laxity":
        urg = torch.maximum(obs[..., URG1], obs[..., URG2])
        key = urg * 10.0 + obs[..., NET].abs().clamp(max=9.0)   # urgency dominates
        return torch.argsort(-key, dim=1, stable=True)
    if mode == "random":
        return fixed_perm.expand(B, N)
    return torch.arange(N, device=obs.device).expand(B, N)


class MATNet(nn.Module):
    def __init__(self, obs_dim: int, state_dim: int, action_dim: int = 5, d: int = 64,
                 heads: int = 4, layers: int = 1, log_std_init: float = -1.0):
        super().__init__()
        self.action_dim = action_dim
        self.obs_emb = nn.Sequential(nn.Linear(obs_dim, d), nn.LayerNorm(d), nn.GELU())
        self.state_emb = nn.Linear(state_dim, d)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True), layers)
        self.act_emb = nn.Linear(action_dim + 1, d)          # +1: start-token flag
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True), layers)
        self.mu = nn.Linear(2 * d, action_dim)
        nn.init.orthogonal_(self.mu.weight, 0.01); nn.init.zeros_(self.mu.bias)
        self.log_std = nn.Parameter(torch.full((action_dim,), float(log_std_init)))

    def encode(self, obs_o: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """obs_o (B, N, O) already in decoding order."""
        h = self.obs_emb(obs_o) + self.state_emb(state).unsqueeze(1)
        return self.encoder(h)

    def decode(self, enc: torch.Tensor, prev_act: torch.Tensor) -> torch.Tensor:
        """prev_act (B, K, A+1) shifted actions; returns mu for positions 0..K-1."""
        K = prev_act.shape[1]
        mask = torch.triu(torch.full((K, K), float("-inf"), device=enc.device), 1)
        h = self.decoder(self.act_emb(prev_act), enc, tgt_mask=mask)
        return self.mu(torch.cat([h, enc[:, :K]], -1))

    @staticmethod
    def shifted(actions_o: torch.Tensor) -> torch.Tensor:
        """Teacher-forcing input: start token, then a_0..a_{N-2} (in order)."""
        B, N, A = actions_o.shape
        start = torch.zeros(B, 1, A + 1, device=actions_o.device); start[..., -1] = 1.0
        body = torch.cat([actions_o[:, :-1], torch.zeros(B, N - 1, 1, device=actions_o.device)], -1)
        return torch.cat([start, body], 1)


@dataclass
class MATConfig(MAPPOConfig):
    d_model: int = 64
    mat_heads: int = 4
    mat_layers: int = 1
    order: str = "laxity"
    order_seed: int = 0


class MAT(MAPPO):
    """Drop-in replacement for :class:`MAPPO` inside :class:`Trainer`."""

    def __init__(self, obs_dim: int, state_dim: int, action_dim: int,
                 cfg: Optional[MATConfig] = None, n_agents: int = 25):
        cfg = cfg or MATConfig()
        super().__init__(obs_dim, state_dim, action_dim, cfg)    # builds normalisers, duals, critic
        c = self.cfg
        torch.manual_seed(c.seed)
        self.n_agents = n_agents
        self.net = MATNet(obs_dim, state_dim, action_dim, c.d_model, c.mat_heads,
                          c.mat_layers, c.log_std_init).to(self.device)
        g = torch.Generator().manual_seed(c.order_seed)
        self.register_perm(torch.randperm(n_agents, generator=g))
        self.actor = self.net                                   # for save()/BC plumbing
        self.opt_actor = torch.optim.Adam(self.net.parameters(), lr=c.learning_rate_actor, eps=1e-5)

    def register_perm(self, perm: torch.Tensor):
        self.fixed_perm = perm.to(self.device)

    # ------------------------------------------------------------------
    def _dist(self, mu: torch.Tensor) -> Normal:
        return Normal(mu, self.net.log_std.clamp(-5.0, 1.0).exp().expand_as(mu))

    def _ordered(self, o: torch.Tensor):
        perm = decoding_order(o, self.cfg.order, self.fixed_perm)        # (B, N)
        idx = perm.unsqueeze(-1)
        return perm, torch.gather(o, 1, idx.expand(-1, -1, o.shape[-1]))

    @torch.no_grad()
    def act(self, obs: np.ndarray, state: np.ndarray, deterministic: bool = False):
        o = torch.as_tensor(self._prep_obs(obs), device=self.device)
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        E, N = o.shape[:2]
        perm, o_o = self._ordered(o)
        enc = self.net.encode(o_o, s)
        A = self.action_dim
        prev = torch.zeros(E, N, A + 1, device=self.device); prev[:, 0, -1] = 1.0
        z_o = torch.zeros(E, N, A, device=self.device)
        for k in range(N):
            mu = self.net.decode(enc, prev[:, :k + 1])[:, k]
            dist = self._dist(mu)
            z = mu if deterministic else dist.rsample()
            z_o[:, k] = z
            if k + 1 < N:
                prev[:, k + 1, :A] = Actor.squash(z)
        logp_o = Actor.log_prob(self._dist(self.net.decode(enc, prev)), z_o)   # (E, N)
        inv = torch.argsort(perm, dim=1)
        z = torch.gather(z_o, 1, inv.unsqueeze(-1).expand(-1, -1, A))
        logp = torch.gather(logp_o, 1, inv)
        value = self.critic(o, s)
        if self.value_norm is not None:
            value = self.value_norm.denormalize(value)
        return (Actor.squash(z).cpu().numpy(), z.cpu().numpy(), logp.cpu().numpy(),
                value.cpu().numpy(), torch.zeros_like(value).cpu().numpy())

    def joint_logp(self, o: torch.Tensor, s: torch.Tensor, z: torch.Tensor):
        """Teacher-forced per-household log-prob and entropy, household order."""
        perm, o_o = self._ordered(o)
        A = self.action_dim
        z_o = torch.gather(z, 1, perm.unsqueeze(-1).expand(-1, -1, A))
        enc = self.net.encode(o_o, s)
        mu = self.net.decode(enc, MATNet.shifted(Actor.squash(z_o)))
        dist = self._dist(mu)
        lp_o = Actor.log_prob(dist, z_o)
        inv = torch.argsort(perm, dim=1)
        return torch.gather(lp_o, 1, inv), dist.entropy().sum(-1).mean()

    def bc_mean(self, o: torch.Tensor, s: torch.Tensor, a_teacher: torch.Tensor) -> torch.Tensor:
        """Teacher-forced squashed means for BC, household order."""
        perm, o_o = self._ordered(o)
        A = self.action_dim
        a_o = torch.gather(a_teacher, 1, perm.unsqueeze(-1).expand(-1, -1, A))
        mu = self.net.decode(self.net.encode(o_o, s), MATNet.shifted(a_o))
        inv = torch.argsort(perm, dim=1)
        return torch.gather(Actor.squash(mu), 1, inv.unsqueeze(-1).expand(-1, -1, A))

    # ------------------------------------------------------------------
    def update(self, buf, rewards, last_value, cost_returns=None) -> Dict[str, float]:
        c = self.cfg
        T, E, N = buf.T, buf.E, buf.N
        adv, ret = self.compute_gae(rewards, buf.value, last_value)
        if self.obs_rms is not None:
            self.obs_rms.update(buf.obs.reshape(-1, self.obs_dim))
        dev, B = self.device, T * E
        obs_t = torch.as_tensor(self._prep_obs(buf.obs).reshape(B, N, -1), device=dev)
        st_t = torch.as_tensor(buf.state.reshape(B, -1), device=dev)
        z_t = torch.as_tensor(buf.z.reshape(B, N, -1), device=dev)
        lp_old = torch.as_tensor(buf.logp.reshape(B, N), device=dev)
        v_old = torch.as_tensor(buf.value.reshape(B, N), device=dev)
        adv_t = torch.as_tensor(adv.reshape(B, N), device=dev)
        ret_t = torch.as_tensor(ret.reshape(B, N), device=dev)
        if self.value_norm is not None:
            self.value_norm.update(ret_t)
            ret_tg, v_old_n = self.value_norm.normalize(ret_t), self.value_norm.normalize(v_old)
        else:
            ret_tg, v_old_n = ret_t, v_old
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        frac = max(0.0, 1.0 - self.progress) if c.anneal_lr else 1.0
        for g_ in self.opt_actor.param_groups: g_["lr"] = c.learning_rate_actor * frac
        for g_ in self.opt_critic.param_groups: g_["lr"] = c.learning_rate_critic * frac
        ent_coef = c.entropy_coef + (c.entropy_coef_final - c.entropy_coef) * min(1.0, self.progress)
        mb = max(1, B // c.num_minibatches)
        idx = np.arange(B)
        st = {k: [] for k in ("policy_loss", "value_loss", "entropy", "kl", "clip_frac",
                              "grad_actor", "grad_critic")}
        stop = False
        for epoch in range(c.ppo_epochs):
            np.random.shuffle(idx)
            for s0 in range(0, B, mb):
                sel = torch.as_tensor(idx[s0:s0 + mb], device=dev)
                lp, ent = self.joint_logp(obs_t[sel], st_t[sel], z_t[sel])
                ratio = (lp - lp_old[sel]).exp()
                a = adv_t[sel]
                pl = -torch.min(ratio * a, ratio.clamp(1 - c.clip_range, 1 + c.clip_range) * a).mean()
                loss = pl - ent_coef * ent
                if self.bc is not None:
                    loss = loss + self.bc.coef(self.progress) * self.bc.loss(self)
                self.opt_actor.zero_grad(set_to_none=True); loss.backward()
                ga = nn.utils.clip_grad_norm_(self.net.parameters(), c.max_grad_norm_actor)
                if not self.freeze_actor:
                    self.opt_actor.step()
                v = self.critic(obs_t[sel], st_t[sel])
                vc = v_old_n[sel] + (v - v_old_n[sel]).clamp(-c.value_clip_range, c.value_clip_range)
                vl = torch.max((v - ret_tg[sel]).pow(2), (vc - ret_tg[sel]).pow(2)).mean()
                self.opt_critic.zero_grad(set_to_none=True); vl.backward()
                gc = nn.utils.clip_grad_norm_(self.critic.parameters(), c.max_grad_norm_critic)
                self.opt_critic.step()
                with torch.no_grad():
                    lr_ = lp - lp_old[sel]
                    kl = ((lr_.exp() - 1) - lr_).mean().item()
                st["policy_loss"].append(pl.item()); st["value_loss"].append(vl.item())
                st["entropy"].append(ent.item()); st["kl"].append(kl)
                st["clip_frac"].append(((ratio - 1).abs() > c.clip_range).float().mean().item())
                st["grad_actor"].append(float(ga)); st["grad_critic"].append(float(gc))
                if not self.freeze_actor and c.target_kl > 0 and kl > 1.5 * c.target_kl:
                    stop = True; break
            if stop:
                break
        out = {k: float(np.mean(v)) if v else 0.0 for k, v in st.items()}
        out.update({"epochs_run": epoch + 1, "early_stop": float(stop), "mean_return": float(ret.mean()),
                    "mean_adv_abs": float(np.abs(adv).mean()),
                    "log_std": float(self.net.log_std.mean().item()), "entropy_coef": ent_coef})
        return out

    # ------------------------------------------------------------------
    def save(self, path, extra: Optional[dict] = None):
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"kind": "mat", "net": self.net.state_dict(), "critic": self.critic.state_dict(),
                    "fixed_perm": self.fixed_perm.cpu(),
                    "obs_rms": self.obs_rms.state_dict() if self.obs_rms else None,
                    "value_norm": self.value_norm.state_dict() if self.value_norm else None,
                    "constraints": self.constraints.state_dict(), "cfg": asdict(self.cfg),
                    "dims": (self.obs_dim, self.state_dim, self.action_dim),
                    "n_agents": self.n_agents, "extra": extra or {}}, p)

    @classmethod
    def load(cls, path, device: str = "cpu"):
        d = torch.load(path, map_location=device, weights_only=False)
        cfg = MATConfig(**{**d["cfg"], "device": device})
        m = cls(*d["dims"], cfg, n_agents=d["n_agents"])
        m.net.load_state_dict(d["net"]); m.critic.load_state_dict(d["critic"])
        m.register_perm(d["fixed_perm"])          # decoding order is part of the policy
        if d.get("obs_rms") and m.obs_rms: m.obs_rms.load_state_dict(d["obs_rms"])
        if d.get("value_norm") and m.value_norm: m.value_norm.load_state_dict(d["value_norm"])
        m.constraints.load_state_dict(d.get("constraints", {}))
        return m, d.get("extra", {})


def load_any(path, device: str = "cpu"):
    """Load a MAPPO or MAT checkpoint."""
    d = torch.load(path, map_location=device, weights_only=False)
    return (MAT if d.get("kind") == "mat" else MAPPO).load(path, device)
