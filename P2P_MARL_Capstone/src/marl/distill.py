"""Student distillation from the privileged teacher, and the hook for
behaviour-cloning-regularised RL fine-tuning.

Stage 1 (supervised):  fit the shared actor's *squashed mean* to the teacher's
    absolute actions, on observation-normalised data.
Stage 2 (RL):  PPO on the true reward, with an auxiliary BC loss on teacher
    data whose weight is annealed to a small floor.  This plays the role the
    residual trust region played in v1 -- keeping early exploration near a
    known-good policy -- but anchored to the optimum's behaviour rather than to
    the rule controller, and on the *loss* (annealed away) rather than as a
    hard cap.  Cf. DAPG (Rajeswaran et al., 2018).

The critic is warmed up on-policy before the actor is allowed to move, so the
first PPO steps are not taken against an untrained value function -- the most
common way a good initial policy is destroyed in fine-tuning.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
from torch import nn

from .mappo import MAPPO
from .networks import Actor


def _bc_pred(agent, x_rows, s_rows, a_rows, N):
    """Squashed-mean prediction for BC: per-row (MAPPO) or teacher-forced (MAT)."""
    if hasattr(agent, "bc_mean"):
        B = x_rows.shape[0] // N
        return agent.bc_mean(x_rows.reshape(B, N, -1), s_rows,
                             a_rows.reshape(B, N, -1)).reshape(B * N, -1)
    mu, _ = agent.actor(x_rows)
    return Actor.squash(mu)


def fit_student(agent: MAPPO, data: Dict[str, np.ndarray], epochs: int = 30,
                lr: float = 1e-3, batch: int = 4096, val_frac: float = 0.1,
                log_std_after: float = -1.6, seed: int = 0, verbose: bool = True) -> Dict[str, list]:
    """Behaviour-clone the actor on teacher data. Returns train/val loss curves."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    sets = data if isinstance(data, list) else [data]
    N, O = sets[0]["obs"].shape[2], sets[0]["obs"].shape[3]
    Xs, Ss, Ys, G, W = [], [], [], [], []
    gid = 0
    for d in sets:
        D, T = d["obs"].shape[:2]
        Xs.append(d["obs"].reshape(D * T, N, O)); Ss.append(d["state"].reshape(D * T, -1))
        Ys.append(d["act"].reshape(D * T, N, -1))
        G.append(gid + np.repeat(np.arange(D), T)); gid += D
        # each dataset contributes total weight d.get("weight", 1) regardless of its size
        W.append(np.full(D * T, float(d.get("weight", 1.0)) / (D * T)))
    obs = np.concatenate(Xs).reshape(-1, O)
    if agent.obs_rms is not None:
        agent.obs_rms.update(obs)
    X = torch.as_tensor(agent._prep_obs(obs)).reshape(-1, N, O)
    S = torch.as_tensor(np.concatenate(Ss))
    Y = torch.as_tensor(np.concatenate(Ys))
    G = np.concatenate(G); Wt = np.concatenate(W); Wt = Wt / Wt.sum()
    # hold out whole days (states within a day are correlated)
    days = rng.permutation(gid)
    n_val = max(1, int(round(val_frac * gid))) if gid > 1 else 0
    is_val = np.isin(G, days[:n_val])
    tr_idx, va_idx = np.flatnonzero(~is_val), np.flatnonzero(is_val)
    p_tr = Wt[tr_idx] / Wt[tr_idx].sum()
    sb = max(1, batch // N)                                               # states per batch
    opt = torch.optim.Adam(agent.actor.parameters(), lr=lr)
    hist = {"train": [], "val": []}

    def pred(ix):
        return _bc_pred(agent, X[ix].reshape(-1, O), S[ix], Y[ix].reshape(-1, Y.shape[-1]), N)

    for ep in range(epochs):
        perm = rng.choice(tr_idx, size=len(tr_idx), replace=True, p=p_tr)   # weighted epoch
        losses = []
        for s0 in range(0, len(perm), sb):
            ix = torch.as_tensor(perm[s0:s0 + sb])
            loss = (pred(ix) - Y[ix].reshape(-1, Y.shape[-1])).pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        hist["train"].append(float(np.mean(losses)))
        if len(va_idx):
            with torch.no_grad():
                ix = torch.as_tensor(va_idx)
                hist["val"].append(float((pred(ix) - Y[ix].reshape(-1, Y.shape[-1])).pow(2).mean()))
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            print(f"  BC epoch {ep:3d} train {hist['train'][-1]:.4f} val {hist['val'][-1] if hist['val'] else float('nan'):.4f}", flush=True)
    with torch.no_grad():
        agent.actor.log_std.fill_(log_std_after)
    return hist


class BCRegulariser:
    """Auxiliary BC loss on teacher data for PPO fine-tuning (annealed)."""

    def __init__(self, data: Dict[str, np.ndarray], coef_start: float = 1.0,
                 coef_final: float = 0.05, batch: int = 2048, seed: int = 0):
        sets = data if isinstance(data, list) else [data]
        N, O = sets[0]["obs"].shape[2], sets[0]["obs"].shape[3]
        self.N, self.O = N, O
        self.obs = np.concatenate([d["obs"].reshape(-1, N, O) for d in sets])
        self.state = torch.as_tensor(np.concatenate([d["state"].reshape(-1, d["state"].shape[-1]) for d in sets]))
        self.act = torch.as_tensor(np.concatenate([d["act"].reshape(-1, N, d["act"].shape[-1]) for d in sets]))
        self.coef_start, self.coef_final = float(coef_start), float(coef_final)
        self.batch = int(batch)
        self.rng = np.random.default_rng(seed)

    def coef(self, progress: float) -> float:
        p = min(max(progress, 0.0), 1.0)
        return self.coef_start + (self.coef_final - self.coef_start) * p

    def loss(self, agent: MAPPO) -> torch.Tensor:
        sel = self.rng.integers(0, len(self.obs), max(1, self.batch // self.N))
        x = torch.as_tensor(agent._prep_obs(self.obs[sel].reshape(-1, self.O)))
        a = self.act[sel].reshape(-1, self.act.shape[-1])
        return (_bc_pred(agent, x, self.state[sel], a, self.N) - a).pow(2).mean()


class AnchorRegulariser:
    """Keep an RL-fine-tuned policy close to a frozen anchor policy (annealed).

    Targets are the *anchor's own* deterministic actions on a bank of
    community states, so the regulariser pulls toward the DAgger student, not
    toward raw teacher labels (which the pure BC student showed are unusable
    in closed loop).  Analogous to a KL-to-reference term.
    """

    def __init__(self, anchor, obs_bank: np.ndarray, state_bank: np.ndarray,
                 coef_start: float = 1.0, coef_final: float = 0.1, batch: int = 2048, seed: int = 0):
        self.N, self.O = obs_bank.shape[1], obs_bank.shape[2]
        self.obs = obs_bank
        self.state = torch.as_tensor(state_bank)
        acts = []
        for s in range(0, len(obs_bank), 256):
            acts.append(anchor.act(obs_bank[s:s + 256], state_bank[s:s + 256], True)[0])
        self.act = torch.as_tensor(np.concatenate(acts))
        self.coef_start, self.coef_final = float(coef_start), float(coef_final)
        self.batch = int(batch)
        self.rng = np.random.default_rng(seed)

    coef = BCRegulariser.coef
    loss = BCRegulariser.loss
