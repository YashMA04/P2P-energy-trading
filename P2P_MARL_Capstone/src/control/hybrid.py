"""Hybrid execution with fallback: central MAT when reachable, local policy when not.

Deployment architecture (see docs/V2_HYBRID.md):

    operator (central)                       household (local)
    ------------------                       -----------------
    MAT over the whole community   --->      apply MAT setpoint            [online]
    outage-aware community shield            ... or, if the link is down:
                                             local policy on own data      [offline]
                                             (MAPPO student, else rule controller)
                                             + its own per-vehicle shield (always)

* **Everyone reachable**: MAT decides for the whole community.
* **Anyone cut off**: every household runs its local fallback (MAPPO student)
  on its *own* observation -- measured on validation days, this beats keeping
  MAT for the reachable households on a stale picture of the others
  (``scripts/select_degrade.py``).  Reachable households still get the
  operator's community shield.
* **Fallback envelope**: a cut-off household caps its own import at
  ``offline_import_cap_kw`` and the operator reserves that much feeder capacity
  for it, so the two can never collide (CSIP-AUS-style default limit).

``OutageSchedule`` describes who is offline when; ``HybridController`` is a
policy with the same ``act(obs, state, env)`` interface as the baselines.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

MODE_CENTRAL, MODE_LOCAL, MODE_RULE, MODE_STALE = 0, 1, 2, 3

# Chosen on validation days by scripts/select_degrade.py (results/v2/hybrid/degrade_selection.txt):
# switching the whole community to local control won at every share tested,
# down to a single unreachable household (1/25 = 0.04).
DEFAULT_DEGRADE_FRAC = 0.04
MODE_NAMES = {MODE_CENTRAL: "central (MAT)", MODE_LOCAL: "local fallback (MAPPO student)",
              MODE_RULE: "local fallback (rule controller)", MODE_STALE: "no fallback (holds last setpoint)"}


@dataclass
class OutageSchedule:
    """Which households the operator can reach at each step: ``online(t) -> (N,) bool``."""
    n_agents: int
    name: str = "no outage"
    windows: List[tuple] = field(default_factory=list)      # (t_start, t_end, households or None)

    def online(self, t: int) -> np.ndarray:
        on = np.ones(self.n_agents, bool)
        for t0, t1, hh in self.windows:
            if t0 <= t < t1:
                if hh is None:
                    on[:] = False
                else:
                    on[list(hh)] = False
        return on

    # -------- constructors --------
    @classmethod
    def none(cls, n):
        return cls(n, "no outage")

    @classmethod
    def window(cls, n, t0, t1, households=None, name=None):
        return cls(n, name or f"outage {t0 / 4:04.1f}h-{t1 / 4:04.1f}h", [(t0, t1, households)])

    @classmethod
    def random_households(cls, n, frac, seed=0, t0=0, t1=96, name=None):
        k = max(1, int(round(frac * n)))
        hh = np.random.default_rng(seed).choice(n, k, replace=False).tolist()
        return cls(n, name or f"{k} households offline all day", [(t0, t1, hh)])

    @classmethod
    def full(cls, n):
        return cls(n, "operator down all day", [(0, 96, None)])


class HybridController:
    """Central policy for reachable households, local fallback for the rest.

    central : a MAT (or any) agent with ``act(obs, state, deterministic)``
    local   : decentralised agent acting on each household's own obs (MAPPO
              student), or None to fall back to the rule controller
    fallback: "local" (default), "rule", or "stale" -- the naive failure mode
              in which an unreachable household keeps its last setpoint.
    degrade_frac: if set and at least this share of households is unreachable,
              the operator stops running MAT (its picture of the community is
              too stale) and *every* household runs the local policy; the
              reachable ones still get the operator's community shield.
    """
    residual = False

    def __init__(self, central, local=None, schedule: Optional[OutageSchedule] = None,
                 fallback: str = "local", degrade_frac: Optional[float] = None):
        assert fallback in ("local", "rule", "stale")
        self.degrade_frac = degrade_frac
        self.central, self.local, self.schedule = central, local, schedule
        self.fallback = fallback if (fallback != "local" or local is not None) else "rule"
        self.reset_log()

    def reset_log(self):
        self._last_obs = None
        self._last_action = None
        self.modes: List[np.ndarray] = []

    def act(self, obs, state, env=None):
        E, N = obs.shape[:2]
        if env.t == 0:
            self.reset_log()
        on = self.schedule.online(env.t) if self.schedule else np.ones(N, bool)
        online = np.broadcast_to(on[None], (E, N)).copy()
        env.comms_online = None if online.all() else online
        # The operator only has fresh observations for reachable households.
        if self._last_obs is None:
            self._last_obs = obs.copy()
        self._last_obs = np.where(online[..., None], obs, self._last_obs)
        if hasattr(self.central, "env"):
            self.central.env = env
        a_central = self.central.act(self._last_obs, state, True)[0]
        if online.all():
            a = a_central
            mode = np.full((E, N), MODE_CENTRAL)
        elif (self.degrade_frac is not None and self.fallback == "local"
              and (~online).mean() >= self.degrade_frac):
            a = self.local.act(obs, state, True)[0]
            mode = np.full((E, N), MODE_LOCAL)
        else:
            if self.fallback == "local":
                a_off, m_off = self.local.act(obs, state, True)[0], MODE_LOCAL
            elif self.fallback == "rule":
                a_off, m_off = env.reference_actions(), MODE_RULE
            else:  # stale: keep whatever was last received (zeros before the first one)
                a_off = self._last_action if self._last_action is not None else np.zeros_like(a_central)
                m_off = MODE_STALE
            a = np.where(online[..., None], a_central, a_off)
            mode = np.where(online, MODE_CENTRAL, m_off)
        # an unreachable household never receives the new central setpoint
        self._last_action = np.where(online[..., None], a, self._last_action if self._last_action is not None else a)
        self.modes.append(mode[0].copy())
        return a.astype(np.float32)


def proposed_controller(mat, mappo, schedule=None):
    """The deployed configuration: MAT while every household is reachable; as soon as
    any household is cut off, every household runs its local MAPPO student."""
    return HybridController(mat, mappo, schedule, "local", degrade_frac=DEFAULT_DEGRADE_FRAC)


class RuleAgent:
    """The rule controller behind the agent interface (absolute actions)."""
    env = None

    def act(self, obs, state, deterministic=True):
        return (self.env.reference_actions(),)
