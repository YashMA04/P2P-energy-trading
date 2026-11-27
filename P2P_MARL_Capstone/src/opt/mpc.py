"""Receding-horizon model-predictive control: the industry-standard comparator.

At every step the exact dispatch program (``solve_dispatch(exact=True)``) is
solved from the *current* simulator state over the next ``horizon`` steps,
the first step of the plan is applied, and the rest is discarded.

Information: the controller is given the true load/PV profile inside its
window (a perfect forecast over ``horizon`` steps).  The learned agents see
net-load forecasts at 15 min / 1 h / 3 h only, so MPC with ``horizon=12`` has
somewhat *more* information than the agents, and ``horizon=96`` is the oracle.
This is stated wherever MPC is reported.

Beyond the window, stored battery energy is valued at the cheapest tariff of
the day grossed up by the charging loss (the same rate the inventory-neutral
metric uses) and EV requirements become laxity bounds, so the controller does
not empty the battery or abandon a deadline at the edge of its horizon.
"""
from __future__ import annotations

import numpy as np

from .lp import solve_dispatch
from .replay import schedule_to_actions, v2g_scale_fix


class MPCPolicy:
    residual = False

    def __init__(self, ct, env_cfg, horizon: int = 12, exact: bool = True,
                 time_limit: float = 20.0, accept_incumbent: bool = True):
        self.ct, self.cfg = ct, env_cfg
        self.horizon, self.exact, self.time_limit = int(horizon), bool(exact), float(time_limit)
        # For *control*, a time-limited MILP's best incumbent is still a
        # simulator-feasible plan and measured better than falling back to the
        # convex relaxation (EUR 135.93 vs 143.07 battery-neutral).  For
        # *labels* (DAgger) only proven plans are used.  Default: strongest MPC.
        self.accept_incumbent = bool(accept_incumbent)
        self.name = f"mpc_h{horizon}"
        self.p_refill = ct.buy_price.min(axis=0) / np.sqrt(ct.bess_eta)
        self.solve_failures = 0
        self.fallbacks = 0          # exact plan not proven optimal -> convex relaxation used

    def act(self, obs, state, env=None):
        ct, t = self.ct, env.t
        t1 = min(env.n_steps, t + self.horizon)
        acts = []
        for e in range(env.E):
            r = solve_dispatch(ct, self.cfg, load=env.load[e], pv=env.pv[e], t0=t, t1=t1,
                               bess_soc=env.bess_soc[e], ev_soc=env.ev_soc[:, e],
                               exact=self.exact, terminal_bess=True,
                               terminal_value=self.p_refill, time_limit=self.time_limit)
            usable = r is not None and np.isfinite(r.cost) and (
                r.meta.get("proven", False) or self.accept_incumbent)
            if self.exact and not usable:
                self.fallbacks += 1
                r = solve_dispatch(ct, self.cfg, load=env.load[e], pv=env.pv[e], t0=t, t1=t1,
                                   bess_soc=env.bess_soc[e], ev_soc=env.ev_soc[:, e], exact=False,
                                   terminal_bess=True, terminal_value=self.p_refill)
            if r is None or not np.isfinite(r.cost):
                self.solve_failures += 1
                a = np.zeros((ct.n_agents, 5), np.float32); a[:, 4] = 0.5
            else:
                a = v2g_scale_fix(ct, schedule_to_actions(ct, r.x, 0), t)
            acts.append(a)
        return np.stack(acts).astype(np.float32)
