"""Hard safety projection applied between the policy and the physics.

Three layers guard the system, in increasing order of strictness:

1. **Residual trust region** (in the environment): the learned action is a
   bounded correction around a transparent rule-based controller.
2. **Feasibility projection** (this module): device limits and, critically, EV
   departure feasibility are enforced *before* dispatch.
3. **Lagrangian constraints** (in the learner): soft economic constraints —
   individual rationality and feeder congestion — are priced by dual variables.

The distinction matters academically.  Layer 2 encodes constraints that are
physically or contractually non-negotiable and that a policy should never be
asked to discover by trial and error.  If an EV must leave at 07:30 with 30 kWh
on board, no amount of reward shaping should be trusted to guarantee it; the
projection simply removes infeasible actions from the action set.  Layer 3
encodes constraints that are genuinely economic trade-offs, where the correct
shadow price is unknown a priori and is therefore *learned*.

Reference: this is the "shielding"/"safety layer" pattern (Dalal et al., 2018,
*Safe Exploration in Continuous Action Spaces*; Alshiekh et al., 2018), adapted
to a multi-agent dispatch setting.
"""
from __future__ import annotations

import numpy as np


class EVFeasibilityProjection:
    """Force-charge EVs that can no longer meet their departure target.

    The test is *least laxity*, expressed in steps rather than in energy.  At
    each step we ask what the vehicle could still gain if it declined to charge
    right now and then charged flat out for every remaining connected step::

        gain_if_deferred = (usable_steps - 1) * P_max * eta * dt

    If that is less than the remaining deficit, this step is no longer
    optional and the setpoint is raised to the smallest value that keeps the
    target reachable.

    Testing the energy slack over *all* remaining steps instead, as a first
    implementation of this class did, silently fails on the final step: with
    one step left and a 0.4 kWh deficit the slack is still positive, so the
    shield declines to fire and the target is missed by exactly the amount it
    was protecting.  That defect produced ~10 kWh of unmet mobility demand per
    day across the fleet while reporting the shield as active.

    Raising the setpoint only to the level required -- rather than to full
    power -- matters too: forcing every shielded vehicle to maximum draw
    manufactures precisely the synchronised peak the feeder constraint is
    meant to avoid.
    """

    def __init__(self, margin_kwh: float = 0.5, enabled: bool = True, minimal: bool = False):
        self.margin_kwh = float(margin_kwh)
        self.enabled = bool(enabled)
        # v1 raised a shielded car to deficit / per_step ("finish the whole
        # deficit now"), not to the minimum.  At 23:15 that drives the whole
        # fleet to full power simultaneously.  ``minimal`` raises it only to
        # what deferral cannot cover: (deficit + margin - gain_if_deferred).
        self.minimal = bool(minimal)

    def __call__(self, ev_action: np.ndarray, ev_soc: np.ndarray, ctx) -> tuple:
        """Project EV setpoints.

        Parameters
        ----------
        ev_action : (2, E, N) desired setpoints in [-1, 1]
        ev_soc    : (2, E, N) current state of charge in kWh
        ctx       : object exposing ``ev_target``, ``ev_charge_max``, ``ev_eta``,
                    ``ev_at_home`` (2, T, N), ``ev_departure`` (2, N), ``dt``, ``t``

        Returns
        -------
        (projected_action, must_charge_mask)
        """
        if not self.enabled:
            return ev_action, np.zeros_like(ev_action, dtype=bool)

        t = ctx.t
        T = ctx.n_steps
        # Steps remaining at home before departure (or before end of horizon).
        dep = ctx.ev_departure                                   # (2, N)
        horizon = np.where(dep > t, dep - t, T - t)              # (2, N)
        horizon = np.maximum(horizon, 0)
        # Only steps in which the vehicle is actually connected can be used.
        avail = ctx.ev_home_remaining(t)                         # (2, N) counts
        usable = np.minimum(horizon, avail).astype(np.float32)
        if self.minimal:
            # v1 capped by ``horizon = T - 1 - t`` after departure, which
            # excludes the final step although the end-of-day target is
            # checked *after* charging in it.  The connected-step count already
            # runs to the active deadline inclusive, so it alone is correct.
            usable = avail.astype(np.float32)

        eta = np.sqrt(ctx.ev_eta)                                # (2, N)
        per_step = ctx.ev_charge_max * eta * ctx.dt              # (2, N) kWh per step
        gain_if_deferred = np.maximum(usable - 1.0, 0.0) * per_step
        deficit = np.maximum(ctx.ev_target[:, None, :] - ev_soc, 0.0)  # (2,E,N)
        at_home = ctx.ev_at_home[:, t, :][:, None, :] > 0.5
        must = ((gain_if_deferred[:, None, :] < deficit + self.margin_kwh)
                & (deficit > 1e-6) & at_home)

        if self.minimal:
            short = np.maximum(deficit + self.margin_kwh - gain_if_deferred[:, None, :], 0.0)
            required = np.clip(np.minimum(short, deficit) / np.maximum(per_step[:, None, :], 1e-9), 0.0, 1.0)
        else:
            required = np.clip(deficit / np.maximum(per_step[:, None, :], 1e-9), 0.0, 1.0)
        projected = np.where(must, np.maximum(ev_action, required), ev_action)
        # Exposed so the environment can make only the *required* part of the
        # setpoint firm, rather than everything the policy asked for.
        self.last_required = np.where(must, required, 0.0).astype(np.float32)
        return projected.astype(np.float32), must


class CommunityFeasibilityShield:
    """Shared-feeder extension of the least-laxity shield.

    The per-vehicle shield asks "can *this* car still make its deadline?".
    Every car can answer yes while the fleet as a whole cannot: in the v1
    study 41 vehicles were shielded simultaneously at 23:30 against a common
    end-of-day deadline and their firm charging exceeded the feeder.  The
    question has to be asked of the fleet.

    Method (as-late-as-possible allocation of future headroom)::

        H[tau] = feeder_import - base_net_load[tau] - margin      tau > t
        walk tau = T-1 ... t+1 backwards, giving H[tau] to the vehicles that are
        connected at tau and whose active deadline is >= tau, pro rata to their
        unmet need and capped at their charger rating;
        whatever need cannot be placed in the future must be charged NOW.

    That residual becomes a firm minimum setpoint this step.  Base load is
    taken from the episode's own profile (a perfect base-load forecast); with
    noisy forecasts ``margin_kw`` reserves headroom against forecast error.
    Batteries and V2G are ignored when computing headroom, which is
    conservative.  The backward allocation is a heuristic certificate, not an
    exact max-flow; ``tests/test_v2_shield.py`` checks the property that
    matters (no unserved energy under arbitrary actions) directly.
    """

    def __init__(self, margin_kw: float = 0.0, enabled: bool = True, passes: int = 2,
                 need_margin_kwh: float = 0.5):
        self.margin_kw = float(margin_kw)
        # Plan for the per-vehicle shield's own safety margin too; otherwise
        # the fleet plan lands exactly on the deadline and the per-vehicle
        # shield overrides it in the last steps.
        self.need_margin_kwh = float(need_margin_kwh)
        self.enabled = bool(enabled)
        self.passes = int(passes)
        self.last_required = None

    def required_now(self, env) -> np.ndarray:
        """(2, E, N) kW that must be drawn this step for fleet feasibility."""
        ct, t, T, dt = env.ct, env.t, env.n_steps, env.dt
        E, N = env.E, env.N
        eta = np.sqrt(ct.ev_eta)                                  # (2, N)
        pmax = ct.ev_charge_max                                   # (2, N) kW
        need = np.maximum(env._ev_target_t[:, t, :][:, None, :] - env.ev_soc, 0.0)  # (2,E,N) kWh
        need = np.where(need > 1e-6, need + self.need_margin_kwh, 0.0)
        out = np.zeros((2, E, N), np.float32)
        if t >= T - 1 or not (need > 1e-9).any():
            return out
        dep = ct.ev_departure                                     # (2, N)
        # active deadline: last step at which charging still counts
        deadline = np.where(t <= dep, dep, T - 1)                 # (2, N)
        online = getattr(env, "comms_online", None)               # (E, N) bool or None
        cap = getattr(env.cfg, "offline_import_cap_kw", None)
        if online is not None and cap is not None:
            # Envelope mode: offline households are bounded by their own
            # envelope, so the operator reserves `cap` for each of them and plans
            # only the online fleet in what is left.  Offline vehicles are then
            # planned per household inside their envelope (_envelope_required).
            base_on = np.where(online[:, None, :], env.load - env.pv, 0.0).sum(axis=2)
            reserve = (~online).sum(axis=1)[:, None] * cap
            head = np.maximum(env.feeder_import_kw - base_on - reserve - self.margin_kw, 0.0)
            need = np.where(online[None], need, 0.0)
        else:
            base = (env.load - env.pv).sum(axis=2)                # (E, T)
            head = np.maximum(env.feeder_import_kw - base - self.margin_kw, 0.0)  # (E, T) kW
        resid = need.copy()                                       # kWh still to place
        offline = None if (online is None or cap is not None) else np.broadcast_to(~online[None], (2, E, N))
        for tau in range(T - 1, t, -1):
            elig = ((ct.ev_at_home[:, tau, :] > 0.5) & (tau <= deadline))[:, None, :]  # (2,1,N)
            if not elig.any():
                continue
            cap_v = np.where(elig, np.minimum(pmax[:, None, :], resid / (eta[:, None, :] * dt)), 0.0)
            H = head[:, tau].copy()                               # (E,)
            give = np.zeros_like(cap_v)
            # Offline vehicles first: they will be driven by their *local*
            # least-laxity shield, i.e. as late as possible, so the late slots
            # are theirs whatever the operator plans.  Online vehicles get
            # what is left.  With everyone online this is the v2 behaviour.
            groups = (offline, ~offline) if offline is not None else (None,)
            for grp in groups:
                for _ in range(self.passes):                      # pro rata, then redistribute
                    room = np.maximum(cap_v - give, 0.0)
                    if grp is not None:
                        room = np.where(grp, room, 0.0)
                    tot = room.sum(axis=(0, 2))                   # (E,)
                    if not (tot > 1e-9).any():
                        break
                    frac = np.minimum(1.0, H / np.maximum(tot, 1e-9))
                    add = room * frac[None, :, None]
                    give += add
                    H = np.maximum(H - add.sum(axis=(0, 2)), 0.0)
            resid = np.maximum(resid - give * eta[:, None, :] * dt, 0.0)
        home_now = (ct.ev_at_home[:, t, :] > 0.5)[:, None, :]
        out = np.where(home_now, np.minimum(resid / (eta[:, None, :] * dt), pmax[:, None, :]), 0.0)
        return out.astype(np.float32)

    def _envelope_required(self, env, cap: float) -> np.ndarray:
        """(2, E, N) kW each household must draw now to meet its own EV deadlines
        while staying inside its own import envelope ``cap`` -- the same
        as-late-as-possible allocation, run per household on local data only.
        ``self.envelope_infeasible`` counts households whose deadlines cannot
        be met inside the envelope (they then draw what the deadline needs)."""
        ct, t, T, dt = env.ct, env.t, env.n_steps, env.dt
        E, N = env.E, env.N
        eta = np.sqrt(ct.ev_eta)
        pmax = ct.ev_charge_max
        need = np.maximum(env._ev_target_t[:, t, :][:, None, :] - env.ev_soc, 0.0)
        need = np.where(need > 1e-6, need + self.need_margin_kwh, 0.0)
        out = np.zeros((2, E, N), np.float32)
        if t >= T - 1 or not (need > 1e-9).any():
            return out
        deadline = np.where(t <= ct.ev_departure, ct.ev_departure, T - 1)
        head = np.maximum(cap - (env.load - env.pv), 0.0)               # (E, T, N)
        resid = need.copy()
        for tau in range(T - 1, t, -1):
            elig = ((ct.ev_at_home[:, tau, :] > 0.5) & (tau <= deadline))[:, None, :]
            if not elig.any():
                continue
            cap_v = np.where(elig, np.minimum(pmax[:, None, :], resid / (eta[:, None, :] * dt)), 0.0)
            H = head[:, tau, :].copy()                                  # (E, N)
            give = np.zeros_like(cap_v)
            for _ in range(self.passes):
                room = np.maximum(cap_v - give, 0.0)
                tot = room.sum(axis=0)                                  # per household
                frac = np.minimum(1.0, H / np.maximum(tot, 1e-9))
                add = room * frac[None]
                give += add
                H = np.maximum(H - add.sum(axis=0), 0.0)
            resid = np.maximum(resid - give * eta[:, None, :] * dt, 0.0)
        home_now = (ct.ev_at_home[:, t, :] > 0.5)[:, None, :]
        out = np.where(home_now, np.minimum(resid / (eta[:, None, :] * dt), pmax[:, None, :]), 0.0)
        room_now = np.maximum(cap - (env.load[:, t, :] - env.pv[:, t, :]), 0.0)   # (E, N)
        self.envelope_infeasible = int((out.sum(axis=0) > room_now + 1e-6).sum())
        return out.astype(np.float32)

    def __call__(self, ev_action: np.ndarray, env) -> tuple:
        if not self.enabled:
            self.last_required = np.zeros_like(ev_action)
            return ev_action, np.zeros_like(ev_action, dtype=bool)
        req_kw = self.required_now(env)
        req = np.clip(req_kw / np.maximum(env.ct.ev_charge_max[:, None, :], 1e-9), 0.0, 1.0)
        online = getattr(env, "comms_online", None)
        if online is not None:
            # The operator cannot command a household it cannot reach.
            req = np.where(online[None], req, 0.0)
            cap = getattr(env.cfg, "offline_import_cap_kw", None)
            if cap is not None:
                # ...but that household plans its own cars inside its envelope
                # (computed locally from its own data).
                loc = self._envelope_required(env, cap)
                loc = np.clip(loc / np.maximum(env.ct.ev_charge_max[:, None, :], 1e-9), 0.0, 1.0)
                req = np.where(online[None], req, loc)
        must = req > 1e-6
        self.last_required = req.astype(np.float32)
        return np.where(must, np.maximum(ev_action, req), ev_action).astype(np.float32), must


def project_p2p_quotes(quantity: np.ndarray, price: np.ndarray,
                       buy_price: np.ndarray, sell_price: np.ndarray):
    """Map raw actions to admissible market quotes.

    A rational agent never bids above the grid import tariff (it could buy from
    the grid instead) and never asks below the grid export tariff (it could sell
    to the grid instead).  Restricting quotes to ``[sell_price, buy_price]``
    therefore removes only dominated actions.  This keeps the P2P price inside
    the tariff spread by construction, which is the standard individual
    rationality condition for a P2P market and removes an entire failure mode
    in which an untrained policy quotes nonsense and destroys welfare.
    """
    q = np.abs(np.clip(quantity, -1.0, 1.0))
    frac = np.clip(price, 0.0, 1.0)
    spread = np.maximum(buy_price - sell_price, 1e-6)
    quote = sell_price + frac * spread
    return q, quote
