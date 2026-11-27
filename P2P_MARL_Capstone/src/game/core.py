"""Cooperative game theory: core-stable cost allocation and IR-guaranteed settlement.

The cost-sharing game
---------------------
v(S) = least cost at which coalition S can serve itself as a community with
its own feeder share (8 kW import / 5 kW export per member), i.e. the convex
dispatch program of :mod:`src.opt.lp` restricted to S.  Every constraint's
right-hand side is a sum of per-member resources (each member's load, PV,
storage, connection limits and feeder share), so this is a *linear production
game* in the sense of Owen (1975): the grand coalition's optimal dual prices
y give an allocation

    x_i = sum_r  y_r * b_{r,i}          (b_{r,i}: member i's part of row r)

that is budget-balanced (sum_i x_i = v(N), by strong duality) and lies in the
core (sum_{i in S} x_i <= v(S) for every coalition S): no group of households
can do better by leaving.  ``tests/test_v2_game.py`` checks both on sampled
coalitions instead of trusting the theorem's preconditions.

Settlement of a *realised* day
------------------------------
The learned policy does not achieve v(N), so the Owen allocation cannot be
billed directly.  Realised savings  S = sum_i g_i - C  (g_i: household i's
grid-only bill, C: realised community cost) are shared in proportion to the
savings the core allocation assigns, sigma_i = max(g_i - x_i, 0) / sum(...):

    pay_i = g_i - S * sigma_i

Budget balance holds exactly (sum pay_i = C) and, whenever the community
saves at all (S >= 0), pay_i <= g_i for every household: individual
rationality by construction, for any policy, with no multiplier to learn.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from ..opt.lp import solve_dispatch


def owen_allocation(ct, cfg, **lp_kwargs) -> Dict[str, np.ndarray]:
    """Dual-based allocation of the grand coalition's convex dispatch cost."""
    r = solve_dispatch(ct, cfg, exact=False, return_duals=True, feeder_share=True, **lp_kwargs)
    m = r.meta
    off, K, M = m["var_layout"]
    idx = m["households"]
    N = ct.n_agents
    x = np.zeros(N)
    # equality rows: owner given per row (-1 = community row with b = 0)
    y_eq, b_eq, own = r.duals["eq"], m["b_eq"], m["A_eq_owner"]
    for i in range(N):
        x[i] += float(np.dot(y_eq[own == i], b_eq[own == i]))
    # feeder rows: each member owns an equal share of the capacity
    y_ub, b_ub = r.duals["ub"], m["b_ub"]
    x[idx] += float(np.dot(y_ub, b_ub)) / len(idx)
    # variable bounds: owned by the household whose device the variable is
    nvar = len(m["lo"])
    owner = np.full(nvar, -1)
    for name, start in off.items():
        size = K * M * (2 if name in ("ec", "ed", "ee", "z") else 1)
        if start + size > nvar:
            continue
        owner[start:start + size] = np.tile(idx, size // M)
    for marg, bound in ((r.duals["lower"], m["lo"]), (r.duals["upper"], m["hi"])):
        ok = np.isfinite(bound) & (np.abs(marg) > 0)
        np.add.at(x, owner[ok], marg[ok] * bound[ok])
    return {"x": x, "v_grand_reported": r.cost, "lp": r}


def coalition_value(ct, cfg, members: Sequence[int], **lp_kwargs) -> float:
    """v(S): objective value of the coalition's own convex dispatch program."""
    r = solve_dispatch(ct, cfg, exact=False, households=list(members), feeder_share=True,
                       return_duals=True, **lp_kwargs)
    # objective value (incl. the tiny curtailment/throughput regularisers),
    # consistent with what the dual allocation splits
    obj = 0.0
    off, K, M = r.meta["var_layout"]
    c = r.meta["c"]
    for name, arr in r.x.items():
        start = off[name]
        obj += float(np.dot(c[start:start + arr.size], arr.reshape(-1) if arr.ndim == 2
                            else arr.reshape(-1)))
    return obj


def settle(grid_only: np.ndarray, realised_total: float, x: np.ndarray) -> Dict[str, np.ndarray]:
    """IR-guaranteed, budget-balanced bills for a realised day."""
    g = np.asarray(grid_only, float)
    savings = float(g.sum() - realised_total)
    w = np.maximum(g - x, 0.0)
    sigma = w / w.sum() if w.sum() > 1e-12 else np.full(len(g), 1.0 / len(g))
    pay = g - savings * sigma
    return {"pay": pay, "sigma": sigma, "savings": savings,
            "ir": (pay <= g + 1e-9), "budget_gap": float(pay.sum() - realised_total)}
