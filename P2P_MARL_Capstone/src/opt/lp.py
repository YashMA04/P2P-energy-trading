"""General dispatch linear program, consistent with the simulator.

One builder serves four roles, so that every optimisation-based number in the
project comes from the same model:

* the **perfect-foresight bound** (full day, from the initial state);
* **receding-horizon MPC** (from the current state, over a window);
* the **privileged teacher** (full day on perturbed load/PV profiles);
* the **coalition game** (any subset of households, with its feeder share).

Differences from the legacy ``perfect_foresight_lp`` -- each one a place where
the old bound was not settled under the same rules as the simulator:

1. Grid exchange is per household, at that household's own tariff.  The old
   LP priced community exchange at the *mean* tariff; tariffs differ across
   households by up to EUR 0.065/kWh in a single step.
2. Battery degradation (EUR/kWh throughput) is charged, as the simulator does.
3. Optional end-of-day battery condition (SoC_T >= SoC_0), so the bound does
   not count the starting inventory as free energy.
4. Peer-to-peer exchange is explicit (sum of purchases = sum of sales) and
   capped by each household's connection limits, as in the simulator.

Known remaining relaxations, stated rather than hidden:
* the P2P platform fee (1% each side, < EUR 0.5/day) is not modelled;
* EV discharge is not floored at the active target (the simulator floors it).
Both make the LP cheaper or equal, so it remains a valid lower bound on cost.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
from scipy import sparse
from scipy.optimize import linprog


@dataclass
class LPResult:
    cost: float                       # EUR, simulator-consistent objective
    status: str
    x: Dict[str, np.ndarray]          # named schedules, (K, M) or (2, K, M)
    duals: Dict[str, np.ndarray] = field(default_factory=dict)
    meta: Dict = field(default_factory=dict)


def solve_dispatch(ct, cfg, *, load: Optional[np.ndarray] = None,
                   pv: Optional[np.ndarray] = None,
                   households: Optional[Sequence[int]] = None,
                   t0: int = 0, t1: Optional[int] = None,
                   bess_soc: Optional[np.ndarray] = None,
                   ev_soc: Optional[np.ndarray] = None,
                   degradation: bool = True,
                   terminal_bess: bool = True,
                   terminal_value: Optional[np.ndarray] = None,
                   feeder_share: bool = True,
                   exact: bool = False,
                   time_limit: float = 300.0,
                   return_duals: bool = False) -> Optional[LPResult]:
    """Least-cost dispatch over steps ``[t0, t1)`` for ``households``.

    ``bess_soc`` (N,) and ``ev_soc`` (2, N) give the state at the start of
    ``t0`` (defaults: the dataset's initial state, valid only for t0 = 0).
    When ``t1 < T`` the EV requirements that fall after the window become a
    laxity bound at the window end, and stored battery energy is valued at
    ``terminal_value`` EUR/kWh instead of being constrained.
    ``feeder_share``: the feeder capacity available to a coalition is its
    per-household share (needed for the coalition game); for the grand
    coalition this equals the full feeder.
    """
    T, dt = ct.n_steps, ct.dt
    t1 = T if t1 is None else int(t1)
    idx = np.arange(ct.n_agents) if households is None else np.asarray(households, int)
    M, K = len(idx), t1 - t0
    load = (ct.load_kw if load is None else load)[t0:t1][:, idx].astype(float)
    pv = (ct.pv_kw if pv is None else pv)[t0:t1][:, idx].astype(float)
    buy = ct.buy_price[t0:t1][:, idx].astype(float)
    sell = ct.sell_price[t0:t1][:, idx].astype(float)
    b_soc0 = (ct.bess_soc0 if bess_soc is None else bess_soc)[idx].astype(float)
    e_soc0 = (ct.ev_soc0 if ev_soc is None else ev_soc)[:, idx].astype(float)
    eta_b = np.sqrt(ct.bess_eta[idx]).astype(float)
    eta_e = np.sqrt(ct.ev_eta[:, idx]).astype(float)
    mu = float(cfg.unserved_price_multiple)
    deg = float(cfg.bess_degradation_eur_per_kwh) if degradation else 1e-4
    dep_target = getattr(ct, "ev_departure_target", ct.ev_target)[:, idx]
    fin_target = getattr(ct, "ev_final_target", ct.ev_target)[:, idx]
    dep = ct.ev_departure[:, idx].astype(int)
    home = (ct.ev_at_home[:, t0:t1][:, :, idx] > 0.5)
    v2g = np.minimum(ct.ev_discharge_max[:, None, idx], ct.ev_v2g_max[:, t0:t1][:, :, idx])
    F_imp = cfg.feeder_import_kw_per_hh * (M if feeder_share else ct.n_agents)
    F_exp = cfg.feeder_export_kw_per_hh * (M if feeder_share else ct.n_agents)

    # ---------------- variable layout ---------------------------------
    blocks = [("bc", 1), ("bd", 1), ("eb", 1), ("ec", 2), ("ed", 2), ("ee", 2),
              ("imp", 1), ("exp", 1), ("pin", 1), ("pout", 1), ("cur", 1), ("us", 1)]
    if exact:
        blocks += [("y", 1), ("z", 2)]          # y=1 importer; z=1 EV may discharge
    off, cur = {}, 0
    for name, mult in blocks:
        off[name] = cur
        cur += mult * K * M
    nvar = cur
    ar = np.arange(K * M).reshape(K, M)

    def V(name, j=None):
        """(K, M) array of variable indices."""
        base = off[name] + (0 if j is None else j * K * M)
        return base + ar

    lo = np.zeros(nvar)
    hi = np.full(nvar, np.inf)
    c = np.zeros(nvar)

    def setb(name, lo_v=None, hi_v=None, j=None):
        ix = V(name, j)
        if lo_v is not None:
            lo[ix] = np.broadcast_to(lo_v, (K, M))
        if hi_v is not None:
            hi[ix] = np.broadcast_to(hi_v, (K, M))

    setb("bc", hi_v=ct.bess_charge_max[idx][None])
    setb("bd", hi_v=ct.bess_discharge_max[idx][None])
    cap = ct.bess_capacity[idx]
    setb("eb", lo_v=np.minimum(cfg.bess_min_soc * cap, b_soc0)[None],
         hi_v=np.maximum(cfg.bess_max_soc * cap, b_soc0)[None])
    for j in range(2):
        setb("ec", hi_v=np.where(home[j], ct.ev_charge_max[j, idx][None], 0.0), j=j)
        setb("ed", hi_v=np.where(home[j], v2g[j], 0.0), j=j)
        setb("ee", hi_v=ct.ev_capacity[j, idx][None], j=j)
    setb("pin", hi_v=ct.buy_limit_kw[idx][None])
    setb("pout", hi_v=ct.sell_limit_kw[idx][None])
    setb("cur", hi_v=np.maximum(pv, 0.0))

    # objective (EUR): everything multiplied by dt to go from kW to kWh
    c[V("imp").ravel()] = (buy * dt).ravel()
    c[V("exp").ravel()] = (-sell * dt).ravel()
    c[V("us").ravel()] = (buy * mu * dt).ravel()
    c[V("cur").ravel()] = 1e-3 * dt
    c[V("bc").ravel()] += deg * dt
    c[V("bd").ravel()] += deg * dt

    # EV requirements as lower bounds on the energy variables
    for j in range(2):
        for m in range(M):
            d = dep[j, m]
            if t0 <= d < t1:                       # departure inside window
                need = float(dep_target[j, m] - ct.ev_trip_kwh[j, idx[m]])
                k = d - t0
                lo[V("ee", j)[k, m]] = max(lo[V("ee", j)[k, m]], min(need, ct.ev_capacity[j, idx[m]]))
            fin = float(min(fin_target[j, m], ct.ev_capacity[j, idx[m]]))
            if t1 == T:
                lo[V("ee", j)[K - 1, m]] = max(lo[V("ee", j)[K - 1, m]], fin)
            else:
                # Window ends before a deadline: the remaining requirement must
                # still be reachable by charging flat out at every connected
                # step after the window (a laxity bound, not a target).
                full_home = ct.ev_at_home[j, :, idx[m]] > 0.5
                rate = ct.ev_charge_max[j, idx[m]] * eta_e[j, m] * dt
                if t1 <= d:      # still before departure
                    after = full_home[t1:d + 1].sum()
                    req = float(dep_target[j, m]) - after * rate
                else:
                    after = full_home[t1:T].sum()
                    req = fin - after * rate
                if req > 1e-6:
                    lo[V("ee", j)[K - 1, m]] = max(lo[V("ee", j)[K - 1, m]],
                                                    min(req, ct.ev_capacity[j, idx[m]]))
    if t1 == T and terminal_bess:
        lo[V("eb")[K - 1]] = np.maximum(lo[V("eb")[K - 1]], np.minimum(b_soc0, hi[V("eb")[K - 1]]))
    if t1 < T and terminal_value is not None:
        c[V("eb")[K - 1]] -= np.asarray(terminal_value, float)[idx]

    # ---------------- equality constraints -----------------------------
    R, C, Vv, beq, owner = [], [], [], [], []
    r = 0

    def add_rows(cols_coefs, rhs, own):
        """cols_coefs: list of (K,M) index arrays with scalar/array coefs."""
        nonlocal r
        n = rhs.size
        rows = r + np.arange(n).reshape(rhs.shape)
        for cols, coef in cols_coefs:
            R.append(rows.ravel()); C.append(cols.ravel())
            Vv.append(np.broadcast_to(coef, rhs.shape).ravel().astype(float))
        beq.append(rhs.ravel()); owner.append(np.broadcast_to(own, rhs.shape).ravel())
        r += n

    own_m = np.broadcast_to(idx[None], (K, M))
    # (1) household power balance
    terms = [(V("bc"), 1.0), (V("bd"), -1.0), (V("cur"), 1.0), (V("imp"), -1.0),
             (V("exp"), 1.0), (V("pin"), -1.0), (V("pout"), 1.0), (V("us"), -1.0)]
    for j in range(2):
        terms += [(V("ec", j), 1.0), (V("ed", j), -1.0)]
    add_rows(terms, pv - load, own_m)
    # (2) battery dynamics: eb[k] - eb[k-1] - dt*eta*bc + dt/eta*bd = 0 (k=0: = soc0)
    rhs = np.zeros((K, M)); rhs[0] = b_soc0
    prev = np.full((K, M), -1); prev[1:] = V("eb")[:-1]
    t_rows = [(V("eb"), 1.0), (V("bc"), -dt * eta_b[None]), (V("bd"), dt / eta_b[None])]
    add_rows(t_rows, rhs, own_m)
    # previous-SoC coupling added separately (rows K..)
    base = r - K * M
    rr = base + np.arange(K * M).reshape(K, M)[1:]
    R.append(rr.ravel()); C.append(V("eb")[:-1].ravel()); Vv.append(-np.ones(rr.size))
    # (3) EV dynamics with trip drawdown at the departure step
    for j in range(2):
        rhs = np.zeros((K, M))
        steps = np.arange(t0, t1)[:, None]
        rhs -= np.where(steps == dep[j][None], ct.ev_trip_kwh[j, idx][None], 0.0)
        rhs[0] += e_soc0[j]
        add_rows([(V("ee", j), 1.0), (V("ec", j), -dt * eta_e[j][None]),
                  (V("ed", j), dt / eta_e[j][None])], rhs, own_m)
        base = r - K * M
        rr = base + np.arange(K * M).reshape(K, M)[1:]
        R.append(rr.ravel()); C.append(V("ee", j)[:-1].ravel()); Vv.append(-np.ones(rr.size))
    # (4) P2P clears inside the community: sum(pin) - sum(pout) = 0 each step
    add_rows([(V("pin"), np.ones((K, M))), (V("pout"), -np.ones((K, M)))],
             np.zeros((K, M)), own_m)  # placeholder rows, collapsed below
    n_eq = r
    A_eq = sparse.coo_matrix((np.concatenate(Vv), (np.concatenate(R), np.concatenate(C))),
                             shape=(n_eq, nvar)).tocsr()
    b_eq = np.concatenate(beq)
    eq_owner = np.concatenate(owner)
    # collapse the per-household P2P placeholder rows into one row per step
    p_rows = np.arange(n_eq - K * M, n_eq).reshape(K, M)
    collapse = sparse.lil_matrix((K, n_eq))
    for k in range(K):
        collapse[k, p_rows[k]] = 1.0
    keep = np.ones(n_eq, bool); keep[p_rows.ravel()] = False
    A_eq = sparse.vstack([A_eq[keep], collapse.tocsr() @ A_eq]).tocsr()
    b_eq = np.concatenate([b_eq[keep], np.zeros(K)])
    eq_owner = np.concatenate([eq_owner[keep], np.full(K, -1)])

    # ---------------- inequality: feeder ---------------------------------
    rows_i, cols_i, vals_i = [], [], []
    for k in range(K):
        rows_i += [k] * (2 * M) + [K + k] * (2 * M)
        cols_i += list(V("imp")[k]) + list(V("exp")[k]) + list(V("exp")[k]) + list(V("imp")[k])
        vals_i += [1.0] * M + [-1.0] * M + [1.0] * M + [-1.0] * M
    A_ub = sparse.coo_matrix((vals_i, (rows_i, cols_i)), shape=(2 * K, nvar)).tocsr()
    b_ub = np.concatenate([np.full(K, F_imp), np.full(K, F_exp)])
    integrality = None
    if exact:
        A_x, b_x = _exact_rows(V, K, M, nvar, lo, hi, ct, idx, t0, t1, dep,
                               dep_target, fin_target, eta_e, dt, load, pv, e_soc0)
        A_ub = sparse.vstack([A_ub, A_x]).tocsr()
        b_ub = np.concatenate([b_ub, b_x])
        hi[off["y"]:] = 1.0
        integrality = np.zeros(nvar); integrality[off["y"]:] = 1

    opts = {"time_limit": time_limit, "mip_rel_gap": 1e-4} if exact else {}
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=np.stack([lo, hi], 1), method="highs",
                  integrality=integrality, options=opts)
    if res.x is None:
        return LPResult(np.nan, str(res.message), {}, meta={"proven": False, "exact": exact})
    xx = res.x
    def get(name, j=None):
        return xx[V(name, j)]
    x = {k: get(k) for k in ("bc", "bd", "eb", "imp", "exp", "pin", "pout", "cur", "us")}
    for k in ("ec", "ed", "ee"):
        x[k] = np.stack([get(k, 0), get(k, 1)])
    deg_eur = float(cfg.bess_degradation_eur_per_kwh) * dt * float((x["bc"] + x["bd"]).sum())
    cost = float((buy * x["imp"] - sell * x["exp"] + buy * mu * x["us"]).sum() * dt + deg_eur)
    out = LPResult(cost, str(res.message), x,
                   meta={"households": idx, "t0": t0, "t1": t1,
                         "grid_import_kwh": float(x["imp"].sum() * dt),
                         "unserved_kwh": float(x["us"].sum() * dt),
                         "peak_import_kw": float((x["imp"] - x["exp"]).sum(1).max()),
                         "degradation_eur": deg_eur,
                         "v2g_kwh": float(x["ed"].sum() * dt),
                         "bess_end_minus_start_kwh": float(x["eb"][-1].sum() - b_soc0.sum()),
                         "exact": exact,
                         # True only when the solver *proved* the plan is (near-)optimal.
                         # A time-limited MILP returns its incumbent, which can be
                         # arbitrarily bad (measured: EUR 134.52 vs optimal 7.24).
                         "proven": (not exact) or (res.status == 0 and
                                                   float(getattr(res, "mip_gap", 1.0) or 0.0) <= 1e-3),
                         "mip_gap": float(getattr(res, "mip_gap", 0.0) or 0.0),
                         "mip_dual_bound": float(getattr(res, "mip_dual_bound", np.nan) or np.nan)})
    if return_duals:
        out.duals = {"eq": res.eqlin.marginals, "ub": res.ineqlin.marginals,
                     "lower": res.lower.marginals, "upper": res.upper.marginals}
        out.meta.update({"A_eq_owner": eq_owner, "b_eq": b_eq, "b_ub": b_ub,
                         "lo": lo, "hi": hi, "var_layout": (off, K, M), "c": c})
    return out


def _exact_rows(V, K, M, nvar, lo, hi, ct, idx, t0, t1, dep, dep_target, fin_target,
                eta_e, dt, load, pv, e_soc0):
    """Big-M rows for the importer/exporter and EV-floor rules (A x <= b)."""
    rows, cols, vals, b = [], [], [], []
    r = 0

    def row(cs, vs, rhs):
        nonlocal r
        rows.extend([r] * len(cs)); cols.extend(cs); vals.extend(vs); b.append(rhs); r += 1

    big = float(max(load.max(), pv.max()) + ct.bess_charge_max.max() + 2 * ct.ev_charge_max.max() + 50.0)
    Y, IMP, EXP, PIN, POUT = V("y"), V("imp"), V("exp"), V("pin"), V("pout")
    for k in range(K):
        for m in range(M):
            y = Y[k, m]
            row([IMP[k, m], y], [1.0, -big], 0.0)            # imp  <= big*y
            row([PIN[k, m], y], [1.0, -big], 0.0)            # pin  <= big*y
            row([EXP[k, m], y], [1.0, big], big)             # exp  <= big*(1-y)
            row([POUT[k, m], y], [1.0, big], big)            # pout <= big*(1-y)
    for j in range(2):
        Z, ED, EE = V("z", j), V("ed", j), V("ee", j)
        for m in range(M):
            capj = float(ct.ev_capacity[j, idx[m]])
            d = int(dep[j, m])
            for k in range(K):
                t = t0 + k
                floor = float(min(dep_target[j, m] if t <= d else fin_target[j, m], capj))
                vmax = hi[ED[k, m]]
                if not np.isfinite(vmax) or vmax <= 0:
                    hi[Z[k, m]] = 0.0
                    continue
                row([ED[k, m], Z[k, m]], [1.0, -vmax], 0.0)  # ed <= vmax*z
                # discharge only from energy above the floor:
                #   ee_prev - ed*dt/eta >= floor - cap*(1-z)
                prev = EE[k - 1, m] if k > 0 else None
                if prev is None:
                    soc_prev = float(e_soc0[j, m])
                    row([ED[k, m], Z[k, m]], [dt / eta_e[j, m], capj], soc_prev - floor + capj)
                else:
                    row([ED[k, m], prev, Z[k, m]], [dt / eta_e[j, m], -1.0, capj], -floor + capj)
    A = sparse.coo_matrix((vals, (rows, cols)), shape=(r, nvar)).tocsr()
    return A, np.asarray(b, float)
