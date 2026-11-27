"""Build a custom energy community for the stakeholder platform.

A *specification* is a small JSON-friendly dict:

    {"n_households": 25,          # 4..60 homes, drawn from the 250-home dataset
     "seed": 0,                    # which homes are drawn (stratified by PV/battery/EV mix)
     "feeder_kw_per_home": 8.0,    # feeder import limit = this x number of homes
     "export_kw_per_home": 5.0,    # feeder export limit = this x number of homes
     "houses": [{"pv_kw": 2.6, "battery_kwh": 9.3, "evs": 2, "load_scale": 1.0}, ...]}

``default_spec()`` returns the study community (25 homes, seed 0); building it
reproduces the tensors used for every reported result exactly.  Each house entry
overrides that home's real data:

* ``pv_kw``      peak solar output; an existing profile is rescaled, a home without
                 solar gets the dataset's average solar shape scaled to this peak
* ``battery_kwh`` home battery size (0 = none); power 0.5 C, 95 % efficiency and
                 half-full start/end, the dataset medians
* ``evs``        0, 1 or 2 cars; removed cars no longer charge or travel
* ``load_scale`` multiplies the home's everyday demand

The feeder's per-home import limit is also the offline envelope (see
docs/V2_HYBRID.md), so the operator's reservation for cut-off homes always fits.
The learned controllers were trained on the study community only: on a custom
community they run unchanged, and the safety layer still holds, but their cost
is a generalisation result, not a trained one.
"""
from __future__ import annotations

import copy
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ..data.community import build_tensors, load_community, stratified_subset
from ..data.schemas import BESSProfile, CommunityData
from ..utils.config import build_configs, load_config

ROOT = Path(__file__).resolve().parents[2]
LIMITS = {"n_households": (4, 60), "feeder_kw_per_home": (4.0, 17.0), "export_kw_per_home": (2.0, 17.0),
          "pv_kw": (0.0, 15.0), "battery_kwh": (0.0, 30.0), "evs": (0, 2), "load_scale": (0.25, 3.0)}
TOL = 0.051                    # spec values are rounded to 0.1; within this, the real value is kept


class SpecError(ValueError):
    """A specification value is missing or out of range; the message says which."""


@lru_cache(maxsize=1)
def _base():
    cfg = load_config(ROOT / "configs/v2.yaml")
    pool = load_community(ROOT / cfg["data"]["path"])
    env_cfg, _, _ = build_configs(cfg)
    pv = np.stack([h.pv_profile_kw for h in pool.households if h.has_pv and np.max(h.pv_profile_kw) > 0])
    shape = (pv / pv.max(1, keepdims=True)).mean(0)
    return cfg, pool, env_cfg, shape / shape.max()


def study_community():
    """(ct, env_cfg) exactly as every script in the repository builds them."""
    cfg, pool, env_cfg, _ = _base()
    return build_tensors(stratified_subset(pool, cfg["data"]["households"], seed=cfg["data"]["subset_seed"])), env_cfg


def _native(h) -> Dict:
    return {"pv_kw": round(float(np.max(h.pv_profile_kw)), 1) if h.has_pv else 0.0,
            "battery_kwh": round(float(h.bess.capacity_kwh), 1) if (h.has_bess and h.bess) else 0.0,
            "evs": int(sum(bool(ev and ev.max_charge_kw > 0 and ev.capacity_kwh > 0) for ev in (h.ev1, h.ev2))),
            "load_scale": 1.0}


def _homes(n: int, seed: int) -> List:
    cfg, pool, _, _ = _base()
    return list(stratified_subset(pool, n, seed=seed).households)


def default_spec(n: int = 25, seed: int = 0) -> Dict:
    _, _, env_cfg, _ = _base()
    return {"n_households": n, "seed": seed,
            "feeder_kw_per_home": float(env_cfg.feeder_import_kw_per_hh),
            "export_kw_per_home": float(env_cfg.feeder_export_kw_per_hh),
            "houses": [dict(_native(h), source_id=int(h.player_id)) for h in _homes(n, seed)]}


def _num(spec, key, kind=float):
    v = spec.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise SpecError(f"{key} must be a number")
    lo, hi = LIMITS[key]
    if not (lo <= v <= hi):
        raise SpecError(f"{key} must be between {lo:g} and {hi:g} (got {v:g})")
    if kind is int and int(v) != v:
        raise SpecError(f"{key} must be a whole number")
    return kind(v)


def _apply(h, want: Dict, native: Dict, pv_shape: np.ndarray):
    h = copy.deepcopy(h)
    if abs(want["pv_kw"] - native["pv_kw"]) > TOL:
        if want["pv_kw"] <= 0:
            h.has_pv, h.pv_profile_kw = False, np.zeros_like(h.load_profile_kw)
        elif h.has_pv and np.max(h.pv_profile_kw) > 0:
            h.pv_profile_kw = h.pv_profile_kw * (want["pv_kw"] / np.max(h.pv_profile_kw))
        else:
            h.has_pv, h.pv_profile_kw = True, pv_shape * want["pv_kw"]
    if abs(want["battery_kwh"] - native["battery_kwh"]) > TOL:
        c = want["battery_kwh"]
        if c <= 0:
            h.has_bess, h.bess = False, BESSProfile(has_bess=False)
        else:
            h.has_bess = True
            h.bess = BESSProfile(has_bess=True, capacity_kwh=c, max_charge_kw=0.5 * c, max_discharge_kw=0.5 * c,
                                 efficiency=0.95, initial_soc_kwh=0.5 * c, target_final_soc_kwh=0.5 * c)
    if want["evs"] < native["evs"]:
        for name in ("ev2", "ev1")[: native["evs"] - want["evs"]]:
            ev = copy.deepcopy(getattr(h, name))
            for f in ("capacity_kwh", "max_charge_kw", "max_discharge_kw", "initial_soc_kwh",
                      "target_final_soc_kwh", "distance_km", "trip_consumption_kwh", "public_charger_power_kw"):
                setattr(ev, f, 0.0)
            ev.max_discharge_profile = np.zeros(96) if ev.max_discharge_profile is not None else None
            setattr(h, name, ev)
    elif want["evs"] > native["evs"]:
        raise SpecError("a home can only keep or remove the cars it has in the dataset")
    if abs(want["load_scale"] - 1.0) > 1e-9:
        h.load_profile_kw = h.load_profile_kw * want["load_scale"]
    return h


def build_community(spec: Dict) -> Tuple[object, object, List[Dict]]:
    """Return (CommunityTensors, EnvConfig, per-home description) for a specification."""
    if not isinstance(spec, dict):
        raise SpecError("the specification must be an object")
    n = _num(spec, "n_households", int)
    seed = spec.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 9999:
        raise SpecError("seed must be a whole number between 0 and 9999")
    feeder = _num(spec, "feeder_kw_per_home")
    export = _num(spec, "export_kw_per_home")
    _, _, env_cfg, pv_shape = _base()
    homes = _homes(n, seed)
    given = spec.get("houses") or []
    if not isinstance(given, list):
        raise SpecError("houses must be a list")
    out = []
    for i, h in enumerate(homes):
        native = _native(h)
        want = dict(native)
        if i < len(given) and given[i]:
            if not isinstance(given[i], dict):
                raise SpecError(f"home {i + 1}: each home must be an object")
            for k in ("pv_kw", "battery_kwh", "evs", "load_scale"):
                if k in given[i]:
                    try:
                        want[k] = _num(given[i], k, int if k == "evs" else float)
                    except SpecError as e:
                        raise SpecError(f"home {i + 1}: {e}") from None
        out.append(_apply(h, want, native, pv_shape))
    com = CommunityData(households=out); com.compute_aggregates()
    ct = build_tensors(com)
    cfg = replace(env_cfg, feeder_import_kw_per_hh=feeder, feeder_export_kw_per_hh=export,
                  offline_import_cap_kw=feeder)
    from .recorder import households
    meta = households(ct)
    for m, h in zip(meta, homes):
        m["source_id"] = int(h.player_id)
    return ct, cfg, meta
