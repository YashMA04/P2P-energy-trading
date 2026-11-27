"""Custom communities for the stakeholder platform (src/platform/custom.py).

The default specification must reproduce the study community exactly, and every
override must land in the simulator's tensors where the simulator reads it.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.platform import custom
from src.platform.custom import SpecError, build_community, default_spec


@pytest.fixture(scope="module")
def study():
    return custom.study_community()


def test_default_spec_reproduces_the_study_community(study):
    ct0, cfg0 = study
    ct, cfg, meta = build_community(default_spec())
    assert ct.n_agents == 25 and len(meta) == 25
    for name in ("load_kw", "pv_kw", "buy_price", "sell_price", "has_pv", "has_bess", "bess_capacity",
                 "bess_charge_max", "bess_soc0", "ev_capacity", "ev_charge_max", "ev_at_home", "ev_departure"):
        np.testing.assert_allclose(getattr(ct, name), getattr(ct0, name), err_msg=name)
    assert cfg.feeder_import_kw_per_hh == cfg0.feeder_import_kw_per_hh == 8.0
    assert cfg.offline_import_cap_kw == 8.0


def test_default_spec_is_json_friendly_and_describes_each_home():
    import json
    spec = default_spec()
    json.dumps(spec)
    assert spec["n_households"] == 25 and len(spec["houses"]) == 25
    h = spec["houses"][0]
    assert set(h) >= {"pv_kw", "battery_kwh", "evs", "load_scale"}


@pytest.mark.parametrize("n", [4, 12, 40])
def test_community_size(n):
    ct, cfg, meta = build_community(default_spec(n=n))
    assert ct.n_agents == n and ct.load_kw.shape == (96, n)


def test_seed_changes_which_homes_are_drawn():
    a = build_community(default_spec(n=12, seed=0))[0]
    b = build_community(default_spec(n=12, seed=1))[0]
    assert not np.array_equal(a.player_ids, b.player_ids)


def test_overrides_reach_the_simulator():
    spec = default_spec(n=6)
    no_pv = next(i for i, h in enumerate(spec["houses"]) if h["pv_kw"] == 0)
    with_pv = next(i for i, h in enumerate(spec["houses"]) if h["pv_kw"] > 0)
    spec["houses"][no_pv].update(pv_kw=5.0, battery_kwh=10.0, evs=1, load_scale=2.0)
    spec["houses"][with_pv].update(pv_kw=0.0, battery_kwh=0.0, evs=0)
    base = build_community(default_spec(n=6))[0]
    ct, _, meta = build_community(spec)
    # solar added to a home without it: a real PV shape scaled to the requested peak
    assert ct.has_pv[no_pv] == 1 and abs(ct.pv_kw[:, no_pv].max() - 5.0) < 1e-4
    assert ct.has_bess[no_pv] == 1 and abs(ct.bess_capacity[no_pv] - 10.0) < 1e-4
    assert abs(ct.bess_charge_max[no_pv] - 5.0) < 1e-4            # 0.5 C, the dataset median
    assert (ct.ev_charge_max[:, no_pv] > 0).sum() == 1
    np.testing.assert_allclose(ct.load_kw[:, no_pv], 2 * base.load_kw[:, no_pv], rtol=1e-5)
    # everything removed from a home that had it
    assert ct.has_pv[with_pv] == 0 and ct.pv_kw[:, with_pv].max() == 0
    assert ct.has_bess[with_pv] == 0 and ct.bess_charge_max[with_pv] == 0
    assert (ct.ev_charge_max[:, with_pv] > 0).sum() == 0 and ct.ev_trip_kwh[:, with_pv].sum() == 0
    assert meta[no_pv]["pv_kw"] == 5.0 and meta[with_pv]["evs"] == 0
    # untouched homes stay identical
    other = [i for i in range(6) if i not in (no_pv, with_pv)]
    np.testing.assert_allclose(ct.load_kw[:, other], base.load_kw[:, other])


def test_feeder_size_sets_both_limits_and_the_envelope():
    spec = default_spec(n=10); spec["feeder_kw_per_home"] = 6.0; spec["export_kw_per_home"] = 4.0
    _, cfg, _ = build_community(spec)
    assert cfg.feeder_import_kw_per_hh == 6.0 and cfg.feeder_export_kw_per_hh == 4.0
    assert cfg.offline_import_cap_kw == 6.0       # reservation always fits the feeder


@pytest.mark.parametrize("patch", [
    {"n_households": 2}, {"n_households": 99}, {"feeder_kw_per_home": 1.0},
    {"houses": [{"pv_kw": -1}]}, {"houses": [{"evs": 3}]}, {"houses": [{"load_scale": 9}]},
    {"houses": [{"battery_kwh": 100}]}, {"seed": "x"},
])
def test_invalid_specs_are_rejected_with_a_message(patch):
    spec = default_spec(n=4)
    if "houses" in patch:
        spec["houses"][0].update(patch["houses"][0])
    else:
        spec.update(patch)
    with pytest.raises(SpecError) as e:
        build_community(spec)
    assert str(e.value)


def test_short_house_list_is_padded_with_defaults():
    spec = default_spec(n=8); spec["houses"] = spec["houses"][:3]
    ct, _, meta = build_community(spec)
    assert ct.n_agents == 8 and len(meta) == 8
