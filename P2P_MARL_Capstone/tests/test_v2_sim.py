"""Simulation variants for the stakeholder platform (src/platform/sim.py).

Every controller variant runs on the study community and on custom ones, every
outage scenario parses, the safety layer holds for all of them, and the
comparison ranks the variants consistently with the evaluation scripts.
"""
from __future__ import annotations

import pytest

from src.platform import sim
from src.platform.custom import build_community, default_spec, SpecError


@pytest.fixture(scope="module")
def models():
    return sim.load_models(seed=0)


@pytest.fixture(scope="module")
def study():
    return build_community(default_spec())


def test_registry_lists_the_variants_with_descriptions():
    keys = [c["key"] for c in sim.controllers()]
    assert keys[0] == "hybrid"
    assert set(keys) == {"hybrid", "mat_mixed", "mat_rule", "mat_stale", "mappo", "rule"}
    assert all(c["label"] and c["description"] for c in sim.controllers())


@pytest.mark.parametrize("scn,expect_off", [
    ({"kind": "none"}, {0: 0, 70: 0}),
    ({"kind": "window", "start": 68, "end": 84}, {67: 0, 68: 25, 83: 25, 84: 0}),
    ({"kind": "homes", "share": 0.3, "seed": 0}, {0: 8, 95: 8}),
    ({"kind": "homes", "share": 0.3, "seed": 0, "start": 40, "end": 60}, {39: 0, 40: 8}),
    ({"kind": "full"}, {0: 25, 95: 25}),
    ("evening", {68: 25, 90: 0}),
])
def test_scenarios(scn, expect_off):
    s = sim.schedule(scn, 25)
    for t, k in expect_off.items():
        assert (~s.online(t)).sum() == k, (scn, t)


@pytest.mark.parametrize("bad", [{"kind": "moon"}, {"kind": "window", "start": 50, "end": 40},
                                 {"kind": "homes", "share": 1.5}, "nope"])
def test_bad_scenarios_are_rejected(bad):
    with pytest.raises(SpecError):
        sim.schedule(bad, 25)


def test_hybrid_on_the_study_community_matches_the_reported_run(models, study):
    ct, cfg, _ = study
    frames, summary = sim.run_day(ct, cfg, "hybrid", "normal", models)
    assert len(frames) == 96
    # seed-0 final MAT on the reporting day (results/v2/hybrid/hybrid_table.json, worst seed column)
    assert abs(summary["cost_battery_neutral"] - 114.33) < 0.05
    assert summary["unserved_kwh"] < 1e-3 and summary["ev_shortfall_kwh"] < 1e-3


@pytest.mark.parametrize("key", ["hybrid", "mat_mixed", "mat_rule", "mat_stale", "mappo", "rule"])
def test_every_variant_is_safe_on_a_custom_community_with_an_outage(models, key):
    spec = default_spec(n=10, seed=3)
    spec["houses"][0].update(pv_kw=6.0, battery_kwh=12.0)
    spec["houses"][1].update(load_scale=1.8, evs=0)
    ct, cfg, _ = build_community(spec)
    frames, summary = sim.run_day(ct, cfg, key, {"kind": "homes", "share": 0.3, "seed": 1}, models)
    assert len(frames) == 96 and len(frames[0]["hh"]) == 10
    assert summary["unserved_kwh"] < 1e-3 and summary["ev_shortfall_kwh"] < 1e-3
    assert all(f["feeder_kw"] <= f["limit_kw"] + 1e-3 for f in frames)


def test_compare_ranks_variants_and_reports_reference_costs(models, study):
    ct, cfg, _ = study
    res = sim.compare(ct, cfg, "evening", models, keys=["hybrid", "mat_stale", "rule"])
    rows = {r["key"]: r for r in res["rows"]}
    assert res["grid_only"] > rows["rule"]["cost_battery_neutral"] > rows["hybrid"]["cost_battery_neutral"]
    assert rows["mat_stale"]["cost_battery_neutral"] > rows["hybrid"]["cost_battery_neutral"]
    assert [r["rank"] for r in sorted(res["rows"], key=lambda r: r["cost_battery_neutral"])] == [1, 2, 3]
    assert all(0 <= r["saving_pct"] <= 100 for r in res["rows"])


def test_live_hooks_cut_the_link_mid_day(models, study):
    ct, cfg, _ = study
    state = {"cut": False}
    gen = sim.simulate(ct, cfg, "hybrid", "normal", models, cut=lambda: state["cut"])
    frames = []
    for kind, payload in gen:
        if kind == "frame":
            frames.append(payload)
            if payload["t"] == 9:
                state["cut"] = True
    assert frames[5]["online"] == 1.0 and all(f["online"] == 0.0 for f in frames[10:])
