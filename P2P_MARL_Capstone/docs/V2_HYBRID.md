# Hybrid execution: central MAT with a decentralised fallback

MAT is the cheapest controller, but it is centrally executed: the community
operator needs every household's data and sends every household its setpoint.
If that link fails, a purely central design has no answer. This layer adds the
answer and measures what it costs.

All numbers: `python scripts/eval_hybrid.py` → `results/v2/hybrid/hybrid_table.txt`
(about 3 min). Tests: `tests/test_v2_hybrid.py`, `tests/test_v2_platform.py`.

## Architecture

```
 operator (central)                         each household (local, always on)
 ------------------                         ---------------------------------
 MAT over the whole community   ── link ──▶  apply MAT setpoint             [all reachable]
 outage-aware community shield               ── or, as soon as ANY household is cut off ──
 reserves 8 kW per cut-off household         run own MAPPO student on own meter data
                                             cut-off households: import ≤ 8 kW (envelope)
                                             own per-vehicle deadline shield (always)
 last resort: rule controller (no model needed)
```

Three layers of fallback, cheapest first:

1. **Central MAT** while every household is reachable (`MODE_CENTRAL`).
2. **Local MAPPO student** — the decentralised policy trained by the same DAgger
   pipeline. It needs only the household's own observation, so it runs on the
   home energy manager without the link (`MODE_LOCAL`).
3. **Rule controller** when no learned model is available (`MODE_RULE`).

### Why the whole community switches, not only the cut-off households

The first design kept MAT for reachable households and fed it the last known
data for the cut-off ones. Measured, this is worse than switching everyone to the
local policy: MAT's joint decision is built on a stale picture, and the reachable
households then coordinate around a plan the cut-off ones are not following.

Threshold chosen on **validation days** (seed 777000, offline subsets 1–3, never
the reporting day or its subset) by `python scripts/select_degrade.py`:

| Share cut off | Households | Keep MAT for the rest | Everyone local |
|---:|---:|---:|---:|
| 4% | 1 | 118.23 | **116.44** |
| 12% | 3 | 123.03 | **116.70** |
| 20% | 5 | 123.32 | **116.69** |
| 30% | 8 | 129.33 | **118.91** |
| 50% | 12 | 128.41 | **119.46** |

Switching everyone won at every share, down to a single household, so the rule is
"any household unreachable → everyone local" (`DEFAULT_DEGRADE_FRAC = 0.04` in
`src/control/hybrid.py`). This is a negative result about MAT: **it is brittle to
missing inputs**, and the hybrid design is what makes it usable.

### The fallback envelope (the safety guarantee)

When the operator cannot reach a household it cannot know what that household
will draw. Without a limit, a cut-off household that delays its cars to the last
moment can collide with the operator's plan for the others: an adversarial
fallback caused **111.6 kWh unserved** in a full outage, and even the rule
controller caused 18.5 kWh (negative control in `tests/test_v2_hybrid.py`).

The fix is the same idea as an Australian *default operating envelope*
(CSIP-AUS: when a site loses contact with the network operator, it reverts to a
pre-agreed fixed limit):

* a cut-off household caps its own import at 8 kW (`offline_import_cap_kw`), and
  its own deadline shield plans its cars' minimum charge inside that cap;
* the operator reserves 8 kW of feeder capacity for each cut-off household and
  plans the reachable ones in what is left (outage-aware community shield,
  `src/env/safety.py`).

With 25 households × 8 kW = 200 kW = the feeder limit, the reservation is always
feasible. Result: **zero unserved energy and zero EV shortfall** for every outage
pattern tested, **including an adversarial fallback** that defers every car
(`test_envelope_keeps_the_feeder_safe_under_an_adversarial_fallback`, 4 patterns).

## Results (reporting day, battery-neutral €/day, mean of 3 seeds [worst seed])

| Scenario | Proposed | Offline only on MAPPO | Offline hold last setpoint | Offline on rule controller | No envelope (ablation) | Rule controller |
|---|---:|---:|---:|---:|---:|---:|
| No outage | **112.00** [114.33] | 112.00 | 112.00 | 112.00 | 112.00 | 143.90 |
| All cut off 17:00–21:00 | **112.43** [114.78] | 112.43 | 127.33 | 126.94 | 112.44 | 143.90 |
| One household cut off all day | **117.32** [118.25] | 119.24 | 122.77 | 120.10 | 116.06 | 143.91 |
| 30% cut off all day | **118.61** [120.54] | 137.06 | 155.49 | 144.34 | 116.06 | 143.90 |
| Operator down all day | **123.55** [125.30] | 123.55 | 173.84 | 143.86 | 116.06 | 143.86 |

Unserved energy and EV shortfall are 0.00 in every cell. On 8 unseen noisy days
(seed 424242) the proposed controller costs €112.42 / 112.95 / 116.96 / 118.80 /
123.54 for the five scenarios.

What the table says:

* **No outage costs nothing**: the hybrid is exactly MAT (tested).
* **The fallback matters more than MAT does.** Holding the last setpoint — what a
  naive central system does — costs €5–50/day more than the proposed fallback.
* **The price of the guarantee** is the gap between the proposed column and the
  no-envelope ablation: €0 for an evening outage, €1.26–7.49/day for longer ones.
  In these scenarios the ablation happened to stay safe, but the negative control
  shows it is not guaranteed to; we pay up to €7.49/day for a guarantee.
* With the operator down all day the community still saves €52/day against
  grid-only and €20/day against the rule controller.

## Limitations

* Outages are simulated; real communication failure modes (delays, partial
  data, wrong data) are not modelled. The local fallback needs the household's
  own meter and the model on the home energy manager.
* The envelope (8 kW) is a fixed number chosen so the reservation always fits
  the 200 kW feeder; a real network operator would set it per connection.
* A faulty or malicious operator is not covered; only loss of contact is.
* Same single-day limitation as everything else in v2.
