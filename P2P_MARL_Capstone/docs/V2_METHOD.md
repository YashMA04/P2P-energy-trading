# v2: Optimisation-guided multi-agent learning with certified feeder safety and core-stable settlement

This section replaces "constrained MAPPO" as the method. Results are in
`docs/V2_RESULTS.md` (generated numbers) and `results/v2/final_table.txt`.

## 1. What the v1 evidence actually showed (Phase 0)

Measured on the committed v1 checkpoints, before any new method was built:

1. **Training collapsed** in 4/5 seeds 10–50 updates after the trust region
   opened (κ = 1). Final checkpoints cost €147–195/day and all five failed the
   gate. The reported €144.07 came from checkpoints selected *on the reporting
   day itself* (updates 30–120) — selection on the test set, since there is one day.
2. **The headline included battery drawdown.** Every MAPPO checkpoint ended the
   day with all batteries at minimum SoC (−65.8 kWh) while the grid-only
   counterfactual keeps that energy. Battery-neutral, v1 MAPPO recovered
   28.5–35.9% of the achievable saving, the rule controller 29.2%.
3. **The LP bound was not the simulator's problem.** It priced exchange at the
   mean tariff, omitted degradation, counted starting inventory as free, allowed
   a cheap-tariff household to import and resell to peers (562 kWh/day) and
   allowed EV discharge below the trip requirement. Exact, simulator-consistent
   bound: **€98.74/day**; its schedule, replayed through the simulator, settles
   at **€99.72** (the gap is the 1% P2P fee, which the bound omits).
4. **The auction is not where coordination value lies.** Physical dispatch is
   settled before the market clears; the clearing price is a transfer. Replacing
   learned quotes with the rule's quotes changed cost by −€0.76…+€0.03/day.
5. **Seed 4's unserved energy was a shield defect, not a multiplier problem**:
   41 vehicles were shielded simultaneously at 23:30 against the shared
   end-of-day deadline.

## 2. The method, layer by layer

| Layer | Component | Adopted from | New here |
|---|---|---|---|
| Optimisation | Exact dispatch MILP (per-household tariffs, degradation, battery-neutral, importer/exporter and EV-floor binaries); replay-verified in the simulator | MILP dispatch is standard | Simulator-consistency as a *tested invariant* (`test_exact_schedule_is_attainable_in_the_simulator`) |
| Safety (hard) | **Community feasibility shield**: as-late-as-possible allocation of future feeder headroom across the whole fleet; the residual that cannot be placed later is firm now. Only the *minimum* is firm. | Least-laxity-first / deadline scheduling | Fleet-level certificate inside a MARL action pipeline; fixes three v1 shield defects |
| Learning (teacher) | Privileged teacher: exact MILP on 96 perturbed days (reporting day excluded) | Teacher–student distillation | — |
| Learning (student) | **Multi-Agent Transformer** decoding households in **least-laxity order**; separate critic; per-household advantages | MAT (Wen et al., 2022) | Laxity order (serialised with the checkpoint); separate critic to avoid the v1 gradient-coupling defect |
| Learning (closed loop) | **DAgger**: roll out the student, relabel visited states with the MILP re-solved from that state, accepting only proven-optimal plans | DAgger (Ross et al., 2011) | Proven-gap acceptance for MILP experts |
| Learning (RL) | PPO fine-tuning of the MAT student with a KL anchor to the frozen DAgger policy, critic warm-up, per-minibatch KL guard; kept only if it wins on validation days | PPO; DAPG-style regularisation | €113.84 → €112.00 (helps 2 of 3 seeds; neutral on the MAPPO student) |
| Execution | **Hybrid**: MAT centrally; on any communication loss every household switches to its local MAPPO student; cut-off households obey a fixed import envelope | Dynamic operating envelopes / CSIP-AUS default limits | Outage-aware community shield; switching rule chosen on validation days (`docs/V2_HYBRID.md`) |
| Economics | **Owen core allocation** from the LP duals; realised savings shared in proportion to core savings | Owen (1975) linear production games | Individual rationality *by construction* for any policy whenever the community saves; verified on 43 coalitions |
| Evaluation | Battery-neutral cost; validation-day checkpoint selection; MPC comparator; final *and* selected checkpoints | Agarwal et al. (2021) statistics | — |

### Why MAT fits this problem (measured, not argued)

The optimal community schedule is **non-unique**: across perturbed days that
differ by 2% noise, 16–31% of the teacher's label variance at a fixed
(household, time) is day-to-day, while the day's cost moves by €0.35. Many
households are interchangeable for a given job. A parameter-shared per-household
policy (MAPPO) regresses to the *average* of these interchangeable choices — every
EV trickle-charges — and closed-loop cost explodes (€210.70/day from behaviour
cloning). An autoregressive joint policy can represent "household *i* charges,
so household *j* does not": MAT's behaviour-cloned student alone reaches
**€122.95/day** on the same data. The coordination MAT provides is over the
shared feeder and the joint schedule, not over auction quotes.

### Decoding order: hypothesis tested and falsified

The proposal hypothesised that decoding the most constrained (least-laxity)
households first would beat a generic order, as in least-laxity-first
scheduling. At the imitation stage, over three seeds, a fixed *random*
permutation was as good or slightly better (mean €121.56 vs €125.36/day
battery-neutral; see `results/v2/final_table.txt`, imitation-stage block).
The domain-informed order is therefore **not** claimed as a contribution; it is
kept as the default only because it is deterministic and interpretable. What
matters is the autoregressive joint policy itself: every MAT seed beats every
MAPPO seed at the same stage. This agrees with the 2026 order-independent MAT
work (CMAT), which argues order sensitivity is a weakness to remove rather than
a lever to tune.

## 3. Honest scope

* **One day of data.** Every perturbed day is the same day with 2% load / 4% PV
  noise; generalisation to other days, seasons or weather is untested.
* **Centralised execution.** MAT needs all observations and the preceding
  households' actions at decision time — a community-operator-run controller,
  not a decentralised one. The decentralised MAPPO student is the fallback when
  communication fails (`docs/V2_HYBRID.md`); outages are simulated, not measured.
* **Feeder model** is a single aggregate capacity; there is no topology, so no
  graph structure is claimed.
* **The shield assumes a perfect base-load forecast inside the day**;
  `community_shield_margin_kw` reserves headroom against forecast error.
* **MPC** sees the true profile over its 3-hour window (more information than the
  agents' three forecast points); the oracle bound is not causally achievable.
* **Settlement misreporting.** The settlement is computed from household data;
  a household under-reporting load by 20% gains up to €0.65/day. The
  mitigation is to compute it from metered data.
