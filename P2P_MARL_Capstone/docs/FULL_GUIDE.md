# The Complete Guide: Coordinating a Neighbourhood Energy Community with Learning, Optimisation and Game Theory

This guide explains the whole project from the beginning. It assumes no
background in energy systems or machine learning. Each idea is introduced in
plain words first, then made precise, and every design choice says **why** it
was made and **what else was considered**. Part 1 builds up the problem; Part 2
explains the machine-learning method stage by stage; Part 3 explains the
stakeholder platform; Part 4 maps everything to code, tests and results.

Short versions of the same material: `docs/EXPLAINER.md` (two-part summary),
`docs/V2_METHOD.md` (method only), `docs/V2_RESULTS.md` (numbers only). How to
install and run everything: `docs/SETUP.md`.

---

# Part 1. The problem

## 1.1 A neighbourhood that makes and stores its own energy

Picture a street of 25 homes. During the day, 20 of them have solar panels
that often produce more electricity than the home uses. 15 have a home
battery that can store that surplus. Every home has two electric cars that
leave in the morning and must be charged before they leave again. All 25
homes hang off one cable from the local transformer, called the **feeder**.

Three facts make this interesting:

1. **Electricity costs different amounts at different times.** Each home pays
   a time-of-use tariff: cheap at night, expensive in the evening peak. Selling
   surplus back to the grid pays much less than buying.
2. **Neighbours can trade.** If home 3 has spare solar at noon and home 7 needs
   power, it is better for both if 3 sells to 7 at a price between the grid's
   buy and sell prices. This is **peer-to-peer (P2P) trading**. Here it runs as
   a small market every 15 minutes (a uniform-price double auction: buyers and
   sellers submit offers, one clearing price is set, matched energy changes hands).
3. **The cable has a limit.** The feeder can carry 200 kW in (8 kW per home on
   average) and 125 kW out. If every home charges two cars at full speed at
   7 pm, the cable is overloaded. Real networks either reinforce the cable
   (expensive) or manage when things charge.

## 1.2 The decision every 15 minutes

The day is split into 96 steps of 15 minutes. At every step each home decides
five numbers:

* how hard to charge or discharge its battery;
* how hard to charge (or, for vehicle-to-grid, discharge) each of its two cars;
* how much energy to offer or bid in the neighbour market, and at what price.

Multiply that by 25 homes and 96 steps: 12,000 coupled decisions a day. They
are coupled because one home's choice changes the market price, the cable load
and what is left for everyone else.

## 1.3 What "doing well" means

Four requirements, in priority order:

1. **Safety.** Nobody's demand is cut off, every car reaches the charge its
   owner needs by departure, and the feeder is never overloaded. These are hard
   requirements: a cheaper day that strands a car is a failure.
2. **Cost.** The community's total electricity bill should be as low as possible.
3. **Fairness.** Every home should be better off by joining than by staying
   alone, otherwise homes leave and the community collapses.
4. **Robustness.** It must keep working when things go wrong: noisy forecasts,
   homes gaming the market, and loss of communication with the operator.

## 1.4 How we measure cost honestly

* **Grid-only baseline (€175.89/day).** Every home buys what it needs from the
  grid and sells its surplus to the grid, with no coordination. Any useful
  controller must beat this.
* **Exact optimum (€98.74/day).** A mathematical optimiser that knows the whole
  day in advance (perfect foresight) finds the cheapest possible plan. No real
  controller can beat it, because real controllers do not know the future.
* **Share of possible saving.** A controller costing C recovers
  (175.89 − C) / (175.89 − 98.74) of the saving that coordination could
  possibly achieve. 0% = no better than grid-only, 100% = optimal.
* **Battery-neutral cost.** A controller could look cheap by emptying every
  battery during the day and leaving them flat. We charge for that: any
  battery energy used up by the end of the day is bought back at the day's
  cheapest tariff. All reported costs use this.
* **Seeds and validation.** Neural networks depend on random initialisation.
  Every learned method is trained with 3 seeds and we report the mean and the
  spread. Model choices are made on separate **validation days**, never on the
  reporting day, so the reported number is not cherry-picked.

---

# Part 2. The machine-learning method

## 2.1 The options, and why none is enough on its own

| Approach | What it is | Strength | Why not alone |
|---|---|---|---|
| Rule controller | Fixed if-then rules ("charge when cheap") | Simple, transparent, safe with a shield | Ignores neighbours and the cable; €143.90/day, 41% of the possible saving |
| Exact optimisation (MILP) | Solve the whole day as one equation system | Provably optimal | Needs the future; takes seconds to minutes; cannot run on a home device |
| Model-predictive control (MPC) | Optimise a short window ahead using forecasts, repeat every step | Industry standard, uses the optimiser | Short horizon misses the evening peak; slow solves; €135.93/day, 52% |
| Single-agent RL | One neural network controls all homes | Learns from experience | Action space grows with the number of homes; does not scale |
| Independent RL | Each home learns alone | Scales | Homes chase each other's moves; unstable; learns poorly |
| MAPPO (v1) | Multi-agent PPO: homes share training information, act on their own data | Standard cooperative MARL | Learns slowly from cost signals alone; v1 final model €173.43/day, 3% |
| MAT | Transformer that decides homes one after another | Coordinates within each step | Still needs a good training signal |
| Graph neural networks | Learn on the network topology | Natural for grids | Our feeder is a single radial cable; nothing for a graph to exploit; not claimed |

The project's answer is to **combine** them so that each covers the others'
weaknesses: the optimiser provides perfect training signals, the neural
network provides fast decisions without the future, RL polishes the network on
the real objective, the safety layer guarantees the hard constraints, game
theory makes it fair, and a decentralised fallback keeps it running when
communication fails.

## 2.2 The pipeline at a glance

```
                  TRAINING (offline)                                DEPLOYMENT (every 15 minutes)
  ┌──────────────────────────┐   labels   ┌────────────────┐    ┌──────────────────────────────────┐
  │ Exact optimiser (MILP)   │──────────▶│ DAgger          │    │ MAT decides for all homes        │
  │ sees the future, solved  │           │ imitation       │──▶ │   (if any home is cut off:       │
  │ from states the student  │◀──states──│ MAT + MAPPO     │    │    every home runs MAPPO locally) │
  │ actually visits          │           │ students        │    │ Safety layer corrects actions    │
  └──────────────────────────┘           └───────┬────────┘    │ Market clears                     │
                                                 │ PPO         │ Fair settlement at day's end      │
                                                 ▼ fine-tune   └──────────────────────────────────┘
                                          kept only if better
                                          on validation days
```

## 2.3 The simulator (the "environment")

**What.** A program (`src/env/community_env.py`) that plays out a day: given
every home's five decisions at a step, it applies the battery and car physics,
clears the neighbour market, settles with the grid, checks the feeder, and
returns the new state and the cost.

**Why it matters.** Learning needs millions of trial decisions; you cannot
experiment on real homes. Everything (learning, optimisation, the platform)
uses this one simulator so the numbers are comparable.

**Data.** A 250-home dataset (`data/EC_EV_dataset.xlsx`) with demand, solar,
battery and car profiles. The study community is 25 homes drawn from it in a
way that keeps the mix of owners representative (stratified sampling, seed 0).

## 2.4 Stage 1 — the exact optimiser (benchmark and teacher)

**What.** A mixed-integer linear program (MILP) in `src/opt/lp.py`. Linear
program = minimise a cost subject to linear constraints; "mixed-integer" adds
yes/no variables, here "is this home importing or exporting at this step?" and
"is this car allowed to discharge?".

**Why mixed-integer.** v1 used a plain linear program. It let a home buy from
the grid and sell to a neighbour in the same step (562 kWh of paper arbitrage)
and let cars discharge below their safety floor. Those plans cannot happen in
the simulator, so the "optimum" was wrong. The yes/no variables remove them.

**Two jobs.** It gives the honest lower bound (€98.74) and it is the **teacher**
that tells the neural network what a perfect decision would have been.

**Proven only.** The solver stops at a time limit. A plan is used as a label
only when the solver *proves* it optimal (gap ≤ 0.1%); an unproven plan once
cost €134.52 where the true optimum was €7.24 on a partial day.

## 2.5 Stage 2 — the safety layer (shields)

**What.** Two functions that look at any proposed action and correct it before
the simulator runs it (`src/env/safety.py`).

* **Per-vehicle shield.** For each car: if it does not charge now, can it still
  reach its departure target? If not, force the minimum charge needed.
* **Community shield.** All cars share one cable. It plans every car's minimum
  charging **as late as possible** inside the cable's future spare capacity,
  and makes that minimum firm now only when waiting would make the cars
  collide later. So the cars can never jointly overload the feeder.

**Why a shield instead of a penalty.** A penalty ("lose points if a car is
short") teaches a network to *usually* avoid violations. A shield makes
violations impossible regardless of what the network does. For a system that
controls people's cars and a public network, "usually" is not acceptable.
Because of the shield, every controller in this project (even a random one)
is safe; learning only decides how cheap it is.

**What was fixed.** v1's shield made the whole setpoint firm (not just the
minimum), asked for more energy than needed, and was off by one at the last
step. Fixing them is the main reason the rule controller went from €153.35 to
€143.90.

## 2.6 Stage 3 — learning from the teacher (imitation and DAgger)

**Plain idea.** Instead of letting the network discover good behaviour by trial
and error (slow), show it what the perfect planner would do and ask it to copy.

**Behaviour cloning (tried first).** Record the optimiser on 96 perturbed
training days and train the network to reproduce its decisions. It failed
(€190–211/day). The reason is well known: the network makes a small mistake,
lands in a situation the teacher never visited, and has no idea what to do, so
errors compound.

**DAgger (what works).** "Dataset Aggregation": let the *student* drive the
simulator, and at each state it actually reaches, re-solve the optimiser from
that exact state to get the correct decision there. Add these labels to the
training set, retrain, repeat (4 rounds). The student learns how to recover
from its own mistakes. This is what took the method from failure to 80% of the
possible saving.

**Why imitation counts as part of an RL method.** The problem is a sequential
decision process; DAgger is the standard algorithm for learning a policy in one
from an expert. RL proper comes in Stage 5.

## 2.7 Stage 4 — two students: MAPPO and MAT

The same DAgger pipeline trains two network designs.

**MAPPO student** (`src/marl/mappo.py`). Each home has the same small network
(shared weights) that sees only that home's data and outputs its five
decisions. During training a "critic" sees the whole community, but at run
time each home acts alone. This is **centralised training, decentralised
execution (CTDE)**: it can run on each home's own device with no
communication. Result: €116.06/day (78%).

**Multi-Agent Transformer, MAT** (`src/marl/mat.py`). A transformer (the
architecture behind modern language models) treats the 25 homes as a sequence:

* the **encoder** reads every home's situation and lets each home "attend" to
  the others (who has surplus, whose car is urgent);
* the **decoder** decides one home at a time, each seeing the decisions already
  made for the homes before it, so later homes can adapt to earlier ones
  within the same 15-minute step.

The order is by EV urgency (least slack first). Result: €113.84/day (80%).

**Why MAT is better here.** Coordination inside a step matters (two homes
should not both grab the last spare cable capacity). MAPPO homes cannot see
each other's choices; MAT's decoder can. The cost is that MAT is a **central**
controller: it needs every home's data and sends every home its decision,
which is what Stage 8 deals with.

**An honest negative result.** We expected urgency order to matter. A random
fixed order did as well, so the benefit comes from joint attention and
sequential decoding, not the particular order.

## 2.8 Stage 5 — reinforcement-learning fine-tuning

**What RL is.** An agent acts, receives a reward (here: minus the cost), and
adjusts its behaviour to earn more reward over time. **PPO** (proximal policy
optimisation) is the most widely used algorithm: it improves the policy in
small, clipped steps so it does not collapse.

**Why fine-tune after imitation.** The teacher optimised a slightly simplified
picture (perfect foresight, idealised market); the student must perform in the
real simulator with noise. RL lets it adjust to what actually happens.

**How it is kept safe from forgetting.** Starting PPO from a good network can
destroy it in a few updates. Three guards:

* a **KL anchor** penalises drifting away from the DAgger policy;
* a **critic warm-up** trains the value estimate first, before the policy moves;
* the fine-tuned model **replaces the DAgger model only if it is cheaper on
  validation days**.

**Result.** Seed 0: €116.20 → €114.33. Seed 2: €111.85 → €108.21. Seed 1: no
validation gain, so the DAgger model was kept (its last update happened to be
cheaper on the reporting day, €111.23, but using that would be cherry-picking).
Mean €113.84 → **€112.00 (83% of the possible saving)**. RL adds a modest but
real gain; most of the work is done by the teacher.

## 2.9 Stage 6 — the market and fair settlement (game theory)

**The fairness problem.** The community saves money as a whole, but how much
of it should each home get? A home with a big battery that absorbed everyone's
noon surplus helped more than a home with nothing. If a home would pay less
alone, it will leave.

**Cooperative game theory** answers this. Treat every subset of homes
("coalition") as a team and ask how much that team could save on its own. An
allocation is in the **core** if no group of homes could do better by breaking
away. We compute each home's share from the optimiser's shadow prices (the
dual values: what one more kWh of each home's flexibility is worth), which
gives an **Owen allocation**, and check the core condition on 43 coalitions.

**Settlement.** Each home pays its grid-only bill minus its share of the
community's actual saving. It is budget-balanced (the shares add up to the
real total) and no home pays more than alone whenever the community saves.
The platform's Bills view shows this per home.

## 2.10 Stage 7 — strategic robustness

Homes are people; some may try to game the market by shading their bids or
withholding supply. We measured how much a single home could gain by doing so
(`scripts/strategic_eval.py`). Settling on **metered** energy rather than on
what homes report removes the gain from misreporting quantities.

## 2.11 Stage 8 — hybrid execution with fallback

**The question.** What happens if the operator's link to some homes fails (a
router, a cloud outage, a cut fibre)?

**Three designs were compared:**

1. *Fully central, no fallback.* Cut-off homes repeat the last instruction.
   Measured: up to €173.84/day with the operator down, as bad as no
   coordination.
2. *Keep MAT for reachable homes, MAPPO for cut-off ones.* Sounds sensible, but
   MAT then plans on stale data for part of the community. On validation days it
   lost to option 3 at every share of cut-off homes tested, down to a single
   home (`scripts/select_degrade.py`).
3. **Proposed: switch the whole community to local MAPPO as soon as anyone is
   cut off.** MAPPO needs no communication, so it degrades gracefully.

**The fallback envelope.** When the operator cannot see a home, it cannot know
what that home will draw. Without a limit, a cut-off home that charges late
could collide with the operator's plan for everyone else; a deliberately bad
fallback caused 111.6 kWh of unserved demand in a test. So each cut-off home
caps its own import at 8 kW and the operator reserves 8 kW of cable for it
(25 × 8 kW = the full 200 kW, so the reservation always fits). This is the
same idea as Australia's **default operating envelopes** in the CSIP-AUS
standard, where a site that loses contact with the network falls back to a
pre-agreed fixed limit.

**Results** (reporting day, mean of 3 seeds, zero unserved energy everywhere):

| Scenario | Proposed | MAT, no fallback | Rule controller |
|---|---:|---:|---:|
| No outage | €112.00 | €112.00 | €143.90 |
| Everyone cut off 17:00–21:00 | €112.43 | €127.33 | €143.90 |
| One home cut off all day | €117.32 | €122.77 | €143.91 |
| 30% cut off all day | €118.61 | €155.49 | €143.90 |
| Operator down all day | €123.55 | €173.84 | €143.86 |

The envelope costs €0–7.49/day in long outages: the price of a guarantee.

## 2.12 Why this combination is the best approach for this problem

| Requirement | What delivers it | Evidence |
|---|---|---|
| Safety (cars, demand, cable) | Shields + fallback envelope | Zero unserved energy and zero shortfall on every seed, day and outage, including an adversarial fallback |
| Low cost without knowing the future | Teacher (MILP) + DAgger + MAT | 80% of possible saving vs 52% for MPC and 41% for rules |
| Uses the real objective | PPO fine-tuning, validation-gated | 80% → 83% |
| Fast decisions | A neural network forward pass (milliseconds) | vs seconds-to-minutes for MPC solves |
| Fairness | Core-stable Owen settlement | Every home better off whenever the community saves |
| Resilience | Decentralised MAPPO fallback | €112–124/day in every outage, vs €144 rules and up to €174 central-only |
| Honesty | Validation-only selection, 3 seeds, unseen test days | Unseen noisy days: €112.42, matching the reporting day |

Each component exists because removing it breaks one requirement: without the
teacher the network learns poorly (v1); without RL it misses the last few
percent; without the shield safety is only statistical; without the
settlement homes may leave; without the fallback an outage is costly.

**How it relates to practice.** Real deployments already use pieces of this:
network operators in Australia publish dynamic operating envelopes and
default limits for lost contact; Germany's §14a EnWG lets operators curtail
controllable devices such as chargers to 4.2 kW in exchange for lower network
charges; EU energy-community rules let members share energy with allocation
keys. Pure P2P trading trials (for example Power Ledger's in Australia) found
the economics hard on their own. This project's position is that coordination
plus a fair settlement, protected by envelopes, is the practical route, with
learning making the coordination fast.

## 2.13 What did not work (kept in the record)

* Behaviour cloning alone: €190–211/day.
* Predicting state-of-charge targets instead of power: €363–369/day.
* Urgency order in MAT versus random order: no difference.
* RL fine-tuning on seed 1: rejected by validation.
* MAT running on stale data during a partial outage: worse than switching to MAPPO.
* MPC accepting only proven solutions: €143.07 (worse than accepting good incumbents, €135.93).

Details: `docs/V2_DISCARDED.md`.

## 2.14 Limitations

* One day of data plus noisy variants; no seasons.
* Trained on one 25-home community. Other communities run in the platform as a
  generalisation test, not a trained result.
* Outages are simulated as loss of contact only.
* The market is a simulation of a uniform-price auction, not a regulated product.

---

# Part 3. The stakeholder platform

## 3.1 Why build one

The people who decide whether a community adopts a controller (residents, a
network operator, a regulator, an examiner) will not read Python. They need to
**see** what the controller does, **try** what-if cases, and **check** that
they are treated fairly. The platform is that window.

## 3.2 Two ways to open it

* **Replay** (no installation): open `platform/index.html` (3D) or
  `platform/classic.html` (2D) in a browser, or the published web page. It
  plays recorded days of the 25-home study community for all 6 controllers in
  all 5 preset scenarios (30 recorded days).
* **Live**: run `python platform/server.py` and open `http://localhost:8765`
  (3D) or `http://localhost:8765/classic.html` (2D). The simulator runs as you
  watch, with your own community and outage, and you can cut the link mid-day.

The 3D and 2D pages have the same features. The 3D page needs internet to load
its graphics library; the 2D page works offline and supports light and dark
themes.

## 3.3 What is on screen

**Top bar.** What is running (scenario, controller, community size) and the
actions: Set up, Compare, Bills.

**Status tiles.**

* *Saving vs grid-only*: money saved since midnight against every home buying
  alone, and as a percentage. It can be negative early in the day because
  batteries and cars are charged on cheap night power that is used later.
* *Feeder load*: current power on the cable and its share of the limit, plus
  the tariff now.
* *Control*: who is deciding (central MAT, local fallback, rule controller).
* *Demand served*: 100% unless any demand could not be met.

**3D scene** (3D page). Homes with solar panels that glow with real output, a
battery column that fills and empties, cars that leave and return; cables in
the street that light up with the power through them (blue towards homes, gold
back to the grid; brighter, wider and faster for more power; neighbour trades
show as gold from exporters feeding blue to importers while the main cable
dims); a ring around the transformer showing load against the limit; an
operator mast with a thin link to each roof, dark orange when cut; a ground
outline per home coloured by who controls it; a day-night sky.

**Charts and board** (2D page). Feeder load with the limit and outage periods;
running cost against grid-only; a tile per home with its control mode, net
power, solar, battery and car levels.

**Timeline.** Play, pause, speed, and a scrubber over the day's feeder curve.
In live mode, **Cut link** cuts the operator's communication to every home.

**Home detail.** Click a home: its equipment, control mode, net power, solar,
battery, each car (home, away, charging), its power over the day, and its bill.

## 3.4 The simulation variants and what each one maps to

| Platform option | Approach it runs | Code |
|---|---|---|
| MAT + MAPPO fallback (proposed) | Stage 4 MAT (fine-tuned, Stage 5) centrally; Stage 8 switch to MAPPO for everyone on any outage; envelope | `proposed_controller` in `src/control/hybrid.py` |
| MAT + MAPPO for cut-off homes only | Stage 8 design 2 (rejected by validation) | `HybridController(..., "local")` |
| MAT + rule fallback | MAT; cut-off homes use the rule controller | `HybridController(..., "rule")` |
| MAT, no fallback | Stage 8 design 1 (naive central) | `HybridController(..., "stale")` |
| MAPPO everywhere | Stage 4 MAPPO student, fully decentralised | `HybridController(mappo, mappo, ...)` |
| Rule controller | Section 2.1 baseline, with the v2 shield | `RuleAgent` / `TariffAwareP2P` |

Every variant runs through the same simulator, shields, market and
settlement, so differences come only from the controller.

**Scenarios.** Normal day; every home cut off 17:00–21:00; one home cut off all
day; 30% of homes cut off all day; operator down all day; and, in live mode, a
custom outage (every home or a chosen share, between any two times).

## 3.5 Building your own community (live mode)

Set up → Community lets you change:

| Setting | Range | What happens |
|---|---|---|
| Number of homes | 4–60 | Homes are drawn from the 250-home dataset keeping the owner mix |
| Draw number | 0–9999 | A different set of homes |
| Import / export limit per home | 4–17 / 2–17 kW | Feeder limits = this × homes; the import limit is also the outage envelope |
| Solar | 0–15 kW peak | The home's profile is rescaled; a home without solar gets the dataset's average solar shape |
| Battery | 0–30 kWh | 0.5 C power, 95% efficiency, half full at start and end (dataset medians) |
| Cars | 0 up to the recorded number | Removed cars stop driving and charging |
| Demand | ×0.25–×3 | Multiplies everyday demand |

"Set for every home" applies one value to all homes; changed cells are
highlighted; "Reset" restores the dataset.

**Important.** The models were trained on the 25-home study community. On a
different community they still run (MAT has no fixed number of homes and the
MAPPO student acts per home) and the shields still guarantee safety, but the
costs are a test of how well they generalise. The page says this whenever you
run a custom community.

## 3.6 Compare and Bills

**Compare** runs all six controllers on the same community and scenario and
ranks them by battery-neutral cost, with the saving against grid-only, the peak
feeder load and any unserved energy. Ties share a rank. **Watch** opens any of
them. **Bills** lists each home's grid-only bill, its settled bill and its
saving, with a check that every home is better off.

The platform uses the seed-0 models, so its numbers differ a little from the
3-seed means in Part 2 (a normal day is €114.33 instead of €112.00).

## 3.7 How the platform works inside

```
 browser (index.html or classic.html)            Python server (platform/server.py)
 ─────────────────────────────────────           ────────────────────────────────────
 Set up  ── POST /api/session {spec, scenario, controller} ──▶ build community (custom.py)
 Run day ── GET  /api/stream?session=…  ◀── one frame per step (Server-Sent Events) ── sim.py
 Cut link ─ GET  /api/control?session=…&cut=1 ──▶ the schedule switches every home offline
 Compare ── POST /api/compare {spec, scenario} ◀── table + all six runs
 Replay  ── data/traces.js (recorded by scripts/export_platform_data.py)
```

* `src/platform/custom.py` turns a JSON specification into the simulator's
  input and rejects invalid values with a message saying which field and why.
* `src/platform/sim.py` holds the six controllers, the scenarios, a generator
  that yields one frame per step, and the comparison.
* `src/platform/recorder.py` describes each step: feeder load, price, running
  costs, and per home its mode, battery and car levels, presence, net power and
  solar output; and the day's bills.
* The server uses only Python's standard library (no web framework to install).

---

# Part 4. Where everything is

## 4.1 Approach → code → tests → results

| Approach | Code | Tests | Result files / command |
|---|---|---|---|
| Simulator | `src/env/community_env.py` | `tests/test_all.py` | — |
| Exact optimiser | `src/opt/lp.py`, `src/opt/replay.py` | `test_v2_optimisation.py` | `scripts/phase0_diagnostics.py` |
| Shields | `src/env/safety.py` | `test_v2_shield.py` | — |
| Teacher data | `src/opt/teacher.py`, `scripts/make_teacher.py` | `test_v2_teacher.py` | `results/v2/teacher/` |
| DAgger | `src/opt/dagger.py`, `scripts/train_dagger.py` | `test_v2_mpc_distill.py` | `results/v2/dagger_mat/`, `dagger_mappo/` |
| MAPPO / MAT | `src/marl/mappo.py`, `src/marl/mat.py` | `test_v2_mat.py` | — |
| RL fine-tuning | `scripts/finetune_rl.py`, `scripts/run_ft_mat.sh` | `test_v2_platform.py` (selection rule) | `results/v2/ft_mat/` |
| MPC | `src/opt/mpc.py`, `scripts/run_mpc.py` | `test_v2_mpc_distill.py` | `results/v2/mpc_h12.json` |
| Settlement | `src/game/core.py` | `test_v2_game.py` | — |
| Strategic tests | `src/game/strategic.py`, `scripts/strategic_eval.py` | `test_v2_game.py` | `results/v2/strategic*.json` |
| Hybrid fallback | `src/control/hybrid.py`, `src/control/models.py` | `test_v2_hybrid.py` | `scripts/eval_hybrid.py`, `scripts/select_degrade.py` |
| Platform | `src/platform/`, `platform/` | `test_v2_custom.py`, `test_v2_sim.py`, `test_v2_platform.py`, `test_v2_platform_ui.py` | `platform/data/traces.js` |
| Final numbers | `scripts/final_eval_v2.py`, `scripts/compare_approaches.py` | — | `results/v2/final_table.txt`, `compare_report.txt` |

## 4.2 How the work was tested

Each piece was built in small steps: write the tests for the next piece, write
the code, run the tests, then run the whole suite before moving on. The full
suite has 131 tests (126 run and 5 browser tests skip when Playwright is not installed). Highlights:

* the default community specification reproduces the study community's data
  exactly, value by value;
* the platform's proposed controller reproduces the reported €114.33;
* every controller is safe on a custom community during an outage;
* an adversarial fallback cannot cause unserved energy with the envelope, and
  does without it (a negative control that proves the test can fail);
* a real browser runs the replay, a custom live community with a cable cut, a
  comparison and the bills, on both the 3D and 2D pages, with no errors.

## 4.3 Glossary

| Term | Meaning |
|---|---|
| Agent | A decision-maker; here, each home's controller |
| Policy | The rule (a neural network) mapping what an agent sees to what it does |
| Reinforcement learning (RL) | Learning a policy by trial and error to maximise reward |
| Multi-agent RL (MARL) | RL with several agents whose actions affect each other |
| CTDE | Centralised training, decentralised execution |
| Critic / value function | A network that estimates how good a situation is, used to train the policy |
| PPO | Proximal policy optimisation, a stable RL algorithm |
| MAPPO | PPO for multiple cooperating agents |
| MAT | Multi-Agent Transformer: decides agents in sequence with attention |
| Behaviour cloning | Copying an expert's recorded decisions |
| DAgger | Copying an expert at the states the learner itself visits |
| MILP | Mixed-integer linear program: optimisation with yes/no variables |
| MPC | Model-predictive control: re-optimise a short window every step |
| Shield | A safety filter that corrects unsafe actions |
| Operating envelope | A limit on a site's import or export set by the network operator |
| Feeder | The cable from the transformer that the homes share |
| Uniform-price double auction | A market where buyers and sellers bid and one clearing price is set |
| Core (game theory) | Allocations no group would beat by leaving |
| Owen allocation | Shares computed from the optimiser's shadow prices |
| Battery-neutral cost | Cost with used-up battery energy bought back at the cheapest tariff |
| Seed | The random starting point of a training run |
| Validation days | Separate days used to choose models, never for reporting |
| Grid-only | No coordination: every home deals with the grid alone |

## 4.4 Questions an examiner may ask

**Is this reinforcement learning or just imitation?** Both. The problem is a
multi-agent sequential decision process; the policies are MARL architectures
(MAPPO, MAT) trained with DAgger and then with on-policy PPO against the real
reward. The optimiser is a training-time teacher only.

**Why not pure RL from scratch?** v1 tried it: 3% of the possible saving at its
final checkpoint. Cost-only rewards are sparse and noisy for 12,000 coupled
decisions; a teacher gives a dense, correct signal.

**Why not just run the optimiser (MPC) in real time?** It needs forecasts and
seconds-to-minutes per solve, and with a 3-hour window it reached 52% of the
saving against the network's 80–83%.

**Is a central controller realistic?** Operators already run central systems
(virtual power plants, envelope publication). The hybrid design adds what a
real deployment needs: if the central link fails, homes continue safely on
their own.

**Why should a home trust it?** The settlement guarantees every home is better
off than alone whenever the community saves, and the platform shows each
home's bill.

**What would you do next?** Train and test over a full year of data; train on
communities of different sizes; model communication delays and wrong data;
replace the fixed envelope with operator-published dynamic envelopes.

**How do I check any number?** Section 4.1 lists the command that produces it;
`docs/SETUP.md` explains how to run them.
