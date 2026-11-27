# How to test and compare the approaches

Everything runs on CPU. Commands are run from the project folder
(`P2P_MARL_Capstone/`). Times are for a typical laptop.

**Windows (Command Prompt / PowerShell):** type each command on its own line
*without* anything after `#` — the Windows shell does not treat `#` as a
comment, so `python -m pytest tests -q  # expect ...` fails with
"file or directory not found: #". The `.sh` protocol script needs Git Bash or
WSL; on plain Windows run the individual `python scripts/...` commands from
section 6 instead.

## 0. Install (once)

```
pip install -r requirements.txt
pip install matplotlib
```

(`matplotlib` is only needed for the figures in step 4.)

Use Python 3.10+; the reported numbers were produced with Python 3.13,
torch 2.15, numpy 2.5, scipy 1.18.

## 1. Correctness tests (~2–5 min)

```
python -m pytest tests -q
```

Expected: **126 passed, 5 skipped** (2–10 min depending on the machine). The 5 skipped are browser tests that need Playwright; with it installed, 131 pass. Run `tests`, not the project root (the root also
contains scripts that pytest would try to import).

What they guarantee, among others:
* the exact optimum's schedule is settled by the simulator at (almost) its own cost;
* with the v2 shield, *no* action sequence (random, adversarial deferral) causes
  unserved energy or missed EV charging, and the v1 shield demonstrably fails
  the same adversary (negative control);
* MAT log-probabilities are exact, its decoder is causal, and its decoding order
  survives save/load;
* the core allocation is budget-balanced and no tested coalition gains by leaving;
* the settlement never charges a household more than its grid-only bill;
* with the fallback envelope, no outage pattern and no fallback behaviour (including an
  adversary that defers every car) causes unserved energy or missed EV charging, and the
  same adversary breaks the feeder without the envelope (negative control);
* the final MAT model is the RL-fine-tuned one only when it wins on validation days;
* the stakeholder platform's replay file is complete and budget-balanced, and the live
  server streams a full day and reacts to a cable cut.

## 2. Compare the approaches (~1–2 min) — **start here**

```bash
python scripts/compare_approaches.py
```

Prints one table: rule controller (v1/v2 shield), v1 MAPPO (as delivered: test-day
pick and final), oracle-guided MAPPO student, MAT student before RL (DAgger only),
**MAT after RL fine-tuning (the latest method)** and MPC. Two day sets:

* **A** — the deterministic reporting day used in the report;
* **B** — 8 noisy days from a seed never used for training, teacher data,
  DAgger or checkpoint selection (`--days 16` for more).

It ends with **`REGRESSION CHECK: PASS`** or **`FAIL`** plus the reason, and
writes `results/v2/compare_report.txt`.

Expected (lower is better, battery-neutral €/day):

| approach | A: mean | B: mean | unserved |
|---|---:|---:|---:|
| Rule controller, v1 shield | 153.35 | 153.10 | 28.77 |
| Rule controller, v2 shield | 143.90 | 143.67 | 0.00 |
| v1 MAPPO (picked on test day) | 151.27 | 151.39 | 4.00 |
| v1 MAPPO (final) | 173.43 | 173.28 | 111.29 |
| MPC, 3-h window | 135.93 | – | 0.00 |
| Oracle-guided MAPPO student | 116.06 | 116.57 | 0.00 |
| MAT student, DAgger only (no RL) | 113.84 | 113.67 | 0.00 |
| **MAT + RL fine-tuning (latest)** | **112.00** | **112.42** | **0.00** |

Exact optimum on day A: €98.74; grid-only: €175.89.

## 3. Full table with statistics (~10 min)

```bash
python scripts/final_eval_v2.py
```

Regenerates `results/v2/final_table.txt` — every controller and ablation, with
IQM, 95% bootstrap CIs, worst seed, share of achievable saving, individual
rationality before and after settlement, and probability of improvement over the
rule controller and MPC. It should match the shipped file exactly
(`git diff` or compare by eye with `docs/V2_RESULTS.md`).

## 3b. Communication outages: the hybrid controller (~3 min)

```
python scripts/eval_hybrid.py
```

Runs five outage scenarios (none, one household cut off all day, every household cut
off 17:00–21:00, 30% cut off all day, operator down all day) for every fallback
option and writes `results/v2/hybrid/hybrid_table.txt`. Expected: zero unserved
energy and zero EV shortfall in every row that has the fallback envelope; the
proposed controller (MAT, whole community switches to the MAPPO student on any
outage) equals MAT with no outage and stays near the MAPPO student otherwise.
See `docs/V2_HYBRID.md`.

The switching threshold was chosen on validation days with
`python scripts/select_degrade.py` (~4 min).

## 3c. Stakeholder platform (website)

Live: build your own community, pick a controller and an outage, watch it run,
cut the operator's link at any moment, and compare every controller:

```
python platform/server.py
```

then open http://localhost:8765 in a browser (stop the server with Ctrl+C).

1. **Set up → Run**: choose one of six controllers (the proposed MAT + MAPPO
   fallback first) and a scenario, or "Custom outage" with your own times.
2. **Set up → Community**: number of homes (4–60), which homes are drawn,
   feeder limits, and solar, battery, cars and demand for every home, or for all
   homes at once. Changed cells are highlighted; "Reset" restores the dataset.
3. **Run day**, then use **Cut link** on the timeline during the day.
4. **Compare** runs all six controllers on the same set-up and ranks them;
   **Watch** opens any of them in 3D. **Bills** shows each home's settlement.

Replay only (no Python needed): double-click `platform/index.html`. It plays the
recorded days in `platform/data/traces.js` (every controller in every preset
scenario); regenerate them with `python scripts/export_platform_data.py` (~1 min).

The 3D page loads three.js from the internet. `platform/classic.html` is a 2D
version with the same features that works offline (live: `http://localhost:8765/classic.html`). What everything on screen means:
`docs/EXPLAINER.md`, part B.

Browser tests (optional): `pip install playwright`, then
`python -m playwright install chromium`, then
`python -m pytest tests/test_v2_platform_ui.py -q`. Without Playwright these
tests are skipped.

## 4. Figures (~1 min)

```bash
python scripts/make_figures_v2.py      # -> results/v2/figures/fig1..fig4 (greyscale PNG)
```

## 5. Individual pieces (optional)

| What | Command | Time |
|---|---|---|
| v1 numbers still reproduce | `python scripts/reproduce_results.py --check` | 10 s |
| v1 policies re-scored battery-neutral | `python scripts/rescore_v1.py` | 1 min |
| MPC baseline | `python scripts/run_mpc.py --horizon 12` (`--proven-only` for the weaker variant) | 15–25 min |
| Strategic robustness, rule controller | `python scripts/strategic_eval.py` | 10 min |
| Strategic robustness, MAT student | `python scripts/strategic_eval.py --ckpt results/v2/dagger_mat/seed0/iter2.pt --out results/v2/strategic_student.json` | 15–25 min |
| RL fine-tuning of MAT (3 seeds) | `bash scripts/run_ft_mat.sh` (or `python scripts/finetune_rl.py --init <dagger iter.pt> --seed 0 --updates 60 --warmup 20 --lr-actor 1e-5 --log-std -3 --anchor 1.0 --anchor-final 1.0 --out results/v2/ft_mat/seed0`) | ~1 h per seed |

## 6. Retrain from scratch (optional, long)

```bash
bash scripts/run_v2_protocol.sh        # resumable; skips anything already finished
```

About 12 h on 2 cores (most of it exact MILP relabelling inside DAgger). To
retrain only one thing, delete its `summary.json` and rerun, or call the
script directly, e.g.

```bash
python scripts/train_dagger.py --arch mat --order laxity --seed 0 --iters 4 --days 8 --out results/v2/dagger_mat_retest
python scripts/train_dagger.py --arch mappo --seed 0 --iters 4 --days 8 --out results/v2/dagger_mappo_retest
```

Retrained numbers will not match to the cent (solver time limits and CPU speed
change which DAgger relabels are proven optimal), but should land within the
reported seed spread (MAT €111.85–116.20, MAPPO €115.09–117.91 on day A).

## 7. If something looks wrong — what to send me

1. `results/v2/compare_report.txt` (it records your Python/torch/numpy versions);
2. the full terminal output of the command that failed (including any traceback);
3. the output of `python -m pytest tests -q` if any test fails.

Typical non-problems:
* small differences (cents) on another machine or library version;
* `HighsMipSolverData::...` lines printed by the MILP solver — harmless log noise;
* MPC and strategic scripts are slow because they solve a MILP at every step.

A real problem is any of: a failing test; unserved energy or EV shortfall above
0 for the final MAT; the final MAT not being the cheapest on day A or B;
unserved energy above 0 in an enveloped row of `hybrid_table.txt`;
`final_eval_v2.py` producing a different table from the shipped one on the same
library versions.
