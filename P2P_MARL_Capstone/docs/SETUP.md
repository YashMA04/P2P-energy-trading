# Setup and Running Guide

Everything in this project runs on an ordinary laptop CPU (no GPU, no
accounts). Commands are the same on Windows, macOS and Linux unless a step
says otherwise. On Windows, type each command on its own line and do not add
`# comments` after a command: Windows `cmd` passes them to the program.

Contents: 1 install · 2 check it works · 3 the stakeholder platform ·
4 reproduce the results · 5 retrain from scratch · 6 tests · 7 troubleshooting ·
8 where things are.

---

## 1. Install (once, about 5–10 minutes)

**You need:** Python 3.10, 3.11, 3.12 or 3.13 and about 3 GB of free disk space
(most of it is PyTorch). Check with `python --version`.

Open a terminal in the `P2P_MARL_Capstone` folder, then:

```
python -m venv venv
```

Activate it:

| System | Command |
|---|---|
| Windows (cmd) | `venv\Scripts\activate` |
| Windows (PowerShell) | `venv\Scripts\Activate.ps1` |
| macOS / Linux | `source venv/bin/activate` |

You should now see `(venv)` at the start of the prompt. Install the packages:

```
pip install -r requirements.txt
```

This installs numpy, scipy, PyTorch, PyYAML, openpyxl and pytest.

**Optional (browser tests only):**

```
pip install playwright
python -m playwright install chromium
```

---

## 2. Check that it works (2–10 minutes)

```
python -m pytest tests -q
```

Expected: `126 passed, 5 skipped` (or `131 passed` if you installed Playwright).
Run it on the `tests` folder, not the project root.

Quick check of the main result against the reported numbers (1–2 minutes):

```
python scripts/compare_approaches.py
```

Expected last lines: a table of approaches and `REGRESSION CHECK: PASS`.

---

## 3. The stakeholder platform

### 3.1 Replay: no Python needed

Double-click either file:

* `platform/index.html`: the 3D control room (needs internet to load its
  graphics library);
* `platform/classic.html`: the same features in 2D, works offline, light and
  dark themes.

It plays recorded days of the 25-home study community: 6 controllers × 5
scenarios. Use **Set up** to choose a controller and scenario, **Compare** to
rank all controllers, **Bills** for each home's bill, and click any home for
its detail.

### 3.2 Live: build your own community and cut the link

```
python platform/server.py
```

Wait for `Feeder Control Room (LIVE) -> http://127.0.0.1:8765`, then open in a
browser:

* `http://localhost:8765` for 3D, or
* `http://localhost:8765/classic.html` for 2D.

The badge says **LIVE**. Then:

1. **Set up → Run**: choose a controller (the proposed one is first) and a
   scenario, or **Custom outage** (every home or a share of homes, between any
   two times).
2. **Set up → Community** (optional): number of homes 4–60, **Draw number**
   (which homes), feeder limits per home, and solar, battery, cars and demand
   for each home. Use **Set for every home** to change all homes at once; edited
   cells are highlighted; **Reset to dataset values** undoes everything.
3. **Run day**. Use **Cut link** on the timeline at any moment to cut the
   operator's communication; **Stop** ends the day early.
4. **Compare** runs all six controllers on your set-up (about 10–40 seconds
   depending on the number of homes) and ranks them; **Watch** opens any of them.
5. **Bills** after the day ends.

Stop the server with Ctrl+C. Options:

```
python platform/server.py --port 9000
python platform/server.py --model-seed 2
python platform/server.py --host 0.0.0.0
```

`--host 0.0.0.0` lets other computers on your network open it at
`http://<your computer's IP>:8765`. `--model-seed` picks which of the three
trained model sets (0, 1, 2) the platform uses; the default is 0.

### 3.3 Re-record the replay data

After retraining, refresh what the replay pages show (about 2 minutes):

```
python scripts/export_platform_data.py
```

This rewrites `platform/data/traces.js`.

---

## 4. Reproduce the reported results from the saved models

No training needed; the trained models are in `results/`.

| What | Command | Time | Output |
|---|---|---|---|
| Main comparison + PASS/FAIL check | `python scripts/compare_approaches.py` | 1–2 min | `results/v2/compare_report.txt` |
| Full table with confidence intervals | `python scripts/final_eval_v2.py` | 5–15 min | `results/v2/final_table.txt` |
| Communication-outage table | `python scripts/eval_hybrid.py` | about 3 min | `results/v2/hybrid/hybrid_table.txt` |
| Fallback switching threshold | `python scripts/select_degrade.py` | about 4 min | `results/v2/hybrid/degrade_selection.txt` |
| Report figures (greyscale) | `python scripts/make_figures_v2.py` | about 1 min | `results/v2/figures/` |
| v1 results still reproduce | `python scripts/reproduce_results.py --check` | about 1 min | ends with `ALL RESULTS REPRODUCED` |
| Strategic robustness, rule controller | `python scripts/strategic_eval.py` | 15–25 min | `results/v2/strategic_rule.json` |
| Diagnostics of the v1 benchmark and shield | `python scripts/phase0_diagnostics.py` | a few min | printed |

All evaluation is deterministic, so the numbers are the same on any machine up
to tiny library differences. `compare_approaches.py` reports a difference
larger than €0.05 as a FAIL so you notice.

---

## 5. Retrain everything from scratch

You only need this to show the training is reproducible; the trained models
are already included. Times are for one laptop CPU core and vary a lot by
machine; the full pipeline takes most of a day. Every step writes to
`results/v2/...`. Delete (or rename) a step's output folder before rerunning it,
because the scripts skip work that already exists.

**macOS / Linux / Git Bash on Windows** (resumable, runs steps 5.1–5.4):

```
bash scripts/run_v2_protocol.sh
bash scripts/run_ft_mat.sh
```

**Windows cmd or PowerShell**: run the individual commands below in order.

### 5.1 Teacher data: the exact optimiser on perturbed training days (hours)

```
python scripts/make_teacher.py --days 48 --seed 0 --out results/v2/teacher/teacher_s0.npz
python scripts/make_teacher.py --days 48 --seed 1 --out results/v2/teacher/teacher_s1.npz
```

### 5.2 Model-predictive control comparator (an hour or more)

```
python scripts/run_mpc.py --horizon 12 --out results/v2/mpc_h12.json
```

### 5.3 DAgger students, 3 seeds each (about 1.5–2 hours per run)

```
python scripts/train_dagger.py --arch mappo --seed 0 --iters 4 --days 8 --out results/v2/dagger_mappo
python scripts/train_dagger.py --arch mat --order laxity --seed 0 --iters 4 --days 8 --out results/v2/dagger_mat
```

Repeat both lines with `--seed 1` and `--seed 2`. Each run keeps every DAgger
round (`iter1.pt` … `iter4.pt`) and records in `summary.json` which round was
best on the validation days.

Optional decoding-order ablation:

```
python scripts/train_dagger.py --arch mat --order random --seed 0 --iters 2 --days 8 --out results/v2/dagger_mat_random
```

### 5.4 RL fine-tuning of MAT (anchored PPO)

For each seed, start from the DAgger round chosen on validation (the number in
`results/v2/dagger_mat/seed0/summary.json`, field `selected_iter_on_val`):

```
python scripts/finetune_rl.py --init results/v2/dagger_mat/seed0/iter2.pt --seed 0 --updates 60 --warmup 20 --lr-actor 1e-5 --log-std -3.0 --anchor 1.0 --anchor-final 1.0 --out results/v2/ft_mat
```

Repeat for seeds 1 and 2 with their own `iterN.pt`. The final model for each
seed is chosen automatically by `src/control/models.py`: the fine-tuned model
if it is cheaper on validation, otherwise the DAgger model.

### 5.5 Then evaluate

```
python scripts/final_eval_v2.py
python scripts/compare_approaches.py
python scripts/select_degrade.py
python scripts/eval_hybrid.py
python scripts/export_platform_data.py
```

If your retrained models give a slightly different mean, update `EXPECTED_A` at
the top of `scripts/compare_approaches.py` to your new mean.

### 5.6 The original v1 study (for reference)

```
python scripts/run_study.py --seeds 0 1 2 --updates 150 --households 25
python scripts/train_v2.py --config configs/v2.yaml --seed 0 --updates 150 --out results/v2/mappo
```

The first reproduces the v1 MAPPO training; the second trains MAPPO from
scratch with all v2 fixes (safety layer, constraint handling, validation-based
selection), which is the "pure RL" comparison point.

---

## 6. Tests

| File | Covers | Time |
|---|---|---|
| `tests/test_all.py`, `tests/test_system.py` | v1 simulator, market, physics, baselines | about 1 min |
| `tests/test_v2_optimisation.py` | exact optimiser, replay in the simulator | about 1 min |
| `tests/test_v2_shield.py` | per-vehicle and community shields | under 1 min |
| `tests/test_v2_teacher.py`, `test_v2_mpc_distill.py` | teacher data, MPC, DAgger, distillation | 1–2 min |
| `tests/test_v2_mat.py` | MAT network, decoding order, saving and loading | under 1 min |
| `tests/test_v2_game.py` | fair settlement (core), strategic tests | about 1 min |
| `tests/test_v2_hybrid.py` | outage safety, adversarial fallback, negative control | about 2 min |
| `tests/test_v2_custom.py` | custom communities, validation of every field | under 1 min |
| `tests/test_v2_sim.py` | the six controllers, scenarios, comparison, link cut | about 2 min |
| `tests/test_v2_platform.py` | model selection, replay data, live server endpoints | about 1 min |
| `tests/test_v2_platform_ui.py` | real browser on the 3D and 2D pages (needs Playwright) | 1–3 min |

Run one file: `python -m pytest tests/test_v2_sim.py -q`. Run one test by name:
`python -m pytest tests -q -k cable_cut`.

The browser tests load three.js from the internet. To run them offline, unpack
the `three@0.128.0` npm package and set `P2P_THREE_DIR` to its folder.

---

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| `pip install` fails on torch | Upgrade pip (`python -m pip install --upgrade pip`) and retry; on old Python versions install Python 3.11 or newer |
| `ERROR: file or directory not found: #` | You added a `# comment` after a command on Windows; remove it |
| `no tests ran` | Run `python -m pytest tests -q` from the project folder (the one with `README.md`) |
| `Address already in use` when starting the server | Another program uses port 8765: `python platform/server.py --port 9000` |
| The 3D page is blank or says it could not start 3D graphics | No internet (the library comes from a CDN) or WebGL disabled: use `classic.html` |
| The page says REPLAY although the server is running | Open it through the server address (`http://localhost:8765`), not by double-clicking the file |
| Compare takes long | It simulates six full days; 60 homes take about a minute |
| A training script seems to do nothing | Its output already exists, so it skips; delete that folder to retrain |
| Different numbers after retraining | Expected for a different machine or library version within cents; update `EXPECTED_A` as described in 5.5 |

---

## 8. Where things are

```
P2P_MARL_Capstone/
├── README.md                 overview and headline results
├── docs/
│   ├── FULL_GUIDE.md / .docx everything from scratch: why each choice, ML and platform
│   ├── EXPLAINER.md / .docx  two-part summary (ML method, platform)
│   ├── SETUP.md              this file
│   ├── HOW_TO_TEST.md        how to test and compare approaches, expected outputs
│   ├── V2_METHOD.md, V2_RESULTS.md, V2_HYBRID.md, V2_DISCARDED.md   details
├── configs/v2.yaml           all settings (environment, training, safety)
├── data/EC_EV_dataset.xlsx   the 250-home dataset
├── src/
│   ├── env/                  simulator, market, safety shields, rule controllers
│   ├── opt/                  exact optimiser, MPC, teacher, DAgger
│   ├── marl/                 MAPPO, MAT, training, distillation
│   ├── game/                 fair settlement, strategic tests
│   ├── control/              hybrid controller with fallback, final model selection
│   └── platform/             custom communities, simulation variants, frame recorder
├── platform/
│   ├── index.html            3D control room
│   ├── classic.html          2D control room
│   ├── server.py             live server
│   └── data/traces.js        recorded days for replay
├── scripts/                  everything you run (training, evaluation, export)
├── results/                  trained models and every result file
└── tests/                    the test suite
```
