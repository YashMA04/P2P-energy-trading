"""Choose the 'switch the whole community to local control' threshold on VALIDATION days.

    python scripts/select_degrade.py        # -> results/v2/hybrid/degrade_selection.{txt,json}

When some households are unreachable, MAT still has to plan for the reachable ones
using a stale picture of the others. If that picture is stale enough it is better
to stop running MAT and let every household run its local MAPPO student (the
reachable ones keep the operator's community shield).  The threshold (share of
households offline) is picked here on noisy validation days (seed 777000, never
the reporting day) with random offline subsets (subset seeds 1-3, not the
subset seed 0 used in the reported 'partial' scenario).
"""
from __future__ import annotations
import json, sys, time
from dataclasses import replace
from pathlib import Path
import numpy as np, torch
root = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(root))
torch.set_num_threads(1)
from src.control.hybrid import HybridController, OutageSchedule
from src.data.community import build_tensors, load_community, stratified_subset
from src.env.community_env import P2PCommunityEnv
from src.utils.config import load_config, build_configs
from scripts.eval_hybrid import students

VAL_SEED, DAYS = 777000, 4
FRACS = (0.04, 0.12, 0.2, 0.3, 0.5)
SUBSETS = (1, 2, 3)


def cost(ct, cfg, ctrl):
    env = P2PCommunityEnv(ct, replace(cfg, n_envs=DAYS, noise_std=0.02, residual=False, action_mode="power"),
                          seed=VAL_SEED)
    obs, st = env.reset(seed=VAL_SEED, deterministic=False)
    done = False
    while not done:
        obs, st, _, done, info = env.step(ctrl.act(obs, st, env))
    ep = info["episode"]
    return float(np.mean(ep["net_cost_inv_neutral_eur"])), float(np.max(ep["unserved_kwh"]))


def main():
    t0 = time.time()
    cfg = load_config(root / "configs/v2.yaml")
    ct = build_tensors(stratified_subset(load_community(root / cfg["data"]["path"]),
                                         cfg["data"]["households"], seed=cfg["data"]["subset_seed"]))
    env_cfg, _, _ = build_configs(cfg)
    n = ct.n_agents
    res = {f: {"keep_mat": [], "all_local": []} for f in FRACS}
    for seed in range(3):
        mat, mappo = students(seed)
        for f in FRACS:
            for sub in SUBSETS:
                sched = OutageSchedule.random_households(n, f, seed=sub)
                for key, d in (("keep_mat", None), ("all_local", 0.0)):
                    c, u = cost(ct, env_cfg, HybridController(mat, mappo, sched, "local", degrade_frac=d))
                    assert u < 1e-3, (seed, f, sub, key, u)
                    res[f][key].append(c)
        print(f"seed {seed} done [{(time.time() - t0) / 60:.1f} min]", flush=True)
    lines = ["Degrade threshold selection (validation days, seed 777000; mean EUR/day over 3 model seeds x 3 subsets)",
             f"{'share offline':>14s} {'households':>10s} {'keep MAT':>9s} {'all local':>9s} {'better':>10s}"]
    thr = None
    for f in FRACS:
        k, l = np.mean(res[f]["keep_mat"]), np.mean(res[f]["all_local"])
        better = "all local" if l < k else "keep MAT"
        if l < k and thr is None:
            thr = f
        lines.append(f"{f:14.2f} {max(1, round(f * n)):10d} {k:9.2f} {l:9.2f} {better:>10s}")
    lines.append(f"selected degrade_frac = {thr}  (smallest share offline at which switching everyone to local wins)")
    txt = "\n".join(lines)
    print(txt)
    out = root / "results/v2/hybrid"; out.mkdir(parents=True, exist_ok=True)
    (out / "degrade_selection.txt").write_text(txt + "\n")
    json.dump({"selected": thr, "results": {str(f): v for f, v in res.items()}},
              open(out / "degrade_selection.json", "w"), indent=1)


if __name__ == "__main__":
    main()
