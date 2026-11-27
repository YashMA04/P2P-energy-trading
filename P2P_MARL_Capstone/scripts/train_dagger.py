"""Oracle-guided student via DAgger.

    python scripts/train_dagger.py --arch mappo --iters 4 --days 8 --out results/v2/dagger_mappo

Iteration 0 = behaviour cloning on teacher days.  Each iteration rolls the
current student out on fresh noisy days, relabels every ``--every`` steps with
the exact MILP re-solved from the visited state, aggregates, and refits.
Closed-loop cost (deterministic reporting day + held-out noisy days) is logged
after every iteration.
"""
from __future__ import annotations
import argparse, json, sys, time
from dataclasses import asdict
from pathlib import Path
import numpy as np, torch
root = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(root))
torch.set_num_threads(1)
from src.data.community import build_tensors, load_community, stratified_subset
from src.env.community_env import EnvConfig
from src.marl.distill import fit_student
from src.marl.mappo import MAPPO, MAPPOConfig
from src.marl.mat import MAT, MATConfig
from src.opt.dagger import dagger_collect
from src.utils.config import load_config, build_configs
sys.path.insert(0, str(root / "scripts"))
from train_student import score, fmt, load_teacher


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v2.yaml")
    ap.add_argument("--teacher", nargs="+", default=["results/v2/teacher/teacher_s0.npz",
                                                     "results/v2/teacher/teacher_s1.npz"])
    ap.add_argument("--arch", default="mappo", choices=["mappo", "mat"])
    ap.add_argument("--order", default="laxity")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--days", type=int, default=8)
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--time-limit", type=float, default=30.0)
    ap.add_argument("--dagger-weight", type=float, default=1.0)
    ap.add_argument("--epochs0", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="results/v2/dagger_mappo")
    a = ap.parse_args()
    cfg = load_config(root / a.config)
    cfg["env"]["residual"] = False
    cfg["env"]["action_mode"] = "power"
    ct = build_tensors(stratified_subset(load_community(root / cfg["data"]["path"]),
                                         cfg["data"]["households"], seed=cfg["data"]["subset_seed"]))
    env_cfg, algo_cfg, _ = build_configs(cfg)
    out = root / a.out / f"seed{a.seed}"; out.mkdir(parents=True, exist_ok=True)
    teacher = load_teacher([root / p for p in a.teacher]); teacher["weight"] = 1.0
    acfg = {**asdict(algo_cfg), "seed": a.seed}
    if a.arch == "mat":
        agent = MAT(45, 20, 5, MATConfig(**acfg, order=a.order, order_seed=a.seed), n_agents=ct.n_agents)
    else:
        agent = MAPPO(45, 20, 5, MAPPOConfig(**acfg))
    extra = {"residual_scale": 1.0, "env_residual": False, "multipliers": {"ir": 0.0, "grid": 0.0}}
    log, agg = [], []
    t0 = time.time()
    fit_student(agent, [teacher], epochs=a.epochs0, seed=a.seed, verbose=False)
    for it in range(a.iters + 1):
        ck = out / f"iter{it}.pt"; agent.save(ck, extra={**extra, "update": it})
        r = score(ct, env_cfg, agent)
        r.update({"iter": it, "minutes": (time.time() - t0) / 60, "dagger_states": int(sum(d["obs"].shape[1] for d in agg))})
        log.append(r); json.dump(log, open(out / "dagger_log.json", "w"), indent=1)
        print(fmt(f"iter {it} ({r['minutes']:.0f} min)", r), flush=True)
        if it == a.iters:
            break
        d = dagger_collect(ct, env_cfg, str(ck), a.days, seed=a.seed * 1000 + it, every=a.every,
                           time_limit=a.time_limit, workers=a.workers)
        print(f"   collected {a.days} student days, rollout cost mean {d['rollout_cost'].mean():.2f}, "
              f"relabel solve failures {d['solve_failures']}", flush=True)
        agg.extend(d["days"])
        for x in agg:
            x["weight"] = a.dagger_weight / len(agg)     # DAgger data totals dagger_weight
        fit_student(agent, [teacher] + agg, epochs=a.epochs, seed=a.seed + it, verbose=False)
    if agg:   # keep the relabelled states for later fine-tuning runs
        np.savez_compressed(out / "dagger_data.npz",
                            **{f"{k}_{i}": d[k][0] for i, d in enumerate(agg) for k in ("obs", "state", "act")})
    best = min(log, key=lambda r: r["val"]["net_cost_inv_neutral_eur"])     # select on VAL
    json.dump({"log": log, "selected_iter_on_val": best["iter"]}, open(out / "summary.json", "w"), indent=1)
    print(f"selected (on validation days): iter {best['iter']} -> {fmt('selected', best)}")


if __name__ == "__main__":
    main()
