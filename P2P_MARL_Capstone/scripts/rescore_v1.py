"""Re-score the v1 policies under the v2 accounting.

    python scripts/rescore_v1.py

Prints raw cost, inventory-neutral cost, battery drawdown, and the share of
achievable saving recovered against the exact (MILP) bound with the end-of-day
battery condition.  Both the best (selected) and final checkpoints are shown.
"""
from __future__ import annotations
import json, sys
from dataclasses import asdict
from pathlib import Path
import numpy as np, torch
root = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(root))
torch.set_num_threads(1)
from src.data.community import build_tensors, load_community, stratified_subset
from src.env.baselines import BASELINES
from src.env.community_env import EnvConfig, P2PCommunityEnv
from src.marl.mappo import MAPPO
from src.marl.trainer import evaluate
from src.opt.lp import solve_dispatch
from src.utils.config import load_config, build_configs


def main():
    cfg = load_config(root / "configs/default.yaml")
    ct = build_tensors(stratified_subset(load_community(root / cfg["data"]["path"]),
                                         cfg["data"]["households"], seed=cfg["data"]["subset_seed"]))
    env_cfg, _, _ = build_configs(cfg)
    base = asdict(env_cfg)
    rows = {}
    for name, cls in BASELINES.items():
        pol = cls()
        env = P2PCommunityEnv(ct, EnvConfig(**{**base, "n_envs": 1, "noise_std": 0.0,
                                               "residual": pol.residual}), seed=12345)
        rows[name] = evaluate(env, pol, 1, is_agent=False)
    for s in range(5):
        for tag in ("best", "final"):
            p = root / f"results/main_study/seed{s}/{tag}.pt"
            if not p.exists():
                continue
            ag, ex = MAPPO.load(p)
            k = ex.get("residual_scale", 1.0)                 # final.pt trained at kappa=1
            lam = ex.get("multipliers") or ag.constraints.multipliers()
            env = P2PCommunityEnv(ct, EnvConfig(**{**base, "n_envs": 1, "noise_std": 0.0,
                                                   "residual": True, "residual_scale": k,
                                                   "residual_price_scale": k}), seed=12345)
            env.set_multipliers(lam)
            rows[f"mappo_s{s}_{tag}"] = evaluate(env, ag, 1)
    grid = float(rows["no_flexibility"]["net_cost_eur"][0])
    b_old = solve_dispatch(ct, env_cfg, exact=False, terminal_bess=False, degradation=False).cost
    b_exact = solve_dispatch(ct, env_cfg, exact=True, terminal_bess=True).cost
    print(f"grid-only {grid:.2f} | exact bound (battery-neutral) {b_exact:.2f}")
    print(f"{'policy':22s} {'raw':>8s} {'inv-neut':>9s} {'drawdown':>9s} {'rec_exact%':>10s} {'unserved':>9s}")
    out = {}
    for k, m in rows.items():
        raw, inv = float(m["net_cost_eur"][0]), float(m["net_cost_inv_neutral_eur"][0])
        rec = 100 * (grid - inv) / (grid - b_exact)
        out[k] = dict(raw=raw, inv_neutral=inv, drawdown=float(m["bess_drawdown_kwh"][0]),
                      recovered_pct=rec, unserved=float(m["unserved_kwh"][0]))
        print(f"{k:22s} {raw:8.2f} {inv:9.2f} {out[k]['drawdown']:9.1f} {rec:10.1f} {out[k]['unserved']:9.2f}")
    out["_bounds"] = {"grid_only": grid, "exact_battery_neutral": b_exact}
    Path(root / "results/v2").mkdir(parents=True, exist_ok=True)
    json.dump(out, open(root / "results/v2/rescore_v1.json", "w"), indent=1)


if __name__ == "__main__":
    main()
