"""Settle receding-horizon MPC on the deterministic reporting day.

    python scripts/run_mpc.py --horizon 12 --out results/v2/mpc_h12.json
"""
from __future__ import annotations
import argparse, json, sys, time
from dataclasses import asdict
from pathlib import Path
import numpy as np
root = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(root))
from src.data.community import build_tensors, load_community, stratified_subset
from src.env.community_env import EnvConfig, P2PCommunityEnv
from src.opt.mpc import MPCPolicy
from src.utils.config import load_config, build_configs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v2.yaml")
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--time-limit", type=float, default=20.0)
    ap.add_argument("--proven-only", action="store_true",
                    help="fall back to the convex plan whenever optimality is not proven")
    ap.add_argument("--out", default="results/v2/mpc_h12.json")
    a = ap.parse_args()
    cfg = load_config(root / a.config)
    ct = build_tensors(stratified_subset(load_community(root / cfg["data"]["path"]),
                                         cfg["data"]["households"], seed=cfg["data"]["subset_seed"]))
    env_cfg, _, _ = build_configs(cfg)
    pol = MPCPolicy(ct, env_cfg, horizon=a.horizon, time_limit=a.time_limit,
                    accept_incumbent=not a.proven_only)
    env = P2PCommunityEnv(ct, EnvConfig(**{**asdict(env_cfg), "n_envs": 1, "noise_std": 0.0,
                                           "residual": False}), seed=0)
    obs, st = env.reset(seed=0, deterministic=True); done = False; t0 = time.time()
    while not done:
        obs, st, _, done, info = env.step(pol.act(obs, st, env))
    ep = {k: float(np.atleast_1d(v)[0]) for k, v in info["episode"].items()}
    ep.update({"horizon": a.horizon, "proven_only": a.proven_only, "fallbacks_to_convex": pol.fallbacks,
               "solve_failures": pol.solve_failures, "seconds": time.time() - t0})
    Path(root / a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(ep, open(root / a.out, "w"), indent=1)
    print(f"MPC h={a.horizon}: cost={ep['net_cost_eur']:.2f} invN={ep['net_cost_inv_neutral_eur']:.2f} "
          f"unserved={ep['unserved_kwh']:.2f} short={ep['ev_shortfall_kwh']:.2f} "
          f"fallbacks={pol.fallbacks} [{ep['seconds']:.0f}s]")


if __name__ == "__main__":
    main()
