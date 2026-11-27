"""Train one seed of a v2 learner and report best AND final checkpoints.

    python scripts/train_v2.py --config configs/v2.yaml --seed 0 --updates 150 --out results/v2/mappo
"""
from __future__ import annotations
import argparse, json, sys, time
from dataclasses import asdict
from pathlib import Path
import numpy as np, torch
root = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(root))
torch.set_num_threads(1)
from src.data.community import build_tensors, load_community, stratified_subset
from src.env.community_env import EnvConfig, P2PCommunityEnv
from src.marl.mappo import MAPPO, MAPPOConfig
from src.marl.trainer import Trainer, TrainConfig, evaluate
from src.utils.config import load_config, build_configs


def eval_ckpt(tensors, env_cfg, path):
    ag, ex = MAPPO.load(path)
    k = ex["residual_scale"]
    env = P2PCommunityEnv(tensors, EnvConfig(**{**asdict(env_cfg), "n_envs": 1, "noise_std": 0.0,
                                                "residual": True, "residual_scale": k,
                                                "residual_price_scale": k}), seed=12345)
    env.set_multipliers(ex["multipliers"])
    m = evaluate(env, ag, 1)
    return {k2: float(v[0]) for k2, v in m.items()} | {"update": ex.get("update")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v2.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--updates", type=int, default=None)
    ap.add_argument("--out", default="results/v2/mappo")
    ap.add_argument("--algo", default="{}", help="JSON algo overrides")
    ap.add_argument("--env", default="{}", help="JSON env overrides")
    a = ap.parse_args()
    cfg = load_config(root / a.config)
    if a.updates: cfg["train"]["updates"] = a.updates
    cfg["algo"].update(json.loads(a.algo)); cfg["env"].update(json.loads(a.env))
    tensors = build_tensors(stratified_subset(load_community(root / cfg["data"]["path"]),
                                              cfg["data"]["households"], seed=cfg["data"]["subset_seed"]))
    env_cfg, algo_cfg, train_cfg = build_configs(cfg)
    out = root / a.out / f"seed{a.seed}"
    tr = Trainer(tensors, EnvConfig(**asdict(env_cfg)),
                 MAPPOConfig(**{**asdict(algo_cfg), "seed": a.seed}),
                 TrainConfig(**{**asdict(train_cfg), "seed": a.seed}), out_dir=out)
    t0 = time.time(); tr.train(verbose=True)
    res = {"seed": a.seed, "wall_min": (time.time() - t0) / 60,
           "best": eval_ckpt(tensors, env_cfg, out / "best.pt"),
           "final": eval_ckpt(tensors, env_cfg, out / "final.pt")}
    json.dump(res, open(out / "summary.json", "w"), indent=1)
    for tag in ("best", "final"):
        r = res[tag]
        print(f"[{tag} @ {r['update']}] cost={r['net_cost_eur']:.2f} inv_neutral={r['net_cost_inv_neutral_eur']:.2f} "
              f"unserved={r['unserved_kwh']:.2f} short={r['ev_shortfall_kwh']:.2f} IR={r['individual_rationality']:.2f}")


if __name__ == "__main__":
    main()
