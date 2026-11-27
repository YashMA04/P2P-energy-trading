"""RL fine-tuning of an oracle-guided student (PPO on the true reward).

    python scripts/finetune_rl.py --init results/v2/dagger_mappo/seed0/iter4.pt --seed 0 \
        --updates 60 --out results/v2/ft_mappo

PPO with per-household advantages and the v2 constraints, starting from the
DAgger student, a critic-only warm-up, and an annealed anchor regulariser
toward the starting policy.  Checkpoints are selected on noisy validation days.
"""
from __future__ import annotations
import argparse, json, sys, time
from dataclasses import asdict
from pathlib import Path
import numpy as np, torch
root = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(root)); sys.path.insert(0, str(root / "scripts"))
torch.set_num_threads(1)
from src.data.community import build_tensors, load_community, stratified_subset
from src.env.community_env import EnvConfig
from src.marl.distill import AnchorRegulariser
from src.marl.mat import load_any
from src.marl.mappo import MAPPOConfig
from src.marl.trainer import Trainer, TrainConfig
from src.utils.config import load_config, build_configs
from train_student import score, fmt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v2.yaml")
    ap.add_argument("--init", required=True)
    ap.add_argument("--teacher", nargs="+", default=["results/v2/teacher/teacher_s0.npz"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--updates", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--lr-actor", type=float, default=5e-5)
    ap.add_argument("--log-std", type=float, default=-2.0)
    ap.add_argument("--anchor", type=float, default=1.0)
    ap.add_argument("--anchor-final", type=float, default=0.1)
    ap.add_argument("--out", default="results/v2/ft_mappo")
    a = ap.parse_args()
    cfg = load_config(root / a.config)
    cfg["env"]["residual"] = False; cfg["env"]["action_mode"] = "power"
    cfg["train"]["updates"] = a.updates; cfg["train"]["eval_every"] = 10
    ct = build_tensors(stratified_subset(load_community(root / cfg["data"]["path"]),
                                         cfg["data"]["households"], seed=cfg["data"]["subset_seed"]))
    env_cfg, algo_cfg, train_cfg = build_configs(cfg)
    out = root / a.out / f"seed{a.seed}"; out.mkdir(parents=True, exist_ok=True)
    tr = Trainer(ct, EnvConfig(**asdict(env_cfg)), MAPPOConfig(**{**asdict(algo_cfg), "seed": a.seed}),
                 TrainConfig(**{**asdict(train_cfg), "seed": a.seed, "critic_warmup_updates": a.warmup}),
                 out_dir=out)
    agent, _ = load_any(root / a.init)
    anchor, _ = load_any(root / a.init)
    agent.cfg.learning_rate_actor = a.lr_actor
    agent.cfg.seed = a.seed
    for g in agent.opt_actor.param_groups: g["lr"] = a.lr_actor
    with torch.no_grad():
        agent.actor.log_std.fill_(a.log_std)
    ds = [dict(np.load(root / p)) for p in a.teacher]
    bank_o = np.concatenate([d["obs"].reshape(-1, *d["obs"].shape[2:]) for d in ds])
    bank_s = np.concatenate([d["state"].reshape(-1, d["state"].shape[-1]) for d in ds])
    sel = np.random.default_rng(a.seed).choice(len(bank_o), min(4000, len(bank_o)), replace=False)
    agent.bc = AnchorRegulariser(anchor, bank_o[sel], bank_s[sel], a.anchor, a.anchor_final, seed=a.seed)
    tr.agent = agent
    res = {"init": a.init, "seed": a.seed, "start": score(ct, env_cfg, agent)}
    print(fmt("start", res["start"]), flush=True)
    t0 = time.time(); tr.train(verbose=True)
    for tag in ("best", "final"):
        ag, ex = load_any(out / f"{tag}.pt")
        res[tag] = score(ct, env_cfg, ag, ex.get("multipliers")); res[tag]["update"] = ex.get("update")
        print(fmt(f"{tag} @ {ex.get('update')}", res[tag]))
    res["wall_min"] = (time.time() - t0) / 60
    json.dump(res, open(out / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
