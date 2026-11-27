"""Oracle-guided MARL: distil the privileged teacher, then RL fine-tune.

    python scripts/train_student.py --teacher results/v2/teacher/teacher_s0.npz \
        --seed 0 --ft-updates 100 --out results/v2/student_mappo

Reports, for the pure student (after behaviour cloning) and for the best and
final fine-tuned checkpoints: cost and inventory-neutral cost on the
deterministic reporting day (never seen in training), and on held-out noisy
validation days.
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
from src.marl.distill import BCRegulariser, fit_student
from src.marl.mappo import MAPPO, MAPPOConfig
from src.marl.mat import MAT, MATConfig, load_any
from src.marl.trainer import Trainer, TrainConfig, evaluate
from src.utils.config import load_config, build_configs

KEYS = ("net_cost_eur", "net_cost_inv_neutral_eur", "unserved_kwh", "ev_shortfall_kwh",
        "individual_rationality", "fairness_jain", "p2p_volume_kwh", "bess_drawdown_kwh")


def load_teacher(paths):
    ds = [dict(np.load(p)) for p in paths]
    return {k: np.concatenate([d[k] for d in ds]) for k in ("obs", "state", "act", "day_info")}


def score(tensors, env_cfg, agent, multipliers=None, val_seed=777000, episodes=8):
    out = {}
    for tag, noisy, E in (("test", False, 1), ("val", True, episodes)):
        env = P2PCommunityEnv(tensors, EnvConfig(**{**asdict(env_cfg), "n_envs": E,
                                                    "noise_std": 0.02 if noisy else 0.0}),
                              seed=val_seed)
        if multipliers:
            env.set_multipliers(multipliers)
        m = evaluate(env, agent, E, seed=val_seed if noisy else 12345, noisy=noisy)
        out[tag] = {k: float(np.mean(m[k])) for k in KEYS} | {
            "unserved_max": float(np.max(m["unserved_kwh"])),
            "shortfall_max": float(np.max(m["ev_shortfall_kwh"]))}
    return out


def fmt(tag, r):
    t, v = r["test"], r["val"]
    return (f"[{tag}] TEST cost={t['net_cost_eur']:.2f} invN={t['net_cost_inv_neutral_eur']:.2f} "
            f"unserved={t['unserved_kwh']:.2f} short={t['ev_shortfall_kwh']:.2f} IR={t['individual_rationality']:.2f} "
            f"| VAL invN={v['net_cost_inv_neutral_eur']:.2f} unserved_max={v['unserved_max']:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v2.yaml")
    ap.add_argument("--teacher", nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bc-epochs", type=int, default=40)
    ap.add_argument("--ft-updates", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--bc-coef", type=float, default=1.0)
    ap.add_argument("--bc-coef-final", type=float, default=0.05)
    ap.add_argument("--lr-actor", type=float, default=1e-4)
    ap.add_argument("--log-std", type=float, default=-1.6)
    ap.add_argument("--out", default="results/v2/student_mappo")
    ap.add_argument("--arch", default="mappo", choices=["mappo", "mat"])
    ap.add_argument("--action-mode", default="soc_target", choices=["power", "soc_target"])
    ap.add_argument("--order", default="laxity", choices=["laxity", "random", "index"])
    a = ap.parse_args()

    cfg = load_config(root / a.config)
    cfg["train"]["updates"] = max(a.ft_updates, 1)
    cfg["env"]["residual"] = False                 # student acts in absolute actions
    cfg["env"]["action_mode"] = a.action_mode
    cfg["algo"]["learning_rate_actor"] = a.lr_actor
    cfg["algo"]["log_std_init"] = a.log_std
    tensors = build_tensors(stratified_subset(load_community(root / cfg["data"]["path"]),
                                              cfg["data"]["households"], seed=cfg["data"]["subset_seed"]))
    env_cfg, algo_cfg, train_cfg = build_configs(cfg)
    out = root / a.out / f"seed{a.seed}"; out.mkdir(parents=True, exist_ok=True)
    data = load_teacher([root / p for p in a.teacher])
    if a.action_mode == "soc_target":
        from src.opt.teacher import soc_target_labels
        data["act"] = soc_target_labels(tensors, data, tensors.dt)
    print(f"teacher: {data['obs'].shape[0]} days, {data['obs'].shape[0] * 96 * tensors.n_agents} samples")

    tr = Trainer(tensors, EnvConfig(**asdict(env_cfg)),
                 MAPPOConfig(**{**asdict(algo_cfg), "seed": a.seed}),
                 TrainConfig(**{**asdict(train_cfg), "seed": a.seed,
                                "critic_warmup_updates": a.warmup}), out_dir=out)
    if a.arch == "mat":
        tr.agent = MAT(tr.env.observation_dim, tr.env.state_dim, tr.env.action_dim,
                       MATConfig(**{**asdict(tr.agent.cfg), "order": a.order, "order_seed": a.seed}),
                       n_agents=tensors.n_agents)
    t0 = time.time()
    hist = fit_student(tr.agent, data, epochs=a.bc_epochs, log_std_after=a.log_std, seed=a.seed)
    res = {"seed": a.seed, "arch": a.arch, "action_mode": a.action_mode, "order": a.order, "bc_hist": hist, "bc_min": (time.time() - t0) / 60}
    res["bc"] = score(tensors, env_cfg, tr.agent)
    tr.agent.save(out / "bc.pt", extra={"update": 0, "residual_scale": 1.0, "env_residual": False,
                                        "multipliers": tr.agent.constraints.multipliers()})
    print(fmt("BC student", res["bc"]), flush=True)

    if a.ft_updates > 0:
        tr.agent.bc = BCRegulariser(data, a.bc_coef, a.bc_coef_final, seed=a.seed)
        tr.train(verbose=True)
        for tag in ("best", "final"):
            ag, ex = load_any(out / f"{tag}.pt")
            res[tag] = score(tensors, env_cfg, ag, ex.get("multipliers"))
            res[tag]["update"] = ex.get("update")
            print(fmt(f"{tag} @ {ex.get('update')}", res[tag]))
    res["wall_min"] = (time.time() - t0) / 60
    json.dump(res, open(out / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
