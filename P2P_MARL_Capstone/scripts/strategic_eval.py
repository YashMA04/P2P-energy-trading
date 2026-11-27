"""Strategic robustness of a policy + the core settlement.

    python scripts/strategic_eval.py                       # rule controller
    python scripts/strategic_eval.py --ckpt results/v2/student_mat/seed0/best.pt
"""
from __future__ import annotations
import argparse, json, sys
from dataclasses import asdict, replace
from pathlib import Path
import numpy as np, torch
root = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(root))
torch.set_num_threads(1)
from src.data.community import build_tensors, load_community, stratified_subset
from src.env.baselines import TariffAwareP2P
from src.env.community_env import EnvConfig
from src.game.core import owen_allocation
from src.game.strategic import market_exploitability, misreport_gain, rollout_bills
from src.marl.mat import load_any
from src.utils.config import load_config, build_configs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v2.yaml")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default="results/v2/strategic_rule.json")
    a = ap.parse_args()
    cfg = load_config(root / a.config)
    ct = build_tensors(stratified_subset(load_community(root / cfg["data"]["path"]),
                                         cfg["data"]["households"], seed=cfg["data"]["subset_seed"]))
    env_cfg, _, _ = build_configs(cfg)
    if a.ckpt:
        pol, ex = load_any(root / a.ckpt)
        k = ex.get("residual_scale", 1.0)
        env_cfg = replace(env_cfg, residual=ex.get("env_residual", env_cfg.residual),
                          residual_scale=k, residual_price_scale=k)
        is_agent, lam = True, ex.get("multipliers")
    else:
        pol, is_agent, lam = TariffAwareP2P(), False, None
        env_cfg = replace(env_cfg, residual=True)
    x = owen_allocation(ct, env_cfg)["x"]
    me = market_exploitability(ct, env_cfg, pol, is_agent, x, multipliers=lam)
    base = rollout_bills(ct, env_cfg, pol, is_agent, multipliers=lam)
    lies = []
    for h in range(ct.n_agents):
        for ls, es in ((0.8, 1.0), (1.2, 1.0), (1.0, 0.8), (1.0, 1.2)):
            lies.append({"household": h, "load_scale": ls, "ev_scale": es,
                         "gain_eur": misreport_gain(ct, env_cfg, base["total"], base["grid_only"], h, ls, es)})
    best_lie = max(lies, key=lambda r: r["gain_eur"])
    res = {"policy": a.ckpt or "tariff_aware_p2p (v2 shield)", "base_total_eur": base["total"],
           "market": me, "misreport": {"rows": lies, "max_gain_eur": best_lie["gain_eur"], "argmax": best_lie}}
    Path(root / a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(root / a.out, "w"), indent=1, default=float)
    print(f"base community cost {base['total']:.2f}")
    print(f"market exploitability: raw bills {me['exploitability_raw_eur']:.3f} EUR/day | "
          f"under core settlement {me['exploitability_settled_eur']:.3f} EUR/day")
    top = sorted(me["rows"], key=lambda r: -r["best_gain_raw_eur"])[:3]
    for r in top:
        print(f"  household {r['household']}: raw gain {r['best_gain_raw_eur']:.3f} via (quote, qty, d_total)={r['best_raw_deviation']}")
    print(f"data misreporting to the settlement: max gain {best_lie['gain_eur']:.3f} EUR/day at {best_lie}")


if __name__ == "__main__":
    main()
