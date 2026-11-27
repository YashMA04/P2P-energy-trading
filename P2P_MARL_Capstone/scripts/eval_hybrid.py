"""Outage scenarios: does the hybrid (MAT + local fallback) stay cheap and safe?

    python scripts/eval_hybrid.py                 # -> results/v2/hybrid/hybrid_table.{txt,json}

Scenarios (deterministic reporting day):
  none      everyone reachable all day
  evening   the operator loses ALL households 17:00-21:00 (EV peak)
  partial   30% of households unreachable all day
  full      operator down all day
Controllers:
  MAT, no fallback          unreachable households keep their last setpoint
  MAT + rule fallback       unreachable households run the rule controller
  MAT + MAPPO fallback      unreachable households run the MAPPO student locally
  ... whole community       as soon as anyone is unreachable, everyone runs the MAPPO
                            student (reachable ones keep the community shield) (proposed)
  MAPPO student everywhere  decentralised policy, same shield semantics
  Rule controller           non-learned reference
Each learned controller uses the seed-s MAT and seed-s MAPPO student, s = 0, 1, 2.
"""
from __future__ import annotations
import json, sys, time
from dataclasses import replace
from pathlib import Path
import numpy as np, torch
root = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(root))
torch.set_num_threads(1)
from src.control.hybrid import (HybridController, OutageSchedule, RuleAgent, proposed_controller,
                                MODE_CENTRAL, MODE_LOCAL, MODE_RULE, MODE_STALE)
from src.data.community import build_tensors, load_community, stratified_subset
from src.env.community_env import P2PCommunityEnv
from src.marl.mat import load_any
from src.utils.config import load_config, build_configs


def students(seed):
    """Final MAT (DAgger + RL fine-tuning, selected on validation) and the MAPPO student."""
    from src.control.models import final_mat_ckpt, final_mappo_ckpt
    return load_any(final_mat_ckpt(seed)[0])[0], load_any(final_mappo_ckpt(seed))[0]


def rollout(ct, cfg, ctrl, seed=12345, noisy=False, trace=False, E=1):
    env = P2PCommunityEnv(ct, replace(cfg, n_envs=E, noise_std=0.02 if noisy else 0.0,
                                      residual=False, action_mode="power"), seed=seed)
    obs, st = env.reset(seed=seed, deterministic=not noisy)
    done, steps = False, []
    while not done:
        a = ctrl.act(obs, st, env)
        obs, st, _, done, info = env.step(a)
        if trace:
            steps.append(env)        # caller extracts what it needs
    ep = {k: (float(np.atleast_1d(v)[0]) if E == 1 else np.asarray(v, float))
          for k, v in info["episode"].items()}
    modes = np.array(ctrl.modes) if hasattr(ctrl, "modes") else None
    return ep, modes, env


def scenarios(n):
    return {"none": OutageSchedule.none(n),
            "evening": OutageSchedule.window(n, 68, 84, name="all households offline 17:00-21:00"),
            "one": OutageSchedule.random_households(n, 0.04, seed=0, name="one household offline all day"),
            "partial": OutageSchedule.random_households(n, 0.3, seed=0, name="30% of households offline all day"),
            "full": OutageSchedule.full(n)}


def controllers(mat, mappo, sched):
    return {"MAT, no fallback (hold last setpoint)": HybridController(mat, None, sched, "stale"),
            "MAT + rule fallback": HybridController(mat, None, sched, "rule"),
            "MAT + MAPPO fallback, offline households only": HybridController(mat, mappo, sched, "local"),
            "MAT + MAPPO fallback, whole community switches (proposed)": proposed_controller(mat, mappo, sched),
            "  same, WITHOUT fallback envelope (ablation)": proposed_controller(mat, mappo, sched),
            "MAPPO student everywhere": HybridController(mappo, mappo, sched, "local"),
            "Rule controller everywhere": HybridController(RuleAgent(), None, sched, "rule")}


def main():
    t0 = time.time()
    cfg = load_config(root / "configs/v2.yaml")
    ct = build_tensors(stratified_subset(load_community(root / cfg["data"]["path"]),
                                         cfg["data"]["households"], seed=cfg["data"]["subset_seed"]))
    env_cfg, _, _ = build_configs(cfg)
    out = {}
    for seed in range(3):
        mat, mappo = students(seed)
        for sname, sched in scenarios(ct.n_agents).items():
            for cname, ctrl in controllers(mat, mappo, sched).items():
                if cname.startswith("Rule") and seed > 0:
                    continue                       # deterministic, seed-independent
                cfg_run = replace(env_cfg, offline_import_cap_kw=None) if "WITHOUT" in cname else env_cfg
                ep, modes, _ = rollout(ct, cfg_run, ctrl)
                share = {k: float((modes == m).mean()) for k, m in
                         (("central", MODE_CENTRAL), ("local", MODE_LOCAL), ("rule", MODE_RULE), ("stale", MODE_STALE))}
                out.setdefault(sname, {}).setdefault(cname, []).append(
                    {"seed": seed, "cost": ep["net_cost_inv_neutral_eur"], "unserved": ep["unserved_kwh"],
                     "shortfall": ep["ev_shortfall_kwh"], "ir": ep["individual_rationality"], "modes": share})
                print(f"seed {seed} | {sname:8s} | {cname:58s} | {ep['net_cost_inv_neutral_eur']:7.2f} EUR "
                      f"| unserved {ep['unserved_kwh']:6.2f} | shortfall {ep['ev_shortfall_kwh']:5.2f}", flush=True)
    # unseen noisy days (seed never used anywhere else) for the proposed controller
    noisy = {}
    for seed in range(3):
        mat, mappo = students(seed)
        for sname, sched in scenarios(ct.n_agents).items():
            ep, _, _ = rollout(ct, env_cfg, proposed_controller(mat, mappo, sched),
                               seed=424242, noisy=True, E=8)
            noisy.setdefault(sname, []).append(
                {"seed": seed, "cost_mean": float(ep["net_cost_inv_neutral_eur"].mean()),
                 "unserved_max": float(ep["unserved_kwh"].max()),
                 "shortfall_max": float(ep["ev_shortfall_kwh"].max())})
            print(f"noisy days seed {seed} | {sname:8s} | proposed | {noisy[sname][-1]}", flush=True)
    out["_noisy_days_proposed"] = noisy
    R = root / "results/v2/hybrid"; R.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(R / "hybrid_table.json", "w"), indent=1)
    lines = ["Outage scenarios (deterministic reporting day, battery-neutral EUR/day; mean over seeds [worst])", ""]
    for sname, sched in scenarios(ct.n_agents).items():
        lines.append(f"Scenario: {sched.name}")
        for cname, rr in out[sname].items():
            if cname.startswith("_"):
                continue
            c = np.array([r["cost"] for r in rr])
            lines.append(f"  {cname:58s} {c.mean():7.2f} [{c.max():7.2f}]  unserved max {max(r['unserved'] for r in rr):6.2f}"
                         f"  shortfall max {max(r['shortfall'] for r in rr):5.2f}  (n={len(rr)})")
        nd = out["_noisy_days_proposed"][sname]
        lines.append(f"  {'proposed, 8 unseen noisy days':58s} {np.mean([r['cost_mean'] for r in nd]):7.2f} "
                     f"[{max(r['cost_mean'] for r in nd):7.2f}]  unserved max {max(r['unserved_max'] for r in nd):6.2f}"
                     f"  shortfall max {max(r['shortfall_max'] for r in nd):5.2f}  (n=3 seeds x 8 days)")
        lines.append("")
    lines.append(f"[{(time.time() - t0) / 60:.1f} min]")
    (R / "hybrid_table.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
