"""Quick side-by-side of the main approaches, with an automatic regression check.

    python scripts/compare_approaches.py              # ~5-10 min on a laptop CPU
    python scripts/compare_approaches.py --days 16    # more unseen noisy days

Evaluates, from saved checkpoints (no training):
  * rule controller, v1 shield and v2 community shield
  * v1 MAPPO as delivered (checkpoint picked on the test day, and final)
  * oracle-guided MAPPO student (DAgger), 3 seeds
  * oracle-guided MAT student (DAgger) -- the latest / proposed method, 3 seeds
  * MPC (read from results/v2/mpc_h12.json; rerun with scripts/run_mpc.py)
on two day sets:
  A. the deterministic reporting day (as in the report);
  B. N noisy days drawn with a seed (424242) that was never used for training,
     teacher data, DAgger rollouts or checkpoint selection.
Costs are battery-neutral EUR/day. Safety (unserved energy, EV shortfall) is
checked on every day.

Ends with a REGRESSION CHECK: PASS if the MAT student is cheaper than every
other approach on both day sets and safe on every day; otherwise FAIL with the
reason. Everything is also written to results/v2/compare_report.txt -- send me
that file if anything fails.
"""
from __future__ import annotations
import argparse, json, platform, sys, time
from dataclasses import replace
from pathlib import Path
import numpy as np, torch
root = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(root))
torch.set_num_threads(1)
from src.data.community import build_tensors, load_community, stratified_subset
from src.env.baselines import TariffAwareP2P
from src.env.community_env import P2PCommunityEnv
from src.marl.mat import load_any
from src.marl.trainer import evaluate
from src.utils.config import load_config, build_configs

UNSEEN_SEED = 424242
EXPECTED_A = 112.00          # final MAT, mean over 3 seeds, reporting day (docs/V2_RESULTS.md)


def run(ct, cfg, pol, is_agent, days, lam=None):
    """Return per-day (inv-neutral cost, unserved, shortfall) for test day + noisy days."""
    out = {}
    for tag, noisy, E, seed in (("report_day", False, 1, 12345), ("unseen_noisy", True, days, UNSEEN_SEED)):
        env = P2PCommunityEnv(ct, replace(cfg, n_envs=E, noise_std=0.02 if noisy else 0.0), seed=seed)
        if lam:
            env.set_multipliers(lam)
        m = evaluate(env, pol, E, seed=seed, is_agent=is_agent, noisy=noisy)
        out[tag] = {"cost": m["net_cost_inv_neutral_eur"].tolist(),
                    "unserved_max": float(np.max(m["unserved_kwh"])),
                    "shortfall_max": float(np.max(m["ev_shortfall_kwh"]))}
    return out


def ckpt(ct, env_cfg, v1_cfg, path, days, v1=False):
    ag, ex = load_any(path)
    k = ex.get("residual_scale", 1.0)
    base = v1_cfg if v1 else env_cfg
    cfg = replace(base, residual=ex.get("env_residual", True), action_mode="power",
                  residual_scale=k, residual_price_scale=k)
    lam = ex.get("multipliers") or (ag.constraints.multipliers() if v1 else None)
    return run(ct, cfg, ag, True, days, lam)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=8)
    a = ap.parse_args()
    t0 = time.time()
    cfg = load_config(root / "configs/v2.yaml")
    ct = build_tensors(stratified_subset(load_community(root / cfg["data"]["path"]),
                                         cfg["data"]["households"], seed=cfg["data"]["subset_seed"]))
    env_cfg, _, _ = build_configs(cfg)
    v1_cfg = replace(env_cfg, community_shield=False, firm_minimum_only=False, ir_episodic=False)
    R = root / "results/v2"
    methods = {}
    print("evaluating rule controllers ...", flush=True)
    methods["Rule controller, v1 shield"] = [run(ct, replace(v1_cfg, residual=True), TariffAwareP2P(), False, a.days)]
    methods["Rule controller, v2 shield"] = [run(ct, replace(env_cfg, residual=True), TariffAwareP2P(), False, a.days)]
    print("evaluating v1 MAPPO checkpoints ...", flush=True)
    for tag, label in (("best", "v1 MAPPO (picked on test day)"), ("final", "v1 MAPPO (final)")):
        methods[label] = [ckpt(ct, env_cfg, v1_cfg, p, a.days, v1=True)
                          for p in sorted((root / "results/main_study").glob(f"seed*/{tag}.pt"))]
    from src.control.models import dagger_ckpt, final_mat_ckpt
    for arch, label in (("mappo", "Oracle-guided MAPPO student"),
                        ("mat", "MAT student, DAgger only (no RL)")):
        print(f"evaluating {label} ...", flush=True)
        methods[label] = [ckpt(ct, env_cfg, v1_cfg, dagger_ckpt(arch, s), a.days) for s in range(3)]
    print("evaluating final MAT (DAgger + RL fine-tuning) ...", flush=True)
    methods["MAT + RL fine-tuning (LATEST)"] = [ckpt(ct, env_cfg, v1_cfg, final_mat_ckpt(s)[0], a.days)
                                                for s in range(3)]
    mpc = R / "mpc_h12.json"
    mpc_cost = json.load(open(mpc))["net_cost_inv_neutral_eur"] if mpc.exists() else None

    lines = [f"Approach comparison  (battery-neutral EUR/day; lower is better)",
             f"python {platform.python_version()}, torch {torch.__version__}, numpy {np.__version__}",
             f"A = deterministic reporting day | B = {a.days} unseen noisy days (seed {UNSEEN_SEED})", "",
             f"{'approach':38s} {'seeds':>5s} {'A: mean':>9s} {'A: worst':>9s} {'B: mean':>9s} {'B: worst':>9s} {'unserved':>9s} {'shortfall':>9s}"]
    summ = {}
    for name, rr in methods.items():
        A = np.array([np.mean(r["report_day"]["cost"]) for r in rr])
        B = np.array([np.mean(r["unseen_noisy"]["cost"]) for r in rr])
        uns = max(max(r["report_day"]["unserved_max"], r["unseen_noisy"]["unserved_max"]) for r in rr)
        sh = max(max(r["report_day"]["shortfall_max"], r["unseen_noisy"]["shortfall_max"]) for r in rr)
        summ[name] = {"A_mean": float(A.mean()), "A_worst": float(A.max()), "B_mean": float(B.mean()),
                      "B_worst": float(B.max()), "unserved_max": uns, "shortfall_max": sh, "n": len(rr)}
        lines.append(f"{name:38s} {len(rr):5d} {A.mean():9.2f} {A.max():9.2f} {B.mean():9.2f} {B.max():9.2f} {uns:9.2f} {sh:9.2f}")
    if mpc_cost is not None:
        lines.append(f"{'MPC, 3-h window (from mpc_h12.json)':38s} {1:5d} {mpc_cost:9.2f} {mpc_cost:9.2f} {'-':>9s} {'-':>9s}")
        summ["MPC"] = {"A_mean": mpc_cost}
    lines += ["", "exact optimum (perfect foresight) on day A: EUR 98.74; grid-only: EUR 175.89", ""]

    # ---- regression check -------------------------------------------------
    latest = summ["MAT + RL fine-tuning (LATEST)"]
    fails = []
    for name, s in summ.items():
        if name.endswith("(LATEST)"):
            continue
        if s["A_mean"] <= latest["A_mean"]:
            fails.append(f"day A: '{name}' ({s['A_mean']:.2f}) is not worse than the latest method ({latest['A_mean']:.2f})")
        if "B_mean" in s and s["B_mean"] <= latest["B_mean"]:
            fails.append(f"days B: '{name}' ({s['B_mean']:.2f}) is not worse than the latest method ({latest['B_mean']:.2f})")
    if latest["unserved_max"] > 1e-3:
        fails.append(f"latest method has unserved energy ({latest['unserved_max']:.3f} kWh)")
    if latest["shortfall_max"] > 1e-3:
        fails.append(f"latest method has EV shortfall ({latest['shortfall_max']:.3f} kWh)")
    if abs(latest["A_mean"] - EXPECTED_A) > 0.05:
        fails.append(f"day A mean {latest['A_mean']:.2f} differs from the reported {EXPECTED_A:.2f} "
                     "(different library versions can shift this slightly; send the report)")
    lines.append("REGRESSION CHECK: " + ("PASS - the latest method is the cheapest approach on both day sets and safe on every day"
                                         if not fails else "FAIL"))
    lines += [f"  - {f}" for f in fails]
    lines.append(f"[{(time.time() - t0) / 60:.1f} min]")
    txt = "\n".join(lines)
    print("\n" + txt)
    (R / "compare_report.txt").write_text(txt + "\n")
    json.dump(summ, open(R / "compare_report.json", "w"), indent=1)
    print(f"\nwritten: {R / 'compare_report.txt'}")


if __name__ == "__main__":
    main()
