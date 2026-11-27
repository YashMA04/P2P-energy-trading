"""Regenerate the v2 comparison table from saved checkpoints and logs.

    python scripts/final_eval_v2.py            # -> results/v2/final_table.{json,txt}

Every row is settled in the simulator on the deterministic reporting day
(never used for training or checkpoint selection).  Learned methods use the
checkpoint selected on held-out noisy validation days, and the FINAL
checkpoint is reported alongside.  Costs are battery-neutral (inventory
drawdown valued at the cheapest tariff of the day); "recovered" is the share
of the achievable saving, grid-only -> exact battery-neutral optimum.
Multi-seed methods report mean, IQM and a 95% bootstrap CI over seeds, and
the probability of improvement over the fixed rule controller and over MPC.
"""
from __future__ import annotations
import json, sys
from dataclasses import asdict, replace
from pathlib import Path
import numpy as np, torch
root = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(root)); sys.path.insert(0, str(root / "scripts"))
torch.set_num_threads(1)
from src.data.community import build_tensors, load_community, stratified_subset
from src.env.baselines import BASELINES
from src.env.community_env import EnvConfig, P2PCommunityEnv
from src.eval.statistics import bootstrap_ci, iqm, probability_of_improvement
from src.game.core import owen_allocation, settle
from src.marl.mat import load_any
from src.marl.trainer import evaluate
from src.opt.lp import solve_dispatch
from src.utils.config import load_config, build_configs

R = root / "results/v2"


def run_policy(ct, env_cfg, pol, is_agent, lam=None):
    env = P2PCommunityEnv(ct, replace(env_cfg, n_envs=1, noise_std=0.0), seed=12345)
    if lam:
        env.set_multipliers(lam)
    m = evaluate(env, pol, 1, is_agent=is_agent)
    out = {k: float(v[0]) for k, v in m.items()}
    out["_bills"] = env.cum_cost[0].tolist()
    out["_grid_only"] = env.baseline_cost_total.tolist()
    return out


def ckpt_row(ct, env_cfg, path):
    ag, ex = load_any(path)
    k = ex.get("residual_scale", 1.0)
    cfg = replace(env_cfg, residual=ex.get("env_residual", True), action_mode="power",
                  residual_scale=k, residual_price_scale=k)
    return run_policy(ct, cfg, ag, True, ex.get("multipliers"))


def main():
    cfg = load_config(root / "configs/v2.yaml")
    ct = build_tensors(stratified_subset(load_community(root / cfg["data"]["path"]),
                                         cfg["data"]["households"], seed=cfg["data"]["subset_seed"]))
    env_cfg, _, _ = build_configs(cfg)
    v1_cfg = replace(env_cfg, community_shield=False, firm_minimum_only=False, ir_episodic=False)
    bound = solve_dispatch(ct, env_cfg, exact=True, terminal_bess=True).cost
    x_core = owen_allocation(ct, env_cfg)["x"]
    rows = {}

    # ---- non-learned ladder -------------------------------------------------
    for name, cls in BASELINES.items():
        p = cls()
        rows[f"{name} (v1 shield)"] = [run_policy(ct, replace(v1_cfg, residual=p.residual), p, False)]
        rows[f"{name} (v2 shield)"] = [run_policy(ct, replace(env_cfg, residual=p.residual), p, False)]
    mpc = R / "mpc_h12.json"
    if mpc.exists():
        rows["MPC, 3-h perfect-forecast window"] = [json.load(open(mpc))]

    # ---- v1 MAPPO (as delivered), selected and final -----------------------
    for tag in ("best", "final"):
        rr = []
        for s in range(5):
            p = root / f"results/main_study/seed{s}/{tag}.pt"
            if p.exists():
                ag, ex = load_any(p)
                k = ex.get("residual_scale", 1.0)
                lam = ex.get("multipliers") or ag.constraints.multipliers()
                rr.append(run_policy(ct, replace(v1_cfg, residual=True, residual_scale=k, residual_price_scale=k),
                                     ag, True, lam))
        if rr:
            rows[f"v1 MAPPO, {'selected on test day' if tag == 'best' else 'final'}"] = rr

    # ---- v2 learners --------------------------------------------------------
    def seeds(d):
        return sorted((R / d).glob("seed*"))

    for d, label in (("smoke_mappo", "MAPPO-v2 (stable PPO from rule controller)"),):
        for tag in ("best", "final"):
            rr = [ckpt_row(ct, env_cfg, s / f"{tag}.pt") for s in seeds(d) if (s / f"{tag}.pt").exists()]
            if rr:
                rows[f"{label}, {'selected' if tag == 'best' else 'final'}"] = rr
    for d, label in (("bc96_power_mappo", "Imitation only (BC on 96 teacher days)"),
                     ("bc_soc_mappo", "Imitation only, SoC-target actions")):
        rr = [ckpt_row(ct, env_cfg, s / "bc.pt") for s in seeds(d) if (s / "bc.pt").exists()]
        if d == "bc_soc_mappo" and rr:
            rr = []  # evaluated in its own action mode below
            for s in seeds(d):
                ag, ex = load_any(s / "bc.pt")
                rr.append(run_policy(ct, replace(env_cfg, residual=False, action_mode="soc_target"), ag, True))
        if rr:
            rows[label] = rr
    for d, label in (("dagger_mappo", "Oracle-guided MAPPO student (DAgger)"),
                     ("dagger_mat", "Oracle-guided MAT student, laxity order (DAgger)"),
                     ):
        sel, fin = [], []
        for s in seeds(d):
            sm = s / "summary.json"
            if not sm.exists():
                continue
            j = json.load(open(sm))
            it_sel, it_fin = j["selected_iter_on_val"], max(r["iter"] for r in j["log"])
            sel.append(ckpt_row(ct, env_cfg, s / f"iter{it_sel}.pt"))
            fin.append(ckpt_row(ct, env_cfg, s / f"iter{it_fin}.pt"))
        if sel:
            rows[f"{label}, selected"] = sel
            rows[f"{label}, final"] = fin
    from src.control.models import final_mat_ckpt
    if (R / "ft_mat").exists():
        rows["FINAL: MAT, DAgger + RL fine-tuning, selected on val"] = [
            ckpt_row(ct, env_cfg, final_mat_ckpt(s)[0]) for s in range(3)]
        rows["FINAL: MAT, DAgger + RL fine-tuning, final update"] = [
            ckpt_row(ct, env_cfg, R / f"ft_mat/seed{s}/final.pt") for s in range(3)]
    for d, label in (("ft_mappo_gentle", "MAPPO student + PPO fine-tuning (anchored)"), ("ft_mappo", "MAPPO student + PPO fine-tuning (standard)")):
        for tag in ("best", "final"):
            rr = [ckpt_row(ct, env_cfg, s / f"{tag}.pt") for s in seeds(d) if (s / f"{tag}.pt").exists()]
            if rr:
                rows[f"{label}, {'selected' if tag == 'best' else 'final'}"] = rr

    # ---- imitation-stage comparison (DAgger iteration 0 = behaviour cloning) --
    imit = {}
    for arch, dirs in (("MAPPO student", ("dagger_mappo", "bc_mappo")),
                       ("MAT student, laxity order", ("dagger_mat", "bc_mat_laxity")),
                       ("MAT student, random order", ("dagger_mat_random",))):
        per_seed = {}
        for d in dirs:
            for s in seeds(d):
                if (s / "iter0.pt").exists() and s.name not in per_seed:
                    per_seed[s.name] = ckpt_row(ct, env_cfg, s / "iter0.pt")["net_cost_inv_neutral_eur"]
        if per_seed:
            imit[arch] = dict(sorted(per_seed.items()))

    # ---- aggregate ----------------------------------------------------------
    grid = rows["no_flexibility (v2 shield)"][0]["net_cost_eur"]
    ref = rows["tariff_aware_p2p (v2 shield)"][0]["net_cost_inv_neutral_eur"]
    mpc_c = rows.get("MPC, 3-h perfect-forecast window", [{"net_cost_inv_neutral_eur": np.nan}])[0]["net_cost_inv_neutral_eur"]
    table = []
    for name, rr in rows.items():
        c = np.array([r["net_cost_inv_neutral_eur"] for r in rr])
        rec = 100 * (grid - c) / (grid - bound)
        ir_settled = []
        for r in rr:
            if "_bills" in r:
                s = settle(np.array(r["_grid_only"]), r["net_cost_eur"], x_core)
                ir_settled.append(float(s["ir"].mean()) if s["savings"] >= 0 else np.nan)
        lo, hi = bootstrap_ci(c) if len(c) > 1 else (c[0], c[0])
        table.append({
            "policy": name, "n_seeds": len(rr),
            "cost_raw": float(np.mean([r["net_cost_eur"] for r in rr])),
            "cost_inv_neutral": float(c.mean()), "iqm": float(iqm(c)) if len(c) > 2 else float(c.mean()),
            "ci_low": float(lo), "ci_high": float(hi), "worst_seed": float(c.max()),
            "recovered_pct": float(rec.mean()),
            "unserved_kwh_max": float(max(r["unserved_kwh"] for r in rr)),
            "ev_shortfall_kwh_max": float(max(r["ev_shortfall_kwh"] for r in rr)),
            "ir_raw_min": float(min(r["individual_rationality"] for r in rr)),
            "ir_settled_min": float(np.nanmin(ir_settled)) if ir_settled and not np.all(np.isnan(ir_settled)) else np.nan,
            "p2p_kwh": float(np.mean([r.get("p2p_volume_kwh", np.nan) for r in rr])),
            "p_beats_rule": probability_of_improvement(c, np.array([ref])),
            "p_beats_mpc": probability_of_improvement(c, np.array([mpc_c])) if np.isfinite(mpc_c) else np.nan,
        })
    # head-to-head of the two oracle-guided students (selected checkpoints)
    a = rows.get("FINAL: MAT, DAgger + RL fine-tuning, selected on val")
    b = rows.get("Oracle-guided MAPPO student (DAgger), selected")
    head2head = None
    if a and b:
        ca = np.array([r["net_cost_inv_neutral_eur"] for r in a]); cb = np.array([r["net_cost_inv_neutral_eur"] for r in b])
        head2head = {"p_mat_beats_mappo_student": probability_of_improvement(ca, cb),
                     "mean_difference_eur": float(cb.mean() - ca.mean())}
    imit_rows = []
    for arch, ps in imit.items():
        v = np.array(list(ps.values()))
        imit_rows.append({"student": arch, "per_seed": ps, "mean": float(v.mean()),
                          "recovered_pct": float(100 * (grid - v.mean()) / (grid - bound))})
    if "MAPPO student" in imit:
        base = np.array(list(imit["MAPPO student"].values()))
        for r in imit_rows:
            r["p_beats_mappo_student"] = probability_of_improvement(np.array(list(r["per_seed"].values())), base)
    if "MAT student, random order" in imit:
        rnd = np.array(list(imit["MAT student, random order"].values()))
        for r in imit_rows:
            r["p_beats_random_order"] = probability_of_improvement(np.array(list(r["per_seed"].values())), rnd)
    out = {"bound_exact_battery_neutral": bound, "grid_only": grid, "table": table,
           "imitation_stage": imit_rows, "mat_vs_mappo_student": head2head}
    json.dump(out, open(R / "final_table.json", "w"), indent=1, default=float)
    hdr = (f"{'policy':58s} {'n':>2s} {'EUR/day':>8s} {'IQM':>7s} {'95% CI':>15s} {'worst':>7s} "
           f"{'recov%':>7s} {'unserv':>7s} {'short':>6s} {'IRraw':>6s} {'IRset':>6s} {'P>rule':>6s} {'P>MPC':>6s}")
    lines = [f"Exact battery-neutral optimum: EUR {bound:.2f}/day | grid-only EUR {grid:.2f}/day", hdr, "-" * len(hdr)]
    for r in table:
        lines.append(f"{r['policy'][:58]:58s} {r['n_seeds']:2d} {r['cost_inv_neutral']:8.2f} {r['iqm']:7.2f} "
                     f"[{r['ci_low']:6.2f},{r['ci_high']:6.2f}] {r['worst_seed']:7.2f} {r['recovered_pct']:7.1f} "
                     f"{r['unserved_kwh_max']:7.2f} {r['ev_shortfall_kwh_max']:6.2f} {r['ir_raw_min']:6.2f} "
                     f"{r['ir_settled_min']:6.2f} {r['p_beats_rule']:6.2f} {r['p_beats_mpc']:6.2f}")
    lines.append(f"{'exact optimum (perfect foresight, oracle)':58s}  - {bound:8.2f}")
    if head2head:
        lines += ["", f"MAT vs MAPPO oracle-guided students (selected): MAT cheaper by EUR {head2head['mean_difference_eur']:.2f}/day on average; "
                      f"P(MAT seed beats MAPPO seed) = {head2head['p_mat_beats_mappo_student']:.2f}"]
    lines += ["", "Imitation stage (behaviour cloning on teacher days, before DAgger), battery-neutral EUR/day:"]
    for r in imit_rows:
        lines.append(f"  {r['student']:28s} " + " ".join(f"{k}={v:7.2f}" for k, v in r["per_seed"].items())
                     + f" | mean {r['mean']:7.2f} | P(beats MAPPO student)={r.get('p_beats_mappo_student', float('nan')):.2f}"
                     + f" | P(beats random order)={r.get('p_beats_random_order', float('nan')):.2f}")
    txt = "\n".join(lines)
    open(R / "final_table.txt", "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
