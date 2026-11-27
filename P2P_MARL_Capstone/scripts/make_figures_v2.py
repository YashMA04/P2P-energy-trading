"""Greyscale figures for the v2 report, regenerated from saved results.

    python scripts/make_figures_v2.py      # -> results/v2/figures/*.png

fig1_ladder.png      share of achievable saving recovered, every controller
fig2_dagger.png      closed-loop cost per DAgger iteration, MAPPO vs MAT students
fig3_evening.png     community feeder import 18:00-24:00: v1 shield vs v2 shield
fig4_strategic.png   best single-household deviation gain: raw bills vs settlement
"""
from __future__ import annotations
import json, sys
from dataclasses import asdict, replace
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
root = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(root))
R = root / "results/v2"; F = R / "figures"; F.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "DejaVu Sans"})
GREYS = ["0.15", "0.45", "0.7", "0.85"]


def fig1():
    t = json.load(open(R / "final_table.json"))
    rows = [r for r in t["table"] if not r["policy"].endswith(", final") and "(v1 shield)" not in r["policy"]
            or r["policy"].startswith("tariff_aware_p2p (v1")]
    rows = sorted(rows, key=lambda r: r["recovered_pct"])
    fig, ax = plt.subplots(figsize=(8.0, 0.30 * len(rows) + 1.2))
    y = np.arange(len(rows))
    learned = ["student" in r["policy"] or "MAPPO" in r["policy"] or "PPO" in r["policy"] for r in rows]
    ax.barh(y, [r["recovered_pct"] for r in rows], color=[GREYS[0] if l else GREYS[2] for l in learned],
            edgecolor="0.1", linewidth=0.4)
    for i, r in enumerate(rows):
        if r["n_seeds"] > 1:
            lo = 100 * (t["grid_only"] - r["ci_high"]) / (t["grid_only"] - t["bound_exact_battery_neutral"])
            hi = 100 * (t["grid_only"] - r["ci_low"]) / (t["grid_only"] - t["bound_exact_battery_neutral"])
            ax.plot([lo, hi], [i, i], color="0.0", lw=1.0)
    ax.set_yticks(y); ax.set_yticklabels([r["policy"] for r in rows], fontsize=7)
    ax.axvline(0, color="0.3", lw=0.6); ax.axvline(100, color="0.3", lw=0.6, ls="--")
    ax.set_xlabel("share of achievable saving recovered (%)\ngrid-only = 0, exact battery-neutral optimum = 100 (dashed)")
    lo_x = -60
    ax.set_xlim(lo_x, 105)
    for i, r in enumerate(rows):
        if r["recovered_pct"] < lo_x:
            ax.text(lo_x + 2, i, f"{r['recovered_pct']:.0f}% (off scale)", va="center", fontsize=6.5, color="white")
    ax.set_title("Controllers on the reporting day\n(dark = learned; whiskers = 95% bootstrap CI over 3 seeds)", fontsize=8)
    fig.tight_layout(); fig.savefig(F / "fig1_ladder.png", dpi=200); plt.close(fig)


def fig2():
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for d, label, ls in (("dagger_mappo", "MAPPO student", "-"), ("dagger_mat", "MAT student (laxity order)", "--"),
                         ("dagger_mat_random", "MAT student (random order)", ":")):
        curves = []
        for s in sorted((R / d).glob("seed*/dagger_log.json")):
            L = json.load(open(s)); curves.append([r["test"]["net_cost_inv_neutral_eur"] for r in L])
        if not curves:
            continue
        n = min(len(c) for c in curves); C = np.array([c[:n] for c in curves])
        ax.plot(range(n), C.mean(0), ls, color="0.1", marker="o", ms=3, label=f"{label} (n={len(C)})")
        if len(C) > 1:
            ax.fill_between(range(n), C.min(0), C.max(0), color="0.8", alpha=0.6, lw=0)
    mpc = json.load(open(R / "mpc_h12.json"))["net_cost_inv_neutral_eur"]
    t = json.load(open(R / "final_table.json"))
    ax.axhline(mpc, color="0.4", lw=0.8, ls="-."); ax.text(0.05, mpc + 1, "MPC, 3-h window", fontsize=7, color="0.3")
    ax.axhline(t["bound_exact_battery_neutral"], color="0.4", lw=0.8)
    ax.text(0.05, t["bound_exact_battery_neutral"] + 1, "exact optimum (oracle)", fontsize=7, color="0.3")
    ax.set_xlabel("DAgger iteration (0 = behaviour cloning only)"); ax.set_ylabel("EUR/day, battery-neutral")
    ax.set_ylim(t["bound_exact_battery_neutral"] - 5, 230); ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(F / "fig2_dagger.png", dpi=200); plt.close(fig)


def fig3():
    from src.data.community import build_tensors, load_community, stratified_subset
    from src.env.baselines import DeadlineCharging
    from src.env.community_env import EnvConfig, P2PCommunityEnv
    from src.utils.config import load_config, build_configs
    cfg = load_config(root / "configs/v2.yaml")
    ct = build_tensors(stratified_subset(load_community(root / cfg["data"]["path"]), 25, seed=0))
    env_cfg, _, _ = build_configs(cfg)
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for v2, ls, lab in ((False, "--", "v1 per-vehicle shield"), (True, "-", "v2 community shield")):
        env = P2PCommunityEnv(ct, replace(env_cfg, n_envs=1, noise_std=0.0, residual=False,
                                          community_shield=v2, firm_minimum_only=v2))
        obs, st = env.reset(seed=0, deterministic=True); pol = DeadlineCharging(); done = False
        prof, uns = [], []
        while not done:
            u0 = env.metrics["unserved_kwh"].copy()
            obs, st, _, done, info = env.step(pol.act(obs, st, env))
            uns.append(float((env.metrics["unserved_kwh"] - u0)[0]) / env.dt)
            prof.append(float(info["congestion"][0]) * env.feeder_import_kw)
        h = np.arange(96) / 4
        ax.plot(h[72:], prof[72:], ls, color="0.1", label=f"{lab}: unserved {sum(uns) * env.dt:.0f} kWh")
    ax.axhline(env.feeder_import_kw, color="0.5", lw=0.8); ax.text(18.1, env.feeder_import_kw + 8, "feeder limit", fontsize=7)
    ax.set_xlabel("hour of day"); ax.set_ylabel("net community demand on feeder (kW)")
    ax.set_title("Adversarial deferral (every car waits for its deadline)", fontsize=8)
    ax.legend(fontsize=7, frameon=False); fig.tight_layout(); fig.savefig(F / "fig3_evening.png", dpi=200); plt.close(fig)


def fig4():
    files = [("strategic_rule.json", "rule controller"), ("strategic_student.json", "oracle-guided student")]
    have = [(json.load(open(R / f)), l) for f, l in files if (R / f).exists()]
    if not have:
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    w = 0.38
    for k, (d, lab) in enumerate(have):
        raw = sorted([r["best_gain_raw_eur"] for r in d["market"]["rows"]], reverse=True)[:10]
        st = sorted([r["best_gain_settled_eur"] for r in d["market"]["rows"]], reverse=True)[:10]
        x = np.arange(len(raw)) + k * w
        ax.bar(x, raw, w, color=GREYS[k], edgecolor="0.1", lw=0.4, label=f"{lab}: auction bill")
        ax.bar(x, st, w * 0.5, color="0.0", lw=0,
               label="core settlement, either policy (black inset)" if k == 0 else None)
    ax.set_xlabel("household (10 most exploitable, ranked)"); ax.set_ylabel("best deviation gain (EUR/day)")
    ax.set_xticks([]); ax.legend(fontsize=7, frameon=False)
    ax.set_title("Bid shading / withholding by one household, others cooperative\n"
                 "(under settlement the rule controller's gains are ~EUR 0.001: bars invisible)", fontsize=8)
    fig.tight_layout(); fig.savefig(F / "fig4_strategic.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4):
        try:
            f(); print("ok", f.__name__)
        except FileNotFoundError as e:
            print("skip", f.__name__, e)
