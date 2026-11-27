"""Phase-0 diagnostics. Run from the project root."""
import sys, json, re, inspect, textwrap
from dataclasses import asdict
from pathlib import Path
import numpy as np, torch
root = Path.cwd(); sys.path.insert(0, str(root))
torch.set_num_threads(1)
from src.data.community import build_tensors, load_community, stratified_subset
from src.env.community_env import EnvConfig, P2PCommunityEnv
from src.env import baselines as B
from src.marl.mappo import MAPPO
from src.utils.config import load_config, build_configs

cfg = load_config(root / "configs/default.yaml")
ct = build_tensors(stratified_subset(load_community(root / cfg["data"]["path"]),
                                     cfg["data"]["households"], seed=cfg["data"]["subset_seed"]))
env_cfg, algo_cfg, train_cfg = build_configs(cfg)

def run(policy, agent=None, extra=None):
    over = dict(n_envs=1, noise_std=0.0, residual=(agent is not None) or policy.residual)
    if extra:
        k = extra.get("residual_scale")
        if k is not None: over.update(residual_scale=k, residual_price_scale=k)
    env = P2PCommunityEnv(ct, EnvConfig(**{**asdict(env_cfg), **over}), seed=12345)
    if extra and extra.get("multipliers"): env.set_multipliers(extra["multipliers"])
    obs, st = env.reset(seed=12345, deterministic=True)
    log = []
    # capture the shield mask
    orig = env.ev_projection.__call__
    cap = {}
    def wrapped(a, soc, ctx):
        out = orig(a, soc, ctx); cap["must"] = out[1]; return out
    env.ev_projection.__call__ = wrapped
    proj = env.ev_projection
    class P:  # wrap so __call__ override works
        def __call__(s, a, soc, ctx):
            out = EV(a, soc, ctx); cap["must"] = out[1]; return out
    EV = proj; env.ev_projection = P()
    done = False
    while not done:
        t = env.t
        a = agent.act(obs, st, True)[0] if agent else policy.act(obs, st, env)
        un0 = env.metrics["unserved_kwh"].copy()
        obs, st, _, done, info = env.step(a)
        log.append(dict(t=t, unserved=float((env.metrics["unserved_kwh"] - un0)[0]),
                        n_must=int(cap["must"].sum()), cong=float(info["congestion"][0])))
    ep = {k: float(np.atleast_1d(v)[0]) for k, v in info["episode"].items()}
    bess_end = float(env.bess_soc.sum()); bess0 = float(ct.bess_soc0.sum())
    return ep, log, bess_end - bess0

print("initial BESS energy (kWh):", float(ct.bess_soc0.sum()),
      " usable above min SoC:", float(np.maximum(ct.bess_soc0 - env_cfg.bess_min_soc * ct.bess_capacity, 0).sum()))

out = {}
ep, log, dE = run(B.TariffAwareP2P())
out["heuristic"] = dict(cost=ep["net_cost_eur"], dBESS=dE, unserved=ep["unserved_kwh"])
for s in range(5):
    ag, extra = MAPPO.load(root / f"results/main_study/seed{s}/best.pt")
    ep, log, dE = run(None, ag, extra)
    bad = [r for r in log if r["unserved"] > 1e-6]
    out[f"mappo{s}"] = dict(cost=ep["net_cost_eur"], dBESS=dE, unserved=ep["unserved_kwh"],
                            unserved_steps=[(r["t"], round(r["unserved"], 3), r["n_must"]) for r in bad],
                            lam=extra.get("multipliers"), kappa=extra.get("residual_scale"),
                            update=extra.get("update"))
    ag2, extra2 = MAPPO.load(root / f"results/main_study/seed{s}/final.pt")
    # final.pt does not store kappa/multipliers; use trained-at values (kappa=1, lam from constraint state)
    lam_f = ag2.constraints.multipliers()
    ep2, _, _ = run(None, ag2, {"residual_scale": 1.0, "multipliers": lam_f})
    out[f"mappo{s}"]["final_cost"] = ep2["net_cost_eur"]; out[f"mappo{s}"]["final_unserved"] = ep2["unserved_kwh"]
for k, v in out.items(): print(k, v)

# ---- LP variants: patch the source of perfect_foresight_lp ----
src = textwrap.dedent(inspect.getsource(B.perfect_foresight_lp))
def lp_variant(degr=False, terminal=False):
    s = src
    if degr:
        s = s.replace('c[off["bc"]:off["bc"] + nb] += 1e-4 * dt',
                      'c[off["bc"]:off["bc"] + nb] += cfg.bess_degradation_eur_per_kwh * dt')
        s = s.replace('c[off["bd"]:off["bd"] + nb] += 1e-4 * dt',
                      'c[off["bd"]:off["bd"] + nb] += cfg.bess_degradation_eur_per_kwh * dt')
        s = s.replace("cost = float(((buy * gi - sell * ge + buy * us * mu) * dt).sum())",
                      "cost = float(((buy * gi - sell * ge + buy * us * mu) * dt).sum()) + cfg.bess_degradation_eur_per_kwh*dt*float(res.x[off['bc']:off['bc']+2*nb].sum())")
    if terminal:
        s = s.replace("    res = linprog(",
                      "    for i in range(N):\n        lo[ieb(T-1, i)] = max(lo[ieb(T-1, i)], float(ct.bess_soc0[i]))\n    res = linprog(", 1)
    ns = dict(B.__dict__); exec(s, ns)
    return ns["perfect_foresight_lp"](ct, env_cfg)["optimal_cost_eur"]

lps = {"as_is": lp_variant(), "+degradation": lp_variant(degr=True),
       "+terminal_soc>=initial": lp_variant(terminal=True),
       "+both": lp_variant(True, True)}
print("LP variants:", {k: round(v, 2) for k, v in lps.items()})
grid = 175.889
for k, v in lps.items():
    rec = [(grid - out[f"mappo{s}"]["cost"]) / (grid - v) for s in range(5)]
    print(f"  savings recovered vs LP[{k}]: MAPPO mean {np.mean(rec):.3f}, heuristic {(grid-out['heuristic']['cost'])/(grid-v):.3f}")
json.dump({"policies": out, "lp": lps}, open(sys.argv[1], "w"), indent=1, default=str)
