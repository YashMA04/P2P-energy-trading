"""Generate privileged-teacher data: exact dispatch on perturbed days.

    python scripts/make_teacher.py --days 48 --seed 0 --out results/v2/teacher/teacher_s0.npz
Day 0 is the nominal (deterministic) day only if --include-nominal is passed;
by default the reported evaluation day is EXCLUDED from the teacher data, so
the student never trains on the exact day it is reported on.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
root = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(root))
from src.data.community import build_tensors, load_community, stratified_subset
from src.opt.teacher import generate_teacher_data
from src.utils.config import load_config, build_configs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v2.yaml")
    ap.add_argument("--days", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--include-nominal", action="store_true")
    ap.add_argument("--out", default="results/v2/teacher/teacher_s0.npz")
    a = ap.parse_args()
    cfg = load_config(root / a.config)
    ct = build_tensors(stratified_subset(load_community(root / cfg["data"]["path"]),
                                         cfg["data"]["households"], seed=cfg["data"]["subset_seed"]))
    env_cfg, _, _ = build_configs(cfg)
    d = generate_teacher_data(ct, env_cfg, a.days, seed=a.seed, noise_std=a.noise,
                              include_nominal=a.include_nominal)
    out = root / a.out; out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **d)
    di = d["day_info"]
    print(f"saved {out}: {di.shape[0]} days | LP mean {di[:, 0].mean():.2f} | sim mean {di[:, 1].mean():.2f} "
          f"| max unserved {di[:, 3].max():.3f} | max shortfall {di[:, 4].max():.3f}")


if __name__ == "__main__":
    main()
