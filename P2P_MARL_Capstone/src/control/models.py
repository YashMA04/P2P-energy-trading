"""Single source of truth for which checkpoint is "the final model" of each method.

The selection rule is applied on held-out noisy VALIDATION days only, never on
the reporting day:

* MAT (final method) -- DAgger student, then anchored PPO fine-tuning (the RL
  stage).  The pre-RL DAgger checkpoint is itself a candidate: if fine-tuning
  does not improve validation cost, the pre-RL model is kept.
* MAPPO student -- DAgger student (the decentralised fallback).
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results/v2"


def dagger_ckpt(arch: str, seed: int) -> Path:
    d = R / f"dagger_{arch}/seed{seed}"
    it = json.load(open(d / "summary.json"))["selected_iter_on_val"]
    return d / f"iter{it}.pt"


def final_mat_ckpt(seed: int) -> tuple:
    """(path, label) of the final MAT model for a seed."""
    pre = dagger_ckpt("mat", seed)
    ft = R / f"ft_mat/seed{seed}/summary.json"
    if not ft.exists():
        return pre, "DAgger (no fine-tuning run)"
    j = json.load(open(ft))
    if j["best"]["val"]["net_cost_inv_neutral_eur"] < j["start"]["val"]["net_cost_inv_neutral_eur"]:
        return ft.parent / "best.pt", f"DAgger + PPO fine-tuning (update {j['best'].get('update')})"
    return pre, "DAgger (fine-tuning rejected on validation)"


def final_mappo_ckpt(seed: int) -> Path:
    return dagger_ckpt("mappo", seed)
