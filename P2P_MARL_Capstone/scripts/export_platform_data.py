"""Record every controller variant in every preset scenario for the replay platform.

    python scripts/export_platform_data.py      # ~2 min -> platform/data/traces.js

The replay page (platform/index.html opened from disk, or the published copy)
needs no Python: it plays these recordings.  Custom communities need the live
server (python platform/server.py).
"""
from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

import torch  # noqa: E402

torch.set_num_threads(1)

from src.platform import sim  # noqa: E402
from src.platform.custom import build_community, default_spec  # noqa: E402


def main():
    t0 = time.time()
    ct, cfg, meta = build_community(default_spec())
    x = sim.settlement_weights(ct, cfg)
    models = sim.load_models(0)
    out = {"meta": {"households": meta, "feeder_limit_kw": cfg.feeder_import_kw_per_hh * ct.n_agents,
                    "generated": date.today().isoformat(), "model_seed": 0,
                    "controllers": sim.controllers(), "scenarios": sim.scenarios()},
           "scenarios": {}}
    for key, label in sim.scenarios().items():
        res = sim.compare(ct, cfg, key, models, x_core=x, with_frames=True)
        out["scenarios"][key] = {"label": label, "runs": res.pop("runs"), "compare": res}
        best = min(res["rows"], key=lambda r: r["cost_battery_neutral"])
        print(f"{label:34s} best: {best['label']} EUR {best['cost_battery_neutral']:.2f}  "
              f"[{(time.time() - t0) / 60:.1f} min]", flush=True)
    p = root / "platform/data/traces.js"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("window.TRACES = " + json.dumps(out, separators=(",", ":")) + ";\n")
    print(f"written {p} ({p.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
