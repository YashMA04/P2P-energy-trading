#!/usr/bin/env bash
# Anchored PPO fine-tuning of the three oracle-guided MAT students (RL stage).
#   bash scripts/run_ft_mat.sh
set -u; cd "$(dirname "$0")/.."
for s in 0 1 2; do
  it=$(python -c "import json;print(json.load(open('results/v2/dagger_mat/seed$s/summary.json'))['selected_iter_on_val'])")
  [ -f results/v2/ft_mat/seed$s/summary.json ] && { echo "SKIP seed$s"; continue; }
  echo "START seed$s from iter$it $(date -u +%H:%M)"
  python scripts/finetune_rl.py --init results/v2/dagger_mat/seed$s/iter$it.pt --seed $s --updates 60 \
      --warmup 20 --lr-actor 1e-5 --log-std -3.0 --anchor 1.0 --anchor-final 1.0 --out results/v2/ft_mat 2>&1 \
      | grep --line-buffered -E "^\[|Traceback|Error"
  echo "DONE seed$s $(date -u +%H:%M)"
done
echo "FT COMPLETE"
