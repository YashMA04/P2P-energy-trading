#!/usr/bin/env bash
# Full v2 protocol, resumable: every run is skipped if its summary already exists.
#   bash scripts/run_v2_protocol.sh
# Then:  python scripts/final_eval_v2.py
set -u
cd "$(dirname "$0")/.."
run() {  # $1 = summary path, rest = command
  local done_file="$1"; shift
  if [ -f "$done_file" ]; then echo "SKIP (done) $done_file"; return; fi
  echo "START $(date -u +%H:%M) $*"
  "$@" 2>&1 | grep --line-buffered -E "^\[|collected|selected|MPC|saved|Traceback|Error" || true
  [ -f "$done_file" ] && echo "DONE  $(date -u +%H:%M) $done_file" || echo "FAILED $done_file"
}
# 1. exact teacher data (2 x 48 perturbed days, reporting day excluded)
run results/v2/teacher/teacher_s0.npz python scripts/make_teacher.py --days 48 --seed 0 --out results/v2/teacher/teacher_s0.npz
run results/v2/teacher/teacher_s1.npz python scripts/make_teacher.py --days 48 --seed 1 --out results/v2/teacher/teacher_s1.npz
# 2. comparators
run results/v2/mpc_h12.json python scripts/run_mpc.py --horizon 12 --out results/v2/mpc_h12.json
# 3. oracle-guided students, 3 seeds each
for s in 0 1 2; do
  run results/v2/dagger_mappo/seed$s/summary.json python scripts/train_dagger.py --arch mappo --seed $s --iters 4 --days 8 --out results/v2/dagger_mappo
  run results/v2/dagger_mat/seed$s/summary.json   python scripts/train_dagger.py --arch mat --order laxity --seed $s --iters 4 --days 8 --out results/v2/dagger_mat
done
# 4. decoding-order ablation (random fixed permutation), 3 seeds, 2 DAgger iterations
for s in 0 1 2; do
  run results/v2/dagger_mat_random/seed$s/summary.json python scripts/train_dagger.py --arch mat --order random --seed $s --iters 2 --days 8 --out results/v2/dagger_mat_random
done
# 5. anchored PPO fine-tuning of the seed-0 students
run results/v2/ft_mappo_gentle/seed0/summary.json python scripts/finetune_rl.py --init results/v2/dagger_mappo/seed0/iter4.pt --seed 0 --updates 60 --warmup 20 --lr-actor 1e-5 --log-std -3.0 --anchor 1.0 --anchor-final 1.0 --out results/v2/ft_mappo_gentle
echo "PROTOCOL COMPLETE"
