#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python experiments/regret.py --clear_old
python experiments/utility_ccdf.py

python experiments/smoothness.py \
  --softmax-search-samples 5000 \
  --softmax-final-samples 50000

python experiments/local_sensitivity.py \
  --k_names k10pct,k33pct,k50pct \
  --L_values 0.2,0.4,0.6,0.8,1.0 \
  --softmax_search_samples 800 \
  --softmax_search_reps 2 \
  --softmax_final_samples 10000 \
  --softmax_final_reps 3

python experiments/baseline_local_sensitivity.py \
  --swiss_candidate_window -1 \
  --threshold_candidate_window -1

echo "Regret, utility CCDF, global smoothness, local sensitivity, and baseline local sensitivity experiments complete. See experiments/results and experiments/figures."
