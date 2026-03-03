# Experiments Runbook

Shared code path:
- `experiments/data_utils.py`: dataset loading, normalization to `[0,1]`, review-scale metadata, and synthetic data generation.
- `experiments/mechanisms.py`: unified mechanism wrappers (Linear Lottery, Softmax, MERIT, Swiss NSF, threshold/randomized baselines) and interval conversion helpers.
- `experiments/utils.py`: shared normalization and dataset helpers.
- `experiments/plot_results.py`: centralized plotting for regret/smoothness figures.

## Run Main Suite
```bash
./experiments/run_all_experiments.sh
```
This runs:
1. `experiments/regret.py` (with `--clear_old`)
2. `experiments/utility_ccdf.py`
3. `experiments/smoothness.py` (empirical estimate of global smoothness)
4. `experiments/local_smoothness.py` (empirical estimate of single-item perturbation local smoothness)
5. `experiments/baseline_local_sensitivity.py` (existing baseline partial-lottery local sensitivity + table)

Expected runtime:
- Full suite: typically ~45 to 70 minutes on a laptop CPU (dominated by local smoothness + MERIT baseline runs).

## Regret (`regret.py`)
Goal:
- Measure regret vs smoothness level `L` across real and synthetic datasets, and compare Linear Lottery vs Softmax under different acceptance-rate settings.

High-level setup:
- Regret vs target smoothness `L` on `Beta`, `ICLR`, `NeurIPS`, `Swiss NSF`.
- `k ∈ {1, 10%, 33%, 50%}`.
- Includes the symmetric Beta sweep (`alpha=beta`) used in current analysis.

Run:
```bash
python experiments/regret.py --clear_old
```

Expected runtime:
- ~8 to 15 minutes (depends on softmax sample settings and machine).

Outputs:
- `experiments/results/regret_vs_L_*.csv`
- `experiments/results/regret_vs_L_drop_*.csv`
- `experiments/results/regret_beta_sweep_*.csv`
- `experiments/figures/regret_vs_L_*.pdf`
- `experiments/figures/regret_beta_sweep_*.pdf`

## Utility CCDF (`utility_ccdf.py`)
Goal:
- Compare utility-tail structure across datasets to interpret why regret differs by dataset.

High-level setup:
- CCDF of normalized mean utilities across the four datasets.

Run:
```bash
python experiments/utility_ccdf.py
```

Expected runtime:
- <1 minute.

Outputs:
- `experiments/results/utility_ccdf_summary.csv`
- `experiments/figures/utility_ccdf_datasets.pdf`

## Global Smoothness (`smoothness.py`)
Goal:
- Empirically assess tightness of global smoothness bounds for Linear Lottery and Softmax on worst-case-inspired constructions.

High-level setup:
- Global smoothness search for Linear Lottery and Softmax.
- Produces empirical smoothness curves and ratio-to-target-`L` curves for `n=100,1000`.
- Also produces epsilon-response diagnostics (`n=100`, `k=10%`, `L=1`) with three panels:
  - near-uniform (`\Delta p_1`)
  - worst-case family around the threshold item (`\Delta p_k`)
  - top-`k` edge perturbation (`\Delta p_k`)
- Near-worst-case construct used in the search:
  - Base utilities: `u_1,...,u_{k-1}=1`, `u_k=B`, `u_{k+1},...,u_n=0`.
  - Perturbation: change only the threshold item (`k`-th item), `u_k -> u_k ± ε` (clipped to `[0,1]`).
  - Grid search over `B` and `ε` (and perturbation direction), then re-estimate the best candidate with larger Monte Carlo for Softmax.

Run:
```bash
python experiments/smoothness.py \
  --softmax-search-samples 5000 \
  --softmax-final-samples 50000
```

Expected runtime:
- ~8 to 15 minutes.

Outputs:
- `experiments/results/global_smoothness_summary.csv`
- `experiments/results/epsilon_response_n100_k10_L1.csv`
- `experiments/figures/global_smoothness_n100_empirical_vs_targetL.pdf`
- `experiments/figures/global_smoothness_n1000_empirical_vs_targetL.pdf`
- `experiments/figures/global_smoothness_n100_ratio_vs_targetL.pdf`
- `experiments/figures/global_smoothness_n1000_ratio_vs_targetL.pdf`
- `experiments/figures/epsilon_response_n100_k10_L1.pdf`

## Local Smoothness (`local_smoothness.py`)
Goal:
- Estimate local sensitivity near the decision boundary under realistic one-review perturbations in each dataset.

High-level setup:
- Single-review, one-tick perturbations at rank `k` and `k+1` for each dataset.
- `k ∈ {10%, 33%, 50%}` and `L ∈ {0.2,0.4,0.6,0.8,1.0}`.
- Softmax uses two-stage estimation:
  - search phase to choose perturbation
  - fresh final resampling for the reported value

Perturbation used (exact):
- Candidate items are only the utility-ranked boundary items: rank `k` and rank `k+1`.
- For each candidate item, change only one observed review entry (the first observed review).
- Evaluate two one-tick edits: `+tick` and `-tick`, clipped to `[0,1]`; report the larger local effect.
- Tick in normalized units:
  - ICLR/NeurIPS (raw 1–10): `1/9`
  - Swiss NSF (raw 1–6): `1/5`
  - Synthetic Beta: treated as 10 ticks on `[0,1]` => `1/9`
- If clipping is active, the actual perturbation size is smaller and recorded as `l11_delta` in the CSV.

Recommended final run (low-noise):
```bash
python experiments/local_smoothness.py \
  --k_names k10pct,k33pct,k50pct \
  --L_values 0.2,0.4,0.6,0.8,1.0 \
  --softmax_search_samples 800 \
  --softmax_search_reps 2 \
  --softmax_final_samples 10000 \
  --softmax_final_reps 3
```

Expected runtime:
- ~20 to 35 minutes (highest-variance cost component).

Outputs:
- `experiments/results/local_smoothness_all.csv`
- `experiments/results/local_smoothness_all_summary.csv`
- `experiments/figures/local_smoothness_k10pct.pdf`
- `experiments/figures/local_smoothness_k33pct.pdf`
- `experiments/figures/local_smoothness_k50pct.pdf`

## Baseline Existing Partial-Lottery Sensitivity (`baseline_local_sensitivity.py`)
Goal:
- Compare worst-case local sensitivity (single-review one-tick perturbation) for existing baseline partial lotteries.

High-level setup:
- Mechanisms: existing baseline partial lotteries (`MERIT`, `Swiss NSF`, `Randomized Threshold`).
- Uses normalized review data and leave-one-out intervals.
- For each candidate item, picks the one-review edit that maximally shifts that item's interval representation.
- Candidate search:
  - `MERIT`: focused around boundary (`k/k+1` by default).
  - `Swiss NSF` and `Randomized Threshold`: all items (`--swiss_candidate_window -1 --threshold_candidate_window -1`).
- Outputs a LaTeX table from the same summary.

Run:
```bash
python experiments/baseline_local_sensitivity.py \
  --swiss_candidate_window -1 \
  --threshold_candidate_window -1
```

Expected runtime:
- ~15 to 30 minutes (MERIT LP solves dominate).

Outputs:
- `experiments/results/baseline_local_sensitivity.csv`
- `experiments/results/baseline_local_sensitivity_summary.csv`
- `experiments/figures/baseline_local_sensitivity_table.tex`
