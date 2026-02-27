# Experiments Runbook

This folder contains experiment scripts and plotting/output utilities.

## 1) Regret Experiments (`regret.py`)

### What this experiment does
- Compares **Linear Lottery** vs **Softmax** regret as a function of smoothness `L`.
- Uses 4 datasets in one run:
  - `beta` (synthetic data)
  - `iclr`
  - `neurips`
  - `swissnsf`
- Uses 3 budget settings:
  - `k=1`
  - `k=10% of n`
  - `k=50% of n`
- Produces 2x2 figures (one panel per dataset) for each `k` setting.

### Script to run
```bash
python experiments/regret.py \
  --experiment cross_dataset \
  --n_softmax_samples 10000 \
  --beta_trials 50 \
  --n_L_points 15
```

### Main parameters
- `--n_softmax_samples`: number of samples to estimate Softmax inclusion probs.
- `--beta_trials`: number of independent synthetic (`beta`) draws.
- `--n_L_points`: number of `L` points per curve.
- `--normalize_by_k` / `--no_normalize_by_k`: plot `Regret/k` vs raw `Regret`.

### Figures + results generated
- `regret_vs_L_cross_{k_config}.pdf`
- `regret_vs_L_cross_{dataset}_{k_config}.csv`
- `regret_vs_L_cross_drop_summary.csv`
- `regret_vs_L_cross_drop_log.csv`

## 2) Regret Re-Plot (`plot_regret_cross.py`)

### What this script does
- Regenerates cross-dataset regret figures from existing CSV outputs.
- Does **not** rerun mechanisms.
- Intended for fast iteration on styling/labels.

### Script to run
```bash
python experiments/plot_regret_cross.py \
  --normalize_by_k \
  --output_dir experiments/results \
  --fig_dir experiments/figures
```

### Figures generated
- `regret_vs_L_cross_{k_config}.pdf`

## 3) Regret Tightness Diagnostics (`regret_tightness.py`)

### What this experiment does
- Computes diagnostics for how tight regret upper bounds are.
- For each `k` config and dataset:
  - ratio diagnostic: `Regret / Upper bound`
  - gap diagnostic: `Upper bound - Regret`
- Uses the same 4 datasets as `regret.py`.

### Script to run
```bash
python experiments/regret_tightness.py \
  --n_softmax_samples 10000 \
  --beta_trials 20 \
  --n_L_points 15
```

### Figures + results generated
- `regret_tightness_ratio_{k_config}.pdf`
- `regret_tightness_gap_{k_config}.pdf`
- `regret_tightness_{dataset}_{k_config}.csv`

## 4) Smoothness Experiments (`smoothness.py`)

### What this experiment does
- Estimates empirical smoothness (local Lipschitz behavior) of mechanisms.
- Two modes:
  - `comparison`: fixed `L`, compare mechanisms.
  - `vs_L`: sweep target `L`.

### Script to run (comparison mode)
```bash
python experiments/smoothness.py \
  --experiment comparison \
  --data neurips \
  --L 2.0 \
  --n_perturbations 200
```

### Script to run (L-sweep mode)
```bash
python experiments/smoothness.py \
  --experiment vs_L \
  --data gaussian \
  --n_perturbations 200
```

### Figures + results generated
- `smoothness_comparison_{data}.pdf`
- `smoothness_hist_{data}.pdf`
- `smoothness_vs_L_{data}.pdf`
- matching CSVs in `experiments/results/`

## Notes
- Plots are written to `experiments/figures/`.
- CSV outputs are written to `experiments/results/`.
- Plot styling utilities live in `experiments/plot_results.py`.
