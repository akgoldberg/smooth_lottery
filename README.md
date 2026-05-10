# Smooth Partial Lotteries

This repository contains the Python implementation for our paper **"Smooth Partial Lotteries"**

The paper studies randomized selection rules for settings such as peer review, grant funding, admissions, and hiring. Existing partial lotteries often use hard score cutoffs: small changes in one review can move a candidate across a boundary and cause a large jump in their chance of selection. This code implements smoother alternatives, where small score changes are guaranteed to cause only bounded changes in selection probabilities.

The main method is the **Linear Lottery**. It assigns each candidate a marginal selection probability from their estimated quality: top candidates may be selected with probability 1, low-scoring candidates with probability 0, and candidates in the middle with probabilities that increase linearly with score. A smoothness parameter `L` controls the tradeoff between stability and selecting the highest-scoring candidates.

## Getting Started

1. **Install Python packages** from the repository root:

```bash
pip install numpy matplotlib pandas scipy jupyter
```

2. **Run the smooth lottery tutorial** in `Getting_Started_Smooth.ipynb`. This notebook is the easiest starting point and shows how to load review data, compute mean utilities, set smoothness parameters, and run both the Linear Lottery and smooth Top-k Softmax.

```bash
jupyter notebook Getting_Started_Smooth.ipynb
```

## Running Smooth Selection Rules

The core implementation is in `smooth_lotteries.py`.

Main functions:
- `linear_lottery_smooth(v, k, L, D_v)` runs the paper's Linear Lottery with the smoothness-calibrated slope `L`.
- `softmax_topk_smooth(v, k, L, D_v)` runs the smooth Top-k Softmax comparison rule.
- `systematic_sampling(k, p)` draws exactly `k` selected candidates from marginal probabilities `p`.

Here `v` is the utility vector, `k` is the number of candidates to select, `L` is the target smoothness level, and `D_v` is the Lipschitz constant of the utility function. For mean review scores with `r` reviews per candidate, use `D_v = 1 / r`.

## Replicating Experiments

Experiment scripts and saved outputs are in `experiments/`. To run the main suite:

```bash
./experiments/run_all_experiments.sh
```

This reproduces the regret, smoothness, local sensitivity, and baseline-comparison experiments used for the paper. For detailed commands, expected runtimes, and output files, see `experiments/EXPERIMENTS_README.md`.

Some baseline comparisons require the extra dependencies described in `baselines/README.md`, because the baseline MERIT implementation solves linear programs. The smooth Linear Lottery and Top-k Softmax experiments only need the lighter setup above.