# Smooth Partial Lotteries

This repository contains code for smooth randomized selection rules from *Smooth Partial Lotteries*.

Core implementation lives in `smooth_lotteries.py` and includes:
- `linear_lottery` / `linear_lottery_smooth`
- `softmax_topk` / `softmax_topk_smooth`
- `systematic_sampling`

## Repository Structure

- `smooth_lotteries.py`: core smooth lottery mechanisms implementation.
- `Getting_Started_Smooth.ipynb`: simple demo notebook for running smooth lotteries.
- `experiments/`: scripts, results, and figures for the paper experiments.
- `baselines/`: clone of <https://github.com/akgoldberg/lottery> used for baseline algorithms and baseline datasets.

## Quick Start

From the repository root:

```bash
pip install numpy matplotlib pandas scipy jupyter
```

Then open:

```bash
jupyter notebook Getting_Started_Smooth.ipynb
```

## Experiments

For experiment details and run commands, see:
- `experiments/EXPERIMENTS_README.md`

## Notes

- Baseline mechanisms (e.g., MERIT / Swiss NSF baseline code paths) are maintained in `baselines/`.
- The root notebook demo intentionally focuses on `smooth_lotteries.py` only.
