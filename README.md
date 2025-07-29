# Randomized Selection under Uncertainty (Peer Review Lottery)

This directory contains Python code to implement and evaluate methods for randomized selection (peer review lotteries) based on our paper ["A Principled Approach to Randomized Selection under
Uncertainty: Applications to Peer Review and Grant Funding"](https://arxiv.org/pdf/2506.19083).

## Getting Started

1. **Clone the repository**: run `git clone https://github.com/akgoldberg/lottery.git` in your terminal.
2. **Install python packages** (`cd` into the `lottery/` directory and run `pip install -r requirements.txt`).*
3. **Install Gurobi software (if needed)**: our MERIT algorithm uses [Gurobi software](https://www.gurobi.com/) to efficiently solve large linear programs. The installation in step (2) gives you a trial license that can be used to solve smaller problems (for example, selecting among a few hundred candidates.) If you run the code on a larger problem instance, you will likely encounter an error "Model too large for size-limited Gurobi license." If this occurs, please obtain Gurobi with a non-trial license. Academics can obtain a license to this software for free. Follow [instructions from Gurobi](https://www.gurobi.com/academia/academic-program-and-licenses/) to install and activate the license.

*Note: to replicate our synthetic data experiments you will additionally need to install the Python pacakges `cvxpy` and `GPy`, but these are not necessary to run MERIT.

### Running the MERIT Selection Algorithm

In order to run our MERIT selection algorithm on your own data, **follow the example given in `Getting_Started_Example.ipynb`.**

### Replicating Experiments

In order to replicate synthetic data experiments from the paper "A Principled Approach to Randomized Selection under
Uncertainty: Applications to Peer Review and Grant Funding", run python scripts `experiments/dataset_sims.py` (for worst-case model) and `experiments/synthetic_sims.py` (for probabilistic model.) Then, analysis of the generated data is replicable in the iPython notebooks `experiments/dataset_analysis.ipynb` and `experiments/synthetic_analysis.ipynb`.

## Useful Files

- `algorithm/merit.py` — Source code implementing the MERIT algorithm
- `experiments/dataset_sims.py` - Code to implement experiments using Swiss NSF and conference data with worst-case intervals.
- `experiments/synthetic_sims.py` - Code to implement experiments on fully synthetic data under linear miscalibration model.
- `data/SwissNSFData/` — All code to obtain Swiss NSF dataset
- `data/ConferenceReviewData/` — All code to obtain conference data (ICLR 2025 and NeurIPS 2024)
