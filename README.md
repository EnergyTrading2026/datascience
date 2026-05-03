# Data Science

Forecasting and optimization for district heating dispatch — EnergyTrading2026 project seminar.

## Teams

- **Forecasting** — demand forecasts (hourly, multi-horizon)
- **Optimization** — hourly MPC dispatch on top of the forecast

## Setup

Requires **Python ≥ 3.11**.

```bash
git clone git@github.com:EnergyTrading2026/datascience.git
cd datascience
pip install -e .
```

This installs the optimization stack (`numpy`, `pandas`, `pyomo`, `highspy`).
Forecasting notebooks pull in additional packages (`scikit-learn`, `statsmodels`,
`tensorflow`, `pmdarima`, `xgboost`) that you'll need to install separately.

## Project Structure

```
src/            Production code (forecasting + optimization)
notebooks/      Jupyter notebooks for exploration and prototyping
tests/          Automated tests (src/ only — notebooks aren't tested)
docs/           Problem definitions and methodology notes
data/           Data (raw and cleaned)
```

Optimization context lives in `docs/optimization/`: problem statement
(`optimization_problem.md`) and hourly MPC notes (`hourly_mpc.md`).

## Entry points

- `optimization-backtest` — runs the hourly MPC backtest harness
  (defined in `pyproject.toml`, see `src/optimization/backtest.py`).

## Workflow

1. Create a feature branch from `main`
2. Make your changes
3. Open a pull request
4. Get 1 approval
5. Merge
