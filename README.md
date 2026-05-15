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
pip install -e '.[optimization,forecasting]'
```

Base deps are the shared minimum (`numpy`, `pandas`, `pyarrow`).
Component-specific deps live behind extras so each Docker image installs
only what it needs:

- `[optimization]` — `pyomo`, `highspy`, `apscheduler` (solver stack)
- `[forecasting]` — `apscheduler` (drives the hourly replay loop)
- `[notebooks]` — `jupyter`, `ipykernel`, `matplotlib`
- `[dev]` — `pytest`

For local development install both component extras as shown above (or
`pip install -e '.[optimization,forecasting,notebooks,dev]'` for everything).
Forecasting notebooks pull in additional packages (`scikit-learn`, `statsmodels`,
`tensorflow`, `pmdarima`, `xgboost`) that you'll need to install separately.

This editable install is the intended setup for local development and tests.
The intended production path for the optimization service is Docker — see
`docs/deploy/README.md`.

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
- Dockerized optimization service — see `docs/deploy/README.md` for the
  production daemon, first-time setup, synthetic forecast smoke test, and
  operational commands.
- Dockerized forecasting service — see
  `docs/forecasting/hourly_inference_pipeline.md` for the hourly replay-loop
  container (a separate Compose stack that drops forecast parquets into the
  same `data/forecast/` directory the optimization daemon reads).
- `scripts/sim_forecaster.py` — local smoke-test helper that writes a
  contract-shaped forecast parquet into `data/forecast/` so the Docker daemon
  can run a full optimization cycle without the real forecasting service.
- `scripts/reset_demo.sh` — wipe both stacks plus shared state and bring
  them back up; required when restarting the replay demo from the beginning.

## Workflow

1. Create a feature branch from `main`
2. Make your changes
3. Open a pull request
4. Get 1 approval
5. Merge
