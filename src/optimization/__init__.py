"""District-heating dispatch optimization (production code).

Hourly MPC: every hour, fetch latest demand forecast (forecasting team) +
published DA electricity prices (SMARD), build a 35h MILP, solve, commit the
first hour of dispatch and persist the carry-over state.

Entry point: `python -m optimization.run --help`.
"""
