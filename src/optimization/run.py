"""CLI entrypoint for one hourly MPC dispatch cycle.

Usage:
    python -m optimization.run \\
        --solve-time 2026-04-28T13:00:00+02:00 \\
        --forecast-path /shared/forecast/latest.parquet \\
        --prices-path  /shared/smard/latest.csv \\
        --state-in     /shared/state/current.json \\
        --state-out    /shared/state/current.json \\
        --dispatch-out /shared/dispatch/<solve_time>.parquet

DevOps wraps this in a container + scheduler. We don't ship the scheduler.

Exit codes:
    0 = success, dispatch + state written
    1 = recoverable failure (forecast/price file missing, horizon too short) — DevOps retries
    2 = solver infeasible — alert, manual intervention needed
    3 = unexpected error — page
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from optimization.adapters import forecast as forecast_io
from optimization.adapters import smard as smard_io
from optimization.config import PlantParams, RuntimeConfig
from optimization.dispatch import extract_dispatch, extract_state
from optimization.model import build_model
from optimization.solve import SolverInfeasibleError, solve
from optimization.state import DispatchState

logger = logging.getLogger("optimization.run")

INT_PER_HOUR = 4


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one hourly MPC dispatch cycle.")
    p.add_argument(
        "--solve-time",
        required=True,
        type=lambda s: pd.Timestamp(s),
        help="Solve timestamp, ISO8601 with timezone (e.g. 2026-04-28T13:00:00+02:00).",
    )
    p.add_argument("--forecast-path", required=True, type=Path)
    p.add_argument(
        "--prices-path",
        required=True,
        type=Path,
        help="SMARD CSV path (semicolon-separated). V2 may switch to live API.",
    )
    p.add_argument(
        "--state-in",
        type=Path,
        help="Path to read carry-over state. Required unless --cold-start.",
    )
    p.add_argument("--state-out", required=True, type=Path)
    p.add_argument("--dispatch-out", required=True, type=Path)
    p.add_argument(
        "--cold-start",
        action="store_true",
        help="Ignore --state-in and start with default state (first deployment).",
    )
    return p.parse_args(argv)


def run_one_cycle(
    solve_time: pd.Timestamp,
    forecast_path: Path,
    prices_path: Path,
    state_in: Path | None,
    state_out: Path,
    dispatch_out: Path,
    cold_start: bool,
    params: PlantParams | None = None,
    runtime: RuntimeConfig | None = None,
) -> int:
    """Run one hourly cycle. Returns exit code (0/1/2)."""
    if solve_time.tzinfo is None:
        logger.error("solve-time must be tz-aware")
        return 1

    params = params or PlantParams()
    rt = runtime or RuntimeConfig()

    # 1. Load state
    if cold_start:
        state = DispatchState.cold_start(solve_time)
        logger.info("cold start: SoC=%.1f, all units off", state.sto_soc_mwh_th)
    else:
        if state_in is None:
            logger.error("--state-in required when not --cold-start")
            return 1
        try:
            state = DispatchState.load(state_in)
        except FileNotFoundError as e:
            logger.error("state file missing: %s", e)
            return 1
        logger.info("loaded state from %s (SoC=%.1f)", state_in, state.sto_soc_mwh_th)

    # 2. Fetch inputs
    try:
        forecast = forecast_io.load_forecast(forecast_path, solve_time)
    except (FileNotFoundError, forecast_io.ForecastSchemaError, ValueError) as e:
        logger.error("forecast load failed: %s", e)
        return 1
    try:
        prices = smard_io.get_published_prices(solve_time, prices_path)
    except (FileNotFoundError, ValueError) as e:
        logger.error("price load failed: %s", e)
        return 1

    # 3. Reconcile horizons
    horizon_h = min(len(forecast), len(prices))
    if horizon_h < rt.horizon_hours_min:
        logger.error(
            "horizon too short: forecast=%dh prices=%dh min=%dh",
            len(forecast), len(prices), rt.horizon_hours_min,
        )
        return 1
    horizon_h = min(horizon_h, rt.horizon_hours_target)
    forecast = forecast.iloc[:horizon_h]
    prices = prices.iloc[:horizon_h]
    logger.info("horizon=%dh (forecast=%dh available, prices=%dh available)",
                horizon_h, len(forecast), len(prices))

    # 4. Build + solve
    try:
        model = build_model(forecast, prices, state, params, demand_safety_factor=rt.demand_safety_factor)
        result = solve(model, time_limit_s=rt.solver_time_limit_s, mip_gap=rt.solver_mip_gap)
    except SolverInfeasibleError as e:
        logger.error("solver infeasible: %s", e)
        return 2
    logger.info(
        "solved in %.2fs, status=%s, objective=%.0f EUR",
        result.solve_time_s, result.status, result.objective_eur or 0,
    )

    # 5. Extract committed dispatch + new state
    commit_intervals = rt.commit_hours * INT_PER_HOUR
    dispatch = extract_dispatch(model, n_intervals=commit_intervals, solve_time=solve_time)
    commit_end = forecast.index[0] + pd.Timedelta(hours=rt.commit_hours)
    new_state = extract_state(model, t_end=commit_intervals, commit_end_time=commit_end)

    # 6. Persist (atomic for state; parquet for dispatch)
    dispatch_out.parent.mkdir(parents=True, exist_ok=True)
    dispatch.to_dataframe().to_parquet(dispatch_out)
    new_state.save(state_out)
    logger.info("wrote dispatch -> %s, state -> %s", dispatch_out, state_out)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return run_one_cycle(
        solve_time=args.solve_time,
        forecast_path=args.forecast_path,
        prices_path=args.prices_path,
        state_in=args.state_in,
        state_out=args.state_out,
        dispatch_out=args.dispatch_out,
        cold_start=args.cold_start,
    )


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        logger.exception("unexpected error")
        sys.exit(3)
