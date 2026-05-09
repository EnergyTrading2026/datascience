"""Synthetic forecast generator for daemon smoke tests.

Mimics the forecasting pipeline by writing parquet files into FORECAST_DIR
in the format the daemon expects: ``<solve_time>.parquet`` with a tz-aware
Europe/Berlin DatetimeIndex and a single ``demand_mw_th`` column.

This is test tooling, not prod. It exists because the real forecasting
service is not yet dockerized; once it is, this script becomes redundant.

Examples:

    # Single drop at the next top-of-hour (typical Mac smoke test)
    python scripts/sim_forecaster.py once

    # Drop a forecast every real hour (multi-hour validation on a Linux VM)
    python scripts/sim_forecaster.py loop --period-s 3600

    # Drop a backdated forecast (past hour, still inside the 2h stale grace)
    # — useful to verify a single cycle without waiting for top-of-hour
    python scripts/sim_forecaster.py once --offset-h 0

The default solve_time is now+1h floored to the hour, which mirrors what the
real forecaster will produce (forecast for the upcoming hour).
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

logger = logging.getLogger("sim_forecaster")
DEMAND_COLUMN = "demand_mw_th"
DEFAULT_HORIZON_HOURS = 35


def write_forecast(
    forecast_dir: Path,
    solve_time_utc: pd.Timestamp,
    demand_mw_th: float,
    horizon_hours: int,
) -> Path:
    forecast_dir.mkdir(parents=True, exist_ok=True)
    solve_time_berlin = solve_time_utc.tz_convert("Europe/Berlin")
    idx = pd.date_range(
        solve_time_berlin, periods=horizon_hours, freq="1h", tz="Europe/Berlin"
    )
    fname = f"{solve_time_utc.strftime('%Y-%m-%dT%H:%M:%SZ')}.parquet"
    out = forecast_dir / fname
    df = pd.DataFrame({DEMAND_COLUMN: [demand_mw_th] * horizon_hours}, index=idx)
    # Atomic write so the daemon never sees a partial parquet.
    tmp = out.with_suffix(out.suffix + ".tmp")
    df.to_parquet(tmp)
    tmp.replace(out)
    logger.info(
        "wrote %s (solve_time=%s, demand=%.1f MW_th, horizon=%dh)",
        out, solve_time_utc.isoformat(), demand_mw_th, horizon_hours,
    )
    return out


def _next_solve_time_utc(offset_hours: int) -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").floor("h") + pd.Timedelta(hours=offset_hours)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Synthetic forecast generator for daemon smoke tests."
    )
    p.add_argument(
        "mode",
        choices=("once", "loop"),
        help="once = drop a single forecast and exit; loop = drop one every --period-s.",
    )
    p.add_argument("--forecast-dir", type=Path, default=Path("data/forecast"))
    p.add_argument("--demand-mw-th", type=float, default=10.0)
    p.add_argument("--horizon-hours", type=int, default=DEFAULT_HORIZON_HOURS)
    p.add_argument(
        "--offset-h",
        type=int,
        default=1,
        help="solve_time = floor(now)+offset_h. Default 1 (next hour).",
    )
    p.add_argument(
        "--period-s",
        type=float,
        default=3600.0,
        help="loop mode: seconds between drops (default 3600 = real hourly cadence).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    if args.mode == "once":
        write_forecast(
            args.forecast_dir,
            _next_solve_time_utc(args.offset_h),
            args.demand_mw_th,
            args.horizon_hours,
        )
        return 0

    # loop
    logger.info(
        "loop mode: period=%.1fs. solve_time advances with wall clock; sub-hour periods "
        "will keep producing the same solve_time until the next top-of-hour, which the "
        "daemon dedupes. Use --period-s 3600 for the realistic prod cadence.",
        args.period_s,
    )
    while True:
        write_forecast(
            args.forecast_dir,
            _next_solve_time_utc(args.offset_h),
            args.demand_mw_th,
            args.horizon_hours,
        )
        time.sleep(args.period_s)


if __name__ == "__main__":
    sys.exit(main())
