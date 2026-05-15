"""Hourly forecast replay loop.

This is the entry point for the forecasting Docker container. There is no
live demand feed yet, so the MVP runs against the historical CSV: starting
from ``csv_end - REPLAY_LOOKBACK_MONTHS`` the virtual ``solve_time`` advances
by one hour per tick, the configured model produces a forecast from the
history up to that point, and the file is written to ``FORECAST_DIR`` for
the optimization daemon to consume. When the virtual clock reaches
``csv_end`` the loop **idles** — no more forecasts, but the heartbeat keeps
ticking so the healthcheck stays green.

The loop deliberately does NOT wrap back to the start: the optimization
daemon's monotonicity check (``solve_time > last_solve_time``) would block
all post-wrap files, so a wrap would silently freeze the pipeline. To run
the MVP again, clear ``data/forecast``, ``data/state`` and ``data/dispatch``
(see ``scripts/reset_demo.sh``) and restart both stacks.

Filename format uses hyphens in the time portion to match the daemon's
scanner regex (Windows-safe).
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Type

import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler

from forecasting.base_model import BaseForecaster
from forecasting.baseline_models import (
    CombinedSeasonalForecaster,
    DailyNaiveForecaster,
    WeeklyNaiveForecaster,
)
from forecasting.data_cleaning import load_and_clean_data
from forecasting.export import export_forecast
from forecasting.fill_missing_data import fill_missing_linear

logger = logging.getLogger("forecasting.replay")

FORECAST_FILENAME_SUFFIX = ".parquet"
HEARTBEAT_NAME = ".forecasting-heartbeat"

MODEL_REGISTRY: dict[str, Type[BaseForecaster]] = {
    "daily_naive": DailyNaiveForecaster,
    "weekly_naive": WeeklyNaiveForecaster,
    "combined_seasonal": CombinedSeasonalForecaster,
}


@dataclass
class Config:
    csv_path: Path
    forecast_dir: Path
    heartbeat_path: Path
    model_name: str
    horizon_hours: int
    lookback_months: int
    tick_interval_s: int

    @classmethod
    def from_env(cls) -> "Config":
        forecast_dir = Path(os.environ.get("FORECAST_DIR", "/shared/forecast"))
        return cls(
            csv_path=Path(os.environ.get(
                "CSV_PATH", "/shared/forecasting/raw/raw_data_measured_demand.csv"
            )),
            forecast_dir=forecast_dir,
            heartbeat_path=forecast_dir / HEARTBEAT_NAME,
            model_name=os.environ.get("MODEL", "combined_seasonal"),
            horizon_hours=_positive_int_env("HORIZON_HOURS", "35"),
            lookback_months=_positive_int_env("REPLAY_LOOKBACK_MONTHS", "3"),
            tick_interval_s=_positive_int_env("TICK_INTERVAL_S", "3600"),
        )


def _positive_int_env(name: str, default: str) -> int:
    raw = os.environ.get(name, default)
    try:
        value = int(raw)
    except ValueError as e:
        raise ValueError(f"{name} must be an integer; got {raw!r}") from e
    if value <= 0:
        raise ValueError(f"{name} must be positive; got {value}")
    return value


class ReplayState:
    """Walks the virtual solve_time forward through the lookback window.

    No wrap-around: once ``virt_solve_time`` reaches ``csv_end`` the state
    is exhausted and ``advance()`` returns ``None`` forever after.
    """

    def __init__(self, csv_end: pd.Timestamp, lookback_months: int):
        if lookback_months <= 0:
            raise ValueError(f"lookback_months must be positive; got {lookback_months}")
        self._csv_end = csv_end
        # pandas DateOffset(months=n) is calendar-aware (handles month length
        # variance) which is what "last N months" naturally means.
        self._replay_start = csv_end - pd.DateOffset(months=lookback_months)
        if self._replay_start >= csv_end:
            raise ValueError(
                f"replay_start {self._replay_start.isoformat()} is not before "
                f"csv_end {csv_end.isoformat()}; lookback too large or csv too short"
            )
        self.virt_solve_time = self._replay_start
        self._exhausted_logged = False

    def advance(self) -> Optional[pd.Timestamp]:
        """Return the next solve_time to forecast for, or None if exhausted."""
        if self.virt_solve_time >= self._csv_end:
            if not self._exhausted_logged:
                logger.info(
                    "replay window exhausted at %s; idling — heartbeat continues",
                    self._csv_end.isoformat(),
                )
                self._exhausted_logged = True
            return None
        current = self.virt_solve_time
        self.virt_solve_time = current + pd.Timedelta(hours=1)
        return current


def _load_history(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"forecast CSV not found at {csv_path}")
    logger.info("loading history from %s", csv_path)
    raw = load_and_clean_data(str(csv_path))
    return fill_missing_linear(raw)


def _build_model(name: str) -> BaseForecaster:
    try:
        cls = MODEL_REGISTRY[name]
    except KeyError as e:
        raise ValueError(
            f"unknown MODEL={name!r}; choices: {sorted(MODEL_REGISTRY)}"
        ) from e
    return cls()


def _touch_heartbeat(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    now = time.time()
    os.utime(path, (now, now))


def _cleanup_forecast_dir(forecast_dir: Path) -> None:
    """Remove leftover *.parquet and the heartbeat before a fresh run.

    A previous container's files would otherwise mix with this run's output,
    and the daemon's monotonicity check would block everything older than the
    last-seen file from the previous run.
    """
    if not forecast_dir.is_dir():
        return
    removed = 0
    for entry in forecast_dir.iterdir():
        if not entry.is_file():
            continue
        if entry.suffix == FORECAST_FILENAME_SUFFIX or entry.name == HEARTBEAT_NAME:
            try:
                entry.unlink()
                removed += 1
            except OSError as e:
                logger.warning("could not remove %s: %s", entry, e)
    if removed:
        logger.info("startup cleanup: removed %d file(s) from %s", removed, forecast_dir)


def _tick(cfg: Config, history: pd.DataFrame, model: BaseForecaster, state: ReplayState) -> None:
    solve_time = state.advance()
    if solve_time is None:
        # Idle: bump heartbeat so the healthcheck stays green.
        _touch_heartbeat(cfg.heartbeat_path)
        return
    cycle_history = history[history.index < solve_time]
    if cycle_history.empty:
        logger.warning("no history before %s; skipping tick", solve_time.isoformat())
        return
    logger.info(
        "tick: solve_time=%s history_rows=%d model=%s horizon=%dh",
        solve_time.isoformat(), len(cycle_history), cfg.model_name, cfg.horizon_hours,
    )
    predictions = model.predict(cycle_history, solve_time, horizon=cfg.horizon_hours)
    out_path = export_forecast(predictions, solve_time, output_dir=str(cfg.forecast_dir))
    _touch_heartbeat(cfg.heartbeat_path)
    logger.info("tick done: wrote %s", out_path)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = Config.from_env()
    logger.info(
        "forecasting replay starting: csv=%s out=%s model=%s horizon=%dh "
        "lookback=%dmo tick=%ds",
        cfg.csv_path, cfg.forecast_dir, cfg.model_name, cfg.horizon_hours,
        cfg.lookback_months, cfg.tick_interval_s,
    )

    history = _load_history(cfg.csv_path)
    csv_end = history.index.max()
    if pd.isna(csv_end):
        logger.error("CSV has no data after cleaning; aborting")
        return 1
    state = ReplayState(csv_end=csv_end, lookback_months=cfg.lookback_months)
    model = _build_model(cfg.model_name)

    cfg.forecast_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_forecast_dir(cfg.forecast_dir)
    _touch_heartbeat(cfg.heartbeat_path)

    # Fire once immediately so a fresh container produces a forecast without
    # waiting a full tick interval.
    _tick(cfg, history, model, state)

    scheduler = BackgroundScheduler(timezone="UTC")
    scheduler.add_job(
        _tick, "interval", args=[cfg, history, model, state],
        seconds=cfg.tick_interval_s, id="replay-tick",
        max_instances=1, coalesce=True,
    )
    scheduler.start()

    shutdown = {"now": False}

    def _on_sig(signum, _frame):
        logger.info("received signal %s, shutting down", signum)
        shutdown["now"] = True

    signal.signal(signal.SIGTERM, _on_sig)
    signal.signal(signal.SIGINT, _on_sig)

    try:
        while not shutdown["now"]:
            time.sleep(1)
    finally:
        scheduler.shutdown(wait=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
