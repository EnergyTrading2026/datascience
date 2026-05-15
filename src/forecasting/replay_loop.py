"""Hourly forecast replay loop.

This is the entry point for the forecasting Docker container. There is no
live demand feed yet, so we simulate one by replaying the last
``REPLAY_LOOKBACK_MONTHS`` of the historical CSV: each wall-clock tick the
virtual ``solve_time`` advances by one hour, the configured model produces
a forecast from the history up to that point, and the file is written to
``FORECAST_DIR`` for the optimization daemon to consume. When the virtual
clock reaches the end of the CSV it wraps back to ``csv_end - lookback``.

The optimization daemon reads the same ``FORECAST_DIR`` (bind-mounted from
``data/forecast/`` on the host). Filenames use hyphens in the time portion
to match its scanner regex.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Type

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
            heartbeat_path=forecast_dir / ".forecasting-heartbeat",
            model_name=os.environ.get("MODEL", "combined_seasonal"),
            horizon_hours=int(os.environ.get("HORIZON_HOURS", "35")),
            lookback_months=int(os.environ.get("REPLAY_LOOKBACK_MONTHS", "3")),
            tick_interval_s=int(os.environ.get("TICK_INTERVAL_S", "3600")),
        )


class ReplayState:
    """Walks the virtual solve_time forward, wrapping at csv_end."""

    def __init__(self, csv_end: pd.Timestamp, lookback_months: int):
        self._csv_end = csv_end
        # pandas DateOffset(months=n) is calendar-aware (handles month length
        # variance) which is what "last N months" naturally means.
        self._replay_start = csv_end - pd.DateOffset(months=lookback_months)
        self.virt_solve_time = self._replay_start

    def advance(self) -> pd.Timestamp:
        """Return the current solve_time, then advance by one hour (wrap on overflow)."""
        current = self.virt_solve_time
        nxt = current + pd.Timedelta(hours=1)
        if nxt >= self._csv_end:
            logger.info(
                "replay window exhausted at %s, wrapping back to %s",
                self._csv_end.isoformat(), self._replay_start.isoformat(),
            )
            self.virt_solve_time = self._replay_start
        else:
            self.virt_solve_time = nxt
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


def _tick(cfg: Config, history: pd.DataFrame, model: BaseForecaster, state: ReplayState) -> None:
    solve_time = state.advance()
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
