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
all post-wrap files, so a wrap would silently freeze the pipeline.

``virt_solve_time`` is persisted to ``FORECAST_DIR/.replay-state.json``
after every successful tick. A container restart (crash, ``docker compose
restart``, image rebuild) therefore resumes from where it left off rather
than re-running the window from the start. To run the demo over from the
beginning, clear ``data/forecast``, ``data/state`` and ``data/dispatch``
(see ``scripts/reset_demo.sh``).

Filename format uses hyphens in the time portion to match the daemon's
scanner regex (Windows-safe).
"""

from __future__ import annotations

import json
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

logger = logging.getLogger(__name__)

FORECAST_FILENAME_SUFFIX = ".parquet"
HEARTBEAT_NAME = ".heartbeat"
REPLAY_STATE_NAME = ".replay-state.json"

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
    replay_state_path: Path
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
            replay_state_path=forecast_dir / REPLAY_STATE_NAME,
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
    """Walks virtual solve_time forward through ``[replay_start, csv_end]``.

    Split into ``peek()`` and ``commit()`` so the caller can advance only
    after a forecast has actually been written. A crash between peek and
    commit repeats the same solve_time on the next tick rather than
    silently dropping it.

    ``virt_solve_time`` is persisted to ``state_path`` after every commit
    (atomic ``.tmp`` + rename). On construction the file is read back; a
    missing, corrupt or out-of-window file falls back to ``replay_start``
    so a stale state from a different CSV cannot silently resume in the
    wrong place.
    """

    def __init__(
        self,
        csv_end: pd.Timestamp,
        lookback_months: int,
        state_path: Optional[Path] = None,
    ):
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
        self._state_path = state_path
        persisted = self._load_persisted() if state_path is not None else None
        self.virt_solve_time = persisted if persisted is not None else self._replay_start
        self._exhausted_logged = False

    def peek(self) -> Optional[pd.Timestamp]:
        """Return the current solve_time, or None if exhausted. No mutation."""
        if self.virt_solve_time >= self._csv_end:
            if not self._exhausted_logged:
                logger.info(
                    "replay window exhausted at %s; idling — heartbeat continues",
                    self._csv_end.isoformat(),
                )
                self._exhausted_logged = True
            return None
        return self.virt_solve_time

    def commit(self) -> None:
        """Advance by one hour and persist.

        No-op once exhausted; defensive guard so a buggy caller can never
        push virt_solve_time past csv_end into the invalid range that
        ``_load_persisted`` rejects on next startup.
        """
        if self.virt_solve_time >= self._csv_end:
            return
        self.virt_solve_time = self.virt_solve_time + pd.Timedelta(hours=1)
        self._persist()

    def _persist(self) -> None:
        if self._state_path is None:
            return
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._state_path.with_name(self._state_path.name + ".tmp")
        payload = json.dumps({"virt_solve_time": self.virt_solve_time.isoformat()})
        tmp.write_text(payload)
        os.replace(tmp, self._state_path)

    def _load_persisted(self) -> Optional[pd.Timestamp]:
        assert self._state_path is not None
        if not self._state_path.exists():
            return None
        try:
            data = json.loads(self._state_path.read_text())
            ts = pd.Timestamp(data["virt_solve_time"])
        except (OSError, json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(
                "could not read persisted replay state %s: %s; restarting from replay_start",
                self._state_path, e,
            )
            return None
        # Strict window check: a state file from a different (longer / shorter)
        # CSV would otherwise resume at a meaningless point. csv_end is the
        # terminal value, so use inclusive upper bound.
        if ts < self._replay_start or ts > self._csv_end:
            logger.warning(
                "persisted virt_solve_time %s outside replay window [%s, %s]; "
                "restarting from replay_start",
                ts.isoformat(), self._replay_start.isoformat(), self._csv_end.isoformat(),
            )
            return None
        logger.info("resuming replay from persisted virt_solve_time %s", ts.isoformat())
        return ts


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
    """Remove leftover ``*.parquet`` and the heartbeat before a fresh run.

    The replay-state file is intentionally NOT cleaned — that file is what
    lets a restart resume from where it left off. Use
    ``scripts/reset_demo.sh`` to wipe everything when you actually want to
    start the demo over from the beginning.
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
    solve_time = state.peek()
    if solve_time is None:
        # Idle: bump heartbeat so the healthcheck stays green.
        _touch_heartbeat(cfg.heartbeat_path)
        return
    cycle_history = history[history.index < solve_time]
    if cycle_history.empty:
        # No usable history for this slot — advance past it. Without the
        # commit the loop would spin forever on the same broken solve_time.
        logger.warning("no history before %s; advancing past it", solve_time.isoformat())
        state.commit()
        return
    logger.info(
        "tick: solve_time=%s history_rows=%d model=%s horizon=%dh",
        solve_time.isoformat(), len(cycle_history), cfg.model_name, cfg.horizon_hours,
    )
    predictions = model.predict(cycle_history, solve_time, horizon=cfg.horizon_hours)
    out_path = export_forecast(predictions, solve_time, output_dir=str(cfg.forecast_dir))
    # Commit only after a parquet is on disk. A crash before this line means
    # the same solve_time is retried on the next tick, not silently skipped.
    state.commit()
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

    # Preflight: fail loud and early on misconfiguration so the container
    # restart-loop carries an obvious message instead of a mid-load traceback.
    if not cfg.csv_path.exists():
        logger.error(
            "CSV not found at %s — check the data/forecasting/raw bind mount",
            cfg.csv_path,
        )
        return 2
    try:
        model = _build_model(cfg.model_name)
    except ValueError as e:
        logger.error("%s", e)
        return 2

    history = _load_history(cfg.csv_path)
    csv_end = history.index.max()
    if pd.isna(csv_end):
        logger.error("CSV has no data after cleaning; aborting")
        return 1

    cfg.forecast_dir.mkdir(parents=True, exist_ok=True)
    # Cleanup deletes leftover parquet + heartbeat but preserves the
    # replay-state file, so the ReplayState constructor below can resume.
    _cleanup_forecast_dir(cfg.forecast_dir)

    state = ReplayState(
        csv_end=csv_end,
        lookback_months=cfg.lookback_months,
        state_path=cfg.replay_state_path,
    )
    _touch_heartbeat(cfg.heartbeat_path)

    # Fire once immediately so a fresh container produces a forecast without
    # waiting a full tick interval. Wrap in try/except so a transient failure
    # at boot doesn't kill the container — the scheduler will retry next tick.
    try:
        _tick(cfg, history, model, state)
    except Exception:
        logger.exception("eager tick failed; scheduler will retry")

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
