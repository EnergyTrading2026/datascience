"""Long-running daemon: runs one MPC cycle per new forecast file.

Architecture:
    1. A periodic scanner (every SCAN_INTERVAL_S seconds) checks FORECAST_DIR
       for the newest ``<solve_time>.parquet`` file. If it is newer than the
       last enqueued/processed solve_time, it enqueues a cycle for it.
    2. A single worker thread drains the queue and runs cycles sequentially.
       Cycles never overlap.

Polling-only (no inotify) is intentional: identical behavior on Linux and
macOS bind-mounts, no platform-specific event semantics, one mechanism to
reason about. At a few-second interval the cost is negligible and pickup
latency is well below the hourly cadence.

After each successful cycle:
    - State JSON is written to ``state/<solve_time>.json``.
    - ``state/current.json`` symlink is atomically retargeted to it.
    - ``state/.heartbeat`` mtime is bumped (used by the container healthcheck).

Two filters decide whether a forecast file is processed:

1. **Monotonicity.** ``solve_time <= last successful solve_time`` → dropped.
   This is the primary guard and is what lets the daemon serve both the
   CSV-backed replay forecaster (solve_times months in the past) and a future
   live forecaster (solve_times near now) with zero code change.

2. **Future-skew cap.** ``solve_time > now + MAX_FUTURE_SKEW_HOURS`` → dropped,
   at *both* the scan layer (so a bogus far-future file cannot win ``max()``
   and poison the scanner watermark) and the worker layer (defense-in-depth).
   A single off-by-year filename would otherwise advance ``last_enqueued``
   to that timestamp and every real subsequent forecast would be filtered
   out as "already seen".

3. **Stale-floor (opt-in).** ``solve_time < now - STALE_GRACE_HOURS`` → dropped,
   *only* when ``STALE_GRACE_HOURS`` env is set to a positive integer.
   Default 0 = disabled, which is required for the CSV replay producer
   (it legitimately writes months-old solve_times). At live-feed cutover
   set ``STALE_GRACE_HOURS=2`` (or similar) so a wedged upstream producer
   feeding hours-old forecasts is rejected instead of silently driving
   the optimizer off real-time prices.
"""
from __future__ import annotations

import logging
import os
import queue
import re
import signal
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler

from optimization.config import PlantConfig, RuntimeConfig
from optimization.run import run_one_cycle
from optimization.state import DispatchState

logger = logging.getLogger("optimization.daemon")

FORECAST_FILENAME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z)\.parquet$")
HEARTBEAT_NAME = ".heartbeat"
CURRENT_NAME = "current.json"
# Cap on how far in the future a forecast's solve_time may sit relative to
# wall-clock now. Guards against off-by-year filenames or a clock-skewed
# upstream writer poisoning the scanner watermark.
MAX_FUTURE_SKEW_HOURS = 1
# Opt-in stale floor: when > 0, reject forecasts whose solve_time is older
# than ``now - STALE_GRACE_HOURS``. Default 0 (disabled) because the CSV
# replay producer legitimately writes months-old solve_times. Set to a
# positive int (e.g. 2) at live-feed cutover to reject forecasts from a
# wedged upstream producer. Parsed at import so a typo crashes the daemon
# at startup rather than silently disabling the guard mid-flight.
def _stale_grace_hours_from_env() -> int:
    raw = os.environ.get("STALE_GRACE_HOURS", "0")
    try:
        value = int(raw)
    except ValueError as e:
        raise ValueError(
            f"STALE_GRACE_HOURS must be an integer; got {raw!r}"
        ) from e
    if value < 0:
        raise ValueError(f"STALE_GRACE_HOURS must be >= 0; got {value}")
    return value


STALE_GRACE_HOURS = _stale_grace_hours_from_env()
SHUTDOWN_SENTINEL = object()
DEFAULT_COMMIT_HOURS = RuntimeConfig().commit_hours


@dataclass
class DaemonConfig:
    forecast_dir: Path
    state_dir: Path
    dispatch_dir: Path
    prices_source: str
    prices_path: Optional[Path]
    resolution: str
    forecast_resolution: str
    scan_interval_s: int
    # Path to plant_config.json, or None to use PlantConfig.legacy_default().
    # The actual PlantConfig is loaded once at daemon start (run()) and passed
    # to every cycle; editing this file mid-run has no effect until restart.
    config_file: Optional[Path] = None

    @classmethod
    def from_env(cls) -> "DaemonConfig":
        prices_path_str = os.environ.get("PRICES_PATH")
        config_file_str = os.environ.get("CONFIG_FILE")
        return cls(
            forecast_dir=Path(os.environ.get("FORECAST_DIR", "/shared/forecast")),
            state_dir=Path(os.environ.get("STATE_DIR", "/shared/state")),
            dispatch_dir=Path(os.environ.get("DISPATCH_DIR", "/shared/dispatch")),
            prices_source=os.environ.get("PRICES_SOURCE", "live"),
            prices_path=Path(prices_path_str) if prices_path_str else None,
            resolution=os.environ.get("RESOLUTION", "quarterhour"),
            forecast_resolution=os.environ.get("FORECAST_RESOLUTION", "hour"),
            scan_interval_s=int(os.environ.get("SCAN_INTERVAL_S", "2")),
            config_file=Path(config_file_str) if config_file_str else None,
        )


@dataclass
class _ScanState:
    """Tracks the newest solve_time the scanner has put on the queue.

    Prevents the scanner from re-enqueueing the same file every tick while a
    cycle is still in flight (the worker only advances last_solve_time on
    successful commit, which can be tens of seconds after enqueue).
    """
    last_enqueued: Optional[pd.Timestamp] = None
    # Throttle for "all forecasts blocked by monotonicity" warning so we don't
    # spam every SCAN_INTERVAL_S. Holds the last blocked solve_time we warned
    # about; reset to None when the scanner successfully enqueues again.
    last_blocked_warning_for: Optional[pd.Timestamp] = None


def _parse_solve_time(filename: str) -> Optional[pd.Timestamp]:
    m = FORECAST_FILENAME_RE.match(filename)
    if not m:
        return None
    # Filenames use hyphens in the time portion (e.g. 2026-05-13T14-00-00Z)
    # because Windows rejects ':' in paths. Restore the ISO form for pandas.
    date_part, time_part = m.group(1).split("T", 1)
    iso = f"{date_part}T{time_part[:2]}:{time_part[3:5]}:{time_part[6:8]}Z"
    return pd.Timestamp(iso)


def _scan_newest_forecast(
    forecast_dir: Path,
    max_solve_time: Optional[pd.Timestamp] = None,
) -> Optional[pd.Timestamp]:
    """Newest parseable forecast filename, optionally capped at ``max_solve_time``.

    The cap exists so a single implausibly-future filename (e.g. clock-skewed
    upstream writer) cannot mask all legitimate forecasts in the same directory.
    """
    if not forecast_dir.is_dir():
        return None
    candidates: list[pd.Timestamp] = []
    for entry in forecast_dir.iterdir():
        if not entry.is_file():
            continue
        ts = _parse_solve_time(entry.name)
        if ts is None:
            continue
        if max_solve_time is not None and ts > max_solve_time:
            continue
        candidates.append(ts)
    return max(candidates) if candidates else None


def _atomic_symlink(target_name: str, link_path: Path) -> None:
    """Point ``link_path`` at ``target_name`` (a sibling filename).

    Implemented as ``symlink → tmp_path → os.replace(link_path)`` so existing
    readers never see a missing or half-written link.
    """
    tmp = link_path.with_suffix(link_path.suffix + ".tmp")
    try:
        if tmp.is_symlink() or tmp.exists():
            tmp.unlink()
    except FileNotFoundError:
        pass
    tmp.symlink_to(target_name)
    os.replace(tmp, link_path)


def _bump_heartbeat(state_dir: Path) -> None:
    hb = state_dir / HEARTBEAT_NAME
    hb.touch(exist_ok=True)
    now = pd.Timestamp.now(tz="UTC").timestamp()
    os.utime(hb, (now, now))


def _read_last_solve_time(
    state_dir: Path,
    commit_hours: int = DEFAULT_COMMIT_HOURS,
) -> Optional[pd.Timestamp]:
    """Last solve_time that completed successfully, or None if no state yet.

    ``current.json`` stores the realized state timestamp at commit_end, not the
    forecast's original solve_time. Convert that back so the next hour's
    forecast is not mistaken for a duplicate.
    """
    current = state_dir / CURRENT_NAME
    if not current.exists():
        return None
    try:
        return DispatchState.load(current).timestamp - pd.Timedelta(hours=commit_hours)
    except (FileNotFoundError, ValueError, KeyError, TypeError) as e:
        logger.warning("state at %s is unreadable; treating as missing (%s)", current, e)
        return None


def _run_one(
    cfg: DaemonConfig,
    solve_time: pd.Timestamp,
    plant_config: Optional[PlantConfig] = None,
) -> int:
    """Execute one cycle for the given solve_time. Returns the exit code from run.py.

    ``plant_config`` is the pre-loaded PlantConfig (from ``cfg.config_file`` or
    None for legacy_default). Threaded through to ``run_one_cycle`` so the
    daemon honors the operator's plant configuration instead of silently using
    the legacy default.
    """
    fname = f"{solve_time.strftime('%Y-%m-%dT%H-%M-%SZ')}.parquet"
    state_fname = f"{solve_time.strftime('%Y-%m-%dT%H-%M-%SZ')}.json"
    forecast_path = cfg.forecast_dir / fname
    dispatch_path = cfg.dispatch_dir / fname
    dispatch_tmp = dispatch_path.with_suffix(dispatch_path.suffix + ".tmp")
    state_current = cfg.state_dir / CURRENT_NAME
    state_dated = cfg.state_dir / state_fname

    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    cfg.dispatch_dir.mkdir(parents=True, exist_ok=True)

    if not state_current.exists():
        logger.error(
            "no state at %s; init_state must seed it before the daemon starts",
            state_current,
        )
        return 1

    try:
        rc = run_one_cycle(
            solve_time=solve_time,
            forecast_path=forecast_path,
            prices_path=cfg.prices_path,
            state_in=state_current,
            state_out=state_dated,
            dispatch_out=dispatch_tmp,
            cold_start=False,
            config=plant_config,
            prices_source=cfg.prices_source,
            resolution=cfg.resolution,
            forecast_resolution=cfg.forecast_resolution,
        )
    except Exception:
        for path in (dispatch_tmp, state_dated):
            try:
                if path.exists() or path.is_symlink():
                    path.unlink()
            except OSError:
                pass
        raise
    if rc == 0:
        os.replace(dispatch_tmp, dispatch_path)
        _atomic_symlink(state_fname, state_current)
        _bump_heartbeat(cfg.state_dir)
        logger.info("cycle ok: current.json -> %s", state_fname)
    else:
        # Drop the dated state file — we don't want to keep an artifact whose
        # symlink was never advanced. If rc != 0 run_one_cycle may not have
        # written it anyway, but cover both cases.
        for path in (dispatch_tmp, state_dated):
            try:
                if path.exists() or path.is_symlink():
                    path.unlink()
            except OSError:
                pass
        logger.warning("cycle failed: rc=%d, state not advanced", rc)
    return rc


def _should_run(
    cfg: DaemonConfig,
    solve_time: pd.Timestamp,
    last_solve_time: Optional[pd.Timestamp],
) -> bool:
    now = pd.Timestamp.now(tz="UTC")
    if solve_time > now.ceil("h") + pd.Timedelta(hours=MAX_FUTURE_SKEW_HOURS):
        logger.warning(
            "skip implausibly future forecast %s (now=%s, max_future_skew=%dh)",
            solve_time.isoformat(), now.isoformat(), MAX_FUTURE_SKEW_HOURS,
        )
        return False
    if (
        STALE_GRACE_HOURS > 0
        and solve_time < now.floor("h") - pd.Timedelta(hours=STALE_GRACE_HOURS)
    ):
        logger.info(
            "skip stale forecast %s (now=%s, grace=%dh)",
            solve_time.isoformat(), now.isoformat(), STALE_GRACE_HOURS,
        )
        return False
    if last_solve_time is not None and solve_time <= last_solve_time:
        logger.info(
            "skip already-processed forecast %s (last_solve_time=%s)",
            solve_time.isoformat(), last_solve_time.isoformat(),
        )
        return False
    return True


def _worker(
    cfg: DaemonConfig,
    work_queue: "queue.Queue[object]",
    stop_event: threading.Event,
    plant_config: Optional[PlantConfig] = None,
) -> None:
    while not stop_event.is_set():
        item = work_queue.get()
        try:
            if item is SHUTDOWN_SENTINEL:
                return
            solve_time: pd.Timestamp = item  # type: ignore[assignment]
            last_solve_time = _read_last_solve_time(cfg.state_dir)
            if not _should_run(cfg, solve_time, last_solve_time):
                continue
            try:
                _run_one(cfg, solve_time, plant_config=plant_config)
            except Exception:
                logger.exception("cycle crashed for solve_time=%s", solve_time.isoformat())
        finally:
            work_queue.task_done()


def _scan(
    cfg: DaemonConfig,
    work_queue: "queue.Queue[object]",
    scan_state: _ScanState,
) -> None:
    """Scan forecast_dir for new files and enqueue the newest unseen one.

    Idempotent on quiet ticks: returns silently if nothing has changed since
    the last enqueue. Logs only when a new file is actually picked up.
    """
    # Cap the candidate set so a single bogus far-future filename cannot win
    # ``max()`` against every legitimate forecast — advancing the watermark
    # to that timestamp would otherwise filter out everything real that
    # follows. The worker re-validates via _should_run (defense-in-depth).
    now = pd.Timestamp.now(tz="UTC")
    max_ts = now.ceil("h") + pd.Timedelta(hours=MAX_FUTURE_SKEW_HOURS)
    newest = _scan_newest_forecast(cfg.forecast_dir, max_solve_time=max_ts)
    if newest is None:
        return
    last_solve_time = _read_last_solve_time(cfg.state_dir)
    if last_solve_time is not None and newest <= last_solve_time:
        # Diagnostic: a forecast file exists but is older than what state
        # already records. Most likely cause: the forecast producer was reset
        # (e.g. the replay loop restarted from csv_end - 3mo) without also
        # wiping data/state/. Warn once per distinct blocked solve_time so the
        # operator sees it in the logs without spam.
        if scan_state.last_blocked_warning_for != newest:
            logger.warning(
                "newest forecast %s is not newer than processed state %s — "
                "daemon is idle. If you reset the forecaster, also reset state "
                "(see scripts/reset_demo.sh).",
                newest.isoformat(), last_solve_time.isoformat(),
            )
            scan_state.last_blocked_warning_for = newest
        return
    if scan_state.last_enqueued is not None and newest <= scan_state.last_enqueued:
        return
    logger.info("scan: enqueueing forecast %s", newest.isoformat())
    work_queue.put(newest)
    scan_state.last_enqueued = newest
    scan_state.last_blocked_warning_for = None


def run(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # APScheduler logs every fired job at INFO; at a 2s scan interval that's
    # one line per second forever. Promote to WARNING so only real problems
    # surface — the daemon's own logger still reports each enqueue.
    logging.getLogger("apscheduler").setLevel(logging.WARNING)
    cfg = DaemonConfig.from_env()
    cfg.forecast_dir.mkdir(parents=True, exist_ok=True)
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    cfg.dispatch_dir.mkdir(parents=True, exist_ok=True)
    if (cfg.state_dir / CURRENT_NAME).exists() and not (cfg.state_dir / HEARTBEAT_NAME).exists():
        if _read_last_solve_time(cfg.state_dir) is not None:
            _bump_heartbeat(cfg.state_dir)
            logger.info("seeded missing heartbeat from existing state")
        else:
            logger.warning(
                "state exists at %s but is unreadable; refusing to seed heartbeat",
                cfg.state_dir / CURRENT_NAME,
            )

    plant_config: Optional[PlantConfig] = None
    if cfg.config_file is not None:
        try:
            plant_config = PlantConfig.from_json(cfg.config_file)
        except (FileNotFoundError, ValueError) as e:
            logger.error("plant config load failed (%s): %s", cfg.config_file, e)
            return 1
        logger.info(
            "loaded plant config from %s (%d HPs, %d boilers, %d CHPs, %d storages)",
            cfg.config_file, len(plant_config.heat_pumps), len(plant_config.boilers),
            len(plant_config.chps), len(plant_config.storages),
        )

    logger.info(
        "daemon starting: forecast=%s state=%s dispatch=%s prices=%s resolution=%s/%s scan=%ds config=%s",
        cfg.forecast_dir, cfg.state_dir, cfg.dispatch_dir,
        cfg.prices_source, cfg.resolution, cfg.forecast_resolution,
        cfg.scan_interval_s,
        cfg.config_file if cfg.config_file is not None else "<legacy_default>",
    )

    work_queue: "queue.Queue[object]" = queue.Queue()
    stop_event = threading.Event()
    scan_state = _ScanState()

    worker_thread = threading.Thread(
        target=_worker,
        args=(cfg, work_queue, stop_event, plant_config),
        name="mpc-worker",
        daemon=True,
    )
    worker_thread.start()

    scheduler = BackgroundScheduler(timezone="UTC")
    # Omit next_run_time: APScheduler defaults to now + interval. Passing None
    # would create the job in paused state, which would silently never fire.
    scheduler.add_job(
        _scan, "interval", args=[cfg, work_queue, scan_state],
        seconds=cfg.scan_interval_s, id="scan",
    )
    scheduler.start()

    # Run one scan immediately so a forecast already on disk at startup is
    # picked up without waiting a full interval.
    _scan(cfg, work_queue, scan_state)

    def _shutdown(signum: int, _frame) -> None:
        logger.info("received signal %d, shutting down", signum)
        stop_event.set()
        work_queue.put(SHUTDOWN_SENTINEL)
        try:
            scheduler.shutdown(wait=False)
        except Exception:
            pass

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    worker_thread.join()
    logger.info("daemon stopped")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(run())
    except Exception:
        logger.exception("daemon crashed")
        sys.exit(3)
