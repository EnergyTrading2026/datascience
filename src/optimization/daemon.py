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

Stale or duplicate forecasts are ignored: anything older than
STALE_GRACE_HOURS or with solve_time <= last successful solve_time is dropped.

Live config reload: when CONFIG_FILE is set, the worker checks the file's
mtime at every cycle boundary. If it changed, the file is re-validated and
the in-memory PlantConfig is swapped before the next solve. The swap is
gated on the asset id set being unchanged — add/remove requires the state
migration path that lives in a follow-up task. Validation errors or asset-id
diffs are logged and the daemon keeps running on the previously-loaded
config; in-flight solves are unaffected because the worker is single-threaded.
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
from optimization.config_validation import (
    ConfigValidationError,
    ValidationResult,
    validate_plant_config,
)
from optimization.run import run_one_cycle
from optimization.state import DispatchState

logger = logging.getLogger("optimization.daemon")

FORECAST_FILENAME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z)\.parquet$")
HEARTBEAT_NAME = ".heartbeat"
CURRENT_NAME = "current.json"
STALE_GRACE_HOURS = 2  # ignore forecasts whose solve_time is older than this
MAX_FUTURE_SKEW_HOURS = 1  # allow the next scheduled hour, reject obviously bad filenames
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
    # When set, the worker re-stats this file at every cycle boundary and
    # swaps the in-memory PlantConfig if mtime changed and the new file
    # validates with the same asset id set. legacy_default mode has no file
    # and so no reload.
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
class _ConfigReloadState:
    """Tracks the mtime watermark for the in-memory PlantConfig.

    Seeded at daemon start from the initial load so the first cycle does not
    spuriously "detect" a change. Advanced unconditionally on every observed
    mtime bump — even when the candidate config is rejected — so a broken
    file does not get re-tried (and re-logged) every cycle until the operator
    saves it again.
    """
    last_mtime_ns: Optional[int] = None


def _format_validation_issues(result: ValidationResult) -> str:
    lines = []
    for issue in result.errors:
        lines.append(f"  ERROR {issue.code} at {issue.path}: {issue.message}")
    for issue in result.warnings:
        lines.append(f"  WARNING {issue.code} at {issue.path}: {issue.message}")
    return "\n".join(lines) if lines else "  (no issues)"


def _maybe_reload_plant_config(
    config_file: Optional[Path],
    current: Optional[PlantConfig],
    reload_state: _ConfigReloadState,
) -> Optional[PlantConfig]:
    """Re-stat ``config_file`` and swap ``current`` if it changed and validates.

    Returns the PlantConfig the next cycle should use — either the freshly
    loaded one or ``current`` unchanged. The watermark in ``reload_state`` is
    advanced on every observed mtime change, including rejections, so a
    broken file does not re-trigger every cycle.

    Reload is rejected (and ``current`` returned) when:
      - the new file fails ConfigValidationError (full result is logged);
      - the new file is unreadable (missing, bad JSON);
      - the asset id set differs from ``current`` (deferred to the live
        add/remove task).
    """
    if config_file is None or current is None:
        return current
    try:
        mtime_ns = config_file.stat().st_mtime_ns
    except OSError as e:
        logger.warning("config file %s not statable (%s); keeping current config", config_file, e)
        return current
    if reload_state.last_mtime_ns is not None and mtime_ns == reload_state.last_mtime_ns:
        return current
    reload_state.last_mtime_ns = mtime_ns
    try:
        candidate = PlantConfig.from_json(config_file)
    except ConfigValidationError as e:
        logger.error(
            "config reload rejected for %s: validation failed\n%s",
            config_file, _format_validation_issues(e.result),
        )
        return current
    except (ValueError, OSError) as e:
        logger.error(
            "config reload rejected for %s: could not load (%s); keeping current config",
            config_file, e,
        )
        return current
    if not current.same_asset_set(candidate):
        logger.error(
            "config reload rejected for %s: asset id set changed — live add/remove "
            "is a separate operation. Keeping current config.",
            config_file,
        )
        return current
    # Surface warnings even though the swap is going through. validate_plant_config
    # was already run inside from_json, but the result was discarded; rerun (cheap)
    # to capture warnings for logging.
    warn_result = validate_plant_config(candidate)
    if warn_result.warnings:
        logger.warning(
            "config reload applied with warnings:\n%s",
            _format_validation_issues(
                ValidationResult(errors=(), warnings=warn_result.warnings)
            ),
        )
    logger.info(
        "config reloaded from %s (parameter-only change; asset set unchanged)",
        config_file,
    )
    return candidate


@dataclass
class _ScanState:
    """Tracks the newest solve_time the scanner has put on the queue.

    Prevents the scanner from re-enqueueing the same file every tick while a
    cycle is still in flight (the worker only advances last_solve_time on
    successful commit, which can be tens of seconds after enqueue).
    """
    last_enqueued: Optional[pd.Timestamp] = None


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
    if solve_time < now.floor("h") - pd.Timedelta(hours=STALE_GRACE_HOURS):
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
    reload_state: Optional[_ConfigReloadState] = None,
) -> None:
    if reload_state is None:
        reload_state = _ConfigReloadState()
    while not stop_event.is_set():
        item = work_queue.get()
        try:
            if item is SHUTDOWN_SENTINEL:
                return
            solve_time: pd.Timestamp = item  # type: ignore[assignment]
            last_solve_time = _read_last_solve_time(cfg.state_dir)
            if not _should_run(cfg, solve_time, last_solve_time):
                continue
            # Cycle boundary: re-stat the config file and swap if a valid
            # parameter-only change is on disk. Skipped cycles above do not
            # observe the new config — they wouldn't have used it anyway.
            plant_config = _maybe_reload_plant_config(
                cfg.config_file, plant_config, reload_state,
            )
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
    # Cap the candidate set to plausibly-current filenames. A single
    # bogus far-future file otherwise wins ``max()`` against every legitimate
    # forecast in the same directory — and advancing the watermark to that
    # timestamp would then filter out everything real that follows.
    now = pd.Timestamp.now(tz="UTC")
    max_ts = now.ceil("h") + pd.Timedelta(hours=MAX_FUTURE_SKEW_HOURS)
    newest = _scan_newest_forecast(cfg.forecast_dir, max_solve_time=max_ts)
    if newest is None:
        return
    last_solve_time = _read_last_solve_time(cfg.state_dir)
    if last_solve_time is not None and newest <= last_solve_time:
        return
    if scan_state.last_enqueued is not None and newest <= scan_state.last_enqueued:
        return
    logger.info("scan: enqueueing forecast %s", newest.isoformat())
    work_queue.put(newest)
    scan_state.last_enqueued = newest


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
    reload_state = _ConfigReloadState()
    if cfg.config_file is not None:
        try:
            plant_config = PlantConfig.from_json(cfg.config_file)
        except (FileNotFoundError, ValueError) as e:
            logger.error("plant config load failed (%s): %s", cfg.config_file, e)
            return 1
        try:
            reload_state.last_mtime_ns = cfg.config_file.stat().st_mtime_ns
        except OSError:
            # Should not happen — from_json just succeeded — but tolerate it so
            # a transient stat failure does not abort startup. A None watermark
            # means the next worker iteration will reload on first stat.
            pass
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
        args=(cfg, work_queue, stop_event, plant_config, reload_state),
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
