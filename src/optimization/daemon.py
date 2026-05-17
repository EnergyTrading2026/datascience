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
gated on three checks — any failing one logs and keeps the previous config:
  1. the new file validates (no errors from validate_plant_payload);
  2. the asset id set is unchanged (add/remove is a separate task, requiring
     state migration);
  3. the currently-stored DispatchState is still feasible under the new
     bounds (e.g. the new floor_mwh_th doesn't strand the SoC outside
     [floor, capacity] — that would make the next solve infeasible).

A no-op reload (mtime bumped via ``touch`` but content identical) advances
the watermark silently — no swap, no log, no warning replay.

RuntimeConfig is intentionally out of scope: it's loaded once at daemon
start and not re-read. Operator changes to horizon hours / safety factor /
solver gap still require a daemon restart.

In-flight solves are unaffected because the worker is single-threaded — the
swap happens between cycles, never inside ``run_one_cycle``.
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
    Issue,
    ValidationResult,
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


def _format_issues(issues: tuple[Issue, ...]) -> str:
    """One ``severity code at path: message`` line per issue.

    Callers gate on ``result.errors`` / ``result.warnings`` already, so the
    empty case is not formatted here.
    """
    return "\n".join(
        f"  {i.severity.upper()} {i.code} at {i.path}: {i.message}" for i in issues
    )


def _log_validation_result(
    config_file: Path, result: ValidationResult, applied: bool
) -> None:
    """Log errors and warnings on separate log records.

    Errors always go to ERROR. Warnings go to WARNING regardless of whether
    the swap was applied — operators care about warnings either way. Each
    record names its severity and count up-front so a log grep for ``ERROR``
    or ``WARNING`` only surfaces the relevant lines.
    """
    if result.errors:
        logger.error(
            "config reload rejected for %s: %d validation error(s)\n%s",
            config_file, len(result.errors), _format_issues(result.errors),
        )
    if result.warnings:
        prefix = "applied with" if applied else "candidate also had"
        logger.warning(
            "config reload %s for %s: %d warning(s)\n%s",
            prefix, config_file, len(result.warnings), _format_issues(result.warnings),
        )


def _log_startup_warnings(config_file: Path, result: ValidationResult) -> None:
    """Surface validation warnings observed during the daemon's initial load.

    Errors abort startup before this is reached; warnings would otherwise be
    silently dropped, hiding policy violations the operator should see on
    first boot. Phrased distinctly from the reload path so a log grep can
    tell first-boot warnings from later live-edit warnings.
    """
    if not result.warnings:
        return
    logger.warning(
        "config loaded from %s with %d warning(s):\n%s",
        config_file, len(result.warnings), _format_issues(result.warnings),
    )


def _log_startup_validation_failure(
    config_file: Path, result: ValidationResult
) -> None:
    """Log every issue from a startup-time ConfigValidationError.

    ``ConfigValidationError.__str__`` only carries the first error message,
    so a plain ``logger.error("%s", e)`` would hide every additional issue.
    The operator wants the full picture in one boot attempt, not a peel-the-
    onion loop. Warnings are surfaced too — they are still relevant context
    even when errors abort the boot.
    """
    if result.errors:
        logger.error(
            "config load rejected for %s: %d validation error(s)\n%s",
            config_file, len(result.errors), _format_issues(result.errors),
        )
    if result.warnings:
        logger.warning(
            "config load for %s also produced %d warning(s):\n%s",
            config_file, len(result.warnings), _format_issues(result.warnings),
        )


def _load_plant_config_with_watermark(
    config_file: Path,
) -> tuple[PlantConfig, Optional[int], ValidationResult]:
    """Load PlantConfig and capture the pre-read mtime watermark.

    Stat happens BEFORE read so that a write landing between the two doesn't
    advance the watermark past content the daemon never observed. If the
    race fires, ``plant_config`` holds the new content but the watermark is
    the older mtime; the next worker stat sees mtime has advanced and
    triggers a (potentially no-op) reload that catches up.

    The returned ValidationResult carries any warnings the payload produced
    so the caller can surface them at startup — errors are raised as
    ``ConfigValidationError`` before this returns.

    Returns ``(cfg, mtime_ns | None, result)``. ``None`` watermark means
    the next worker iteration reloads on its first stat — safe fallback.
    """
    try:
        mtime_ns: Optional[int] = config_file.stat().st_mtime_ns
    except OSError:
        mtime_ns = None
    cfg, result = PlantConfig.from_json_with_result(config_file)
    return cfg, mtime_ns, result


def _maybe_reload_plant_config(
    config_file: Optional[Path],
    current: Optional[PlantConfig],
    reload_state: _ConfigReloadState,
    state_path: Optional[Path] = None,
) -> Optional[PlantConfig]:
    """Re-stat ``config_file`` and swap ``current`` if it changed and validates.

    Returns the PlantConfig the next cycle should use — either the freshly
    loaded one or ``current`` unchanged. The watermark in ``reload_state`` is
    advanced on every observed mtime change, including rejections, so a
    broken file does not re-trigger every cycle.

    Reload is rejected (and ``current`` returned) when:
      - the new file is unreadable (missing, bad JSON);
      - validation fails (full ValidationResult is logged);
      - the asset id set differs from ``current`` (deferred to the live
        add/remove task);
      - the current DispatchState at ``state_path`` is no longer feasible
        under the new config (e.g. SoC outside the new [floor, capacity]).

    If ``state_path`` is None or the state file doesn't exist, the
    feasibility check is skipped (the worker's normal path already requires
    ``current.json`` and will fail loudly later if it is missing).

    A successful reload to an identical config (content unchanged, only
    mtime bumped via ``touch``) is a silent no-op: watermark advances, no
    log, no warning replay.
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
        candidate, result = PlantConfig.from_json_with_result(config_file)
    except ConfigValidationError as e:
        _log_validation_result(config_file, e.result, applied=False)
        return current
    except (ValueError, OSError) as e:
        logger.error(
            "config reload rejected for %s: could not load (%s); keeping current config",
            config_file, e,
        )
        return current

    # No-op: ``touch`` or whitespace-only edits bump mtime but leave content
    # equivalent. PlantConfig is a frozen dataclass so structural equality
    # covers every field that matters. Silent return keeps logs clean.
    if candidate == current:
        return current

    # dt_h is the MILP grid resolution. UnitState.time_in_state_steps is
    # counted in dt_h units, and min_up_steps / min_down_steps are dt_h-step
    # counts too — changing dt_h would silently rescale every commitment
    # constraint against carry-over state from the old grid. Reject it: a
    # grid change is a daemon restart, not a hot reload.
    if candidate.dt_h != current.dt_h:
        logger.error(
            "config reload rejected for %s: dt_h changed (%s → %s). "
            "The MILP grid resolution is not hot-swappable because "
            "DispatchState.time_in_state_steps is measured in dt_h units. "
            "Restart the daemon to pick up a grid change.",
            config_file, current.dt_h, candidate.dt_h,
        )
        if result.warnings:
            _log_validation_result(config_file, result, applied=False)
        return current

    if not current.same_asset_set(candidate):
        logger.error(
            "config reload rejected for %s: asset id set changed — live add/remove "
            "is a separate operation. Keeping current config.",
            config_file,
        )
        # Still surface any warnings the candidate would have had, so the
        # operator sees the full picture before retrying.
        if result.warnings:
            _log_validation_result(config_file, result, applied=False)
        return current

    # State feasibility: a parameter-only change can still strand the stored
    # SoC outside the new [floor, capacity]. Catch it here rather than letting
    # the next MILP solve fail with a less actionable infeasibility.
    if state_path is not None and state_path.exists():
        try:
            state = DispatchState.load(state_path)
        except (ValueError, KeyError, TypeError, OSError) as e:
            # If state is unreadable the daemon has bigger problems; the
            # next _run_one will surface it. Don't block the reload on it.
            logger.warning(
                "config reload: could not read state %s for feasibility check (%s); "
                "applying swap without state check.",
                state_path, e,
            )
        else:
            problems = state.feasible_against(candidate)
            if problems:
                logger.error(
                    "config reload rejected for %s: current state is infeasible "
                    "under the new config:\n%s\n"
                    "Adjust the offending bound or re-seed state before retrying.",
                    config_file,
                    "\n".join(f"  - {p}" for p in problems),
                )
                if result.warnings:
                    _log_validation_result(config_file, result, applied=False)
                return current

    if result.warnings:
        _log_validation_result(config_file, result, applied=True)
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
            # Pass state_path so the reload can revert if the candidate's new
            # SoC bounds would strand the currently-stored state.
            plant_config = _maybe_reload_plant_config(
                cfg.config_file, plant_config, reload_state,
                state_path=cfg.state_dir / CURRENT_NAME,
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
            plant_config, reload_state.last_mtime_ns, load_result = (
                _load_plant_config_with_watermark(cfg.config_file)
            )
        except ConfigValidationError as e:
            # ConfigValidationError is a ValueError subclass — catch it first
            # so the full ValidationResult is logged, not just str(e) which
            # only carries the first error message.
            _log_startup_validation_failure(cfg.config_file, e.result)
            return 1
        except (OSError, ValueError) as e:
            logger.error("plant config load failed (%s): %s", cfg.config_file, e)
            return 1
        _log_startup_warnings(cfg.config_file, load_result)
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
