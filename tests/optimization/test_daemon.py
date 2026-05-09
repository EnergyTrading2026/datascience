"""Daemon and init-state regressions."""
from __future__ import annotations

import json
import os
import queue

import pandas as pd

from optimization import daemon, init_state
from optimization.state import DispatchState


def _make_cfg(tmp_path):
    return daemon.DaemonConfig(
        forecast_dir=tmp_path / "forecast",
        state_dir=tmp_path / "state",
        dispatch_dir=tmp_path / "dispatch",
        prices_source="live",
        prices_path=None,
        resolution="quarterhour",
        forecast_resolution="hour",
        scan_interval_s=2,
    )


def test_read_last_solve_time_undoes_commit_window(tmp_path):
    state_dir = tmp_path / "state"
    current = state_dir / "current.json"
    DispatchState.cold_start(pd.Timestamp("2026-05-07T14:00:00Z")).save(current)

    last_solve_time = daemon._read_last_solve_time(state_dir, commit_hours=1)

    assert last_solve_time == pd.Timestamp("2026-05-07T13:00:00Z")


def test_scan_enqueues_next_hour_after_previous_success(tmp_path):
    forecast_dir = tmp_path / "forecast"
    state_dir = tmp_path / "state"
    dispatch_dir = tmp_path / "dispatch"
    forecast_dir.mkdir()
    state_dir.mkdir()
    dispatch_dir.mkdir()

    last_solve_time = pd.Timestamp.now(tz="UTC").floor("h")
    next_solve_time = last_solve_time + pd.Timedelta(hours=1)
    DispatchState.cold_start(next_solve_time).save(state_dir / "current.json")
    (forecast_dir / f"{next_solve_time.strftime('%Y-%m-%dT%H:%M:%SZ')}.parquet").write_text("")

    cfg = daemon.DaemonConfig(
        forecast_dir=forecast_dir,
        state_dir=state_dir,
        dispatch_dir=dispatch_dir,
        prices_source="live",
        prices_path=None,
        resolution="quarterhour",
        forecast_resolution="hour",
        scan_interval_s=2,
    )
    work_queue: "queue.Queue[object]" = queue.Queue()
    scan_state = daemon._ScanState()

    daemon._scan(cfg, work_queue, scan_state)

    assert work_queue.get_nowait() == next_solve_time
    assert scan_state.last_enqueued == next_solve_time


def test_scan_does_not_re_enqueue_same_solve_time(tmp_path):
    forecast_dir = tmp_path / "forecast"
    state_dir = tmp_path / "state"
    dispatch_dir = tmp_path / "dispatch"
    forecast_dir.mkdir()
    state_dir.mkdir()
    dispatch_dir.mkdir()

    solve_time = pd.Timestamp.now(tz="UTC").floor("h")
    # No state on disk yet — _read_last_solve_time returns None — so dedupe
    # has to rely on _ScanState.last_enqueued. This mirrors the in-flight
    # case: cycle running, last_solve_time not yet advanced.
    (forecast_dir / f"{solve_time.strftime('%Y-%m-%dT%H:%M:%SZ')}.parquet").write_text("")

    cfg = daemon.DaemonConfig(
        forecast_dir=forecast_dir,
        state_dir=state_dir,
        dispatch_dir=dispatch_dir,
        prices_source="live",
        prices_path=None,
        resolution="quarterhour",
        forecast_resolution="hour",
        scan_interval_s=2,
    )
    work_queue: "queue.Queue[object]" = queue.Queue()
    scan_state = daemon._ScanState()

    daemon._scan(cfg, work_queue, scan_state)
    daemon._scan(cfg, work_queue, scan_state)
    daemon._scan(cfg, work_queue, scan_state)

    assert work_queue.qsize() == 1
    assert work_queue.get_nowait() == solve_time


def test_should_run_rejects_implausibly_future_forecast(tmp_path):
    cfg = _make_cfg(tmp_path)
    solve_time = pd.Timestamp.now(tz="UTC").ceil("h") + pd.Timedelta(
        hours=daemon.MAX_FUTURE_SKEW_HOURS + 2
    )

    assert daemon._should_run(cfg, solve_time, last_solve_time=None) is False


def test_read_last_solve_time_treats_partial_state_as_missing(tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "current.json").write_text(json.dumps({"timestamp": "2026-05-07T13:00:00Z"}))

    assert daemon._read_last_solve_time(state_dir) is None


def test_run_one_stages_dispatch_until_state_commit(tmp_path, monkeypatch):
    forecast_dir = tmp_path / "forecast"
    state_dir = tmp_path / "state"
    dispatch_dir = tmp_path / "dispatch"
    forecast_dir.mkdir()
    state_dir.mkdir()
    dispatch_dir.mkdir()

    solve_time = pd.Timestamp("2026-05-07T13:00:00Z")
    current = state_dir / "current.json"
    initial_state = DispatchState.cold_start(solve_time)
    initial_state.save(current)

    seen: dict[str, object] = {}

    def fake_run_one_cycle(**kwargs):
        dispatch_out = kwargs["dispatch_out"]
        state_out = kwargs["state_out"]
        seen["dispatch_out"] = dispatch_out
        seen["state_out"] = state_out
        dispatch_out.write_text("staged dispatch")
        DispatchState.cold_start(solve_time + pd.Timedelta(hours=1)).save(state_out)
        return 0

    monkeypatch.setattr(daemon, "run_one_cycle", fake_run_one_cycle)

    cfg = daemon.DaemonConfig(
        forecast_dir=forecast_dir,
        state_dir=state_dir,
        dispatch_dir=dispatch_dir,
        prices_source="live",
        prices_path=None,
        resolution="quarterhour",
        forecast_resolution="hour",
        scan_interval_s=2,
    )

    rc = daemon._run_one(cfg, solve_time)

    final_dispatch = dispatch_dir / "2026-05-07T13:00:00Z.parquet"
    staged_dispatch = final_dispatch.with_suffix(".parquet.tmp")
    assert rc == 0
    assert seen["dispatch_out"] == staged_dispatch
    assert final_dispatch.exists()
    assert not staged_dispatch.exists()
    assert current.is_symlink()
    assert current.resolve().name == "2026-05-07T13:00:00Z.json"


def test_run_one_publishes_dispatch_before_advancing_state(tmp_path, monkeypatch):
    forecast_dir = tmp_path / "forecast"
    state_dir = tmp_path / "state"
    dispatch_dir = tmp_path / "dispatch"
    forecast_dir.mkdir()
    state_dir.mkdir()
    dispatch_dir.mkdir()

    solve_time = pd.Timestamp("2026-05-07T13:00:00Z")
    current = state_dir / "current.json"
    DispatchState.cold_start(solve_time).save(current)

    def fake_run_one_cycle(**kwargs):
        kwargs["dispatch_out"].write_text("staged dispatch")
        DispatchState.cold_start(solve_time + pd.Timedelta(hours=1)).save(kwargs["state_out"])
        return 0

    replace_calls: list[tuple[os.PathLike[str] | str, os.PathLike[str] | str]] = []
    real_replace = os.replace

    def recording_replace(src, dst):
        replace_calls.append((src, dst))
        real_replace(src, dst)

    monkeypatch.setattr(daemon, "run_one_cycle", fake_run_one_cycle)
    monkeypatch.setattr(daemon.os, "replace", recording_replace)

    cfg = daemon.DaemonConfig(
        forecast_dir=forecast_dir,
        state_dir=state_dir,
        dispatch_dir=dispatch_dir,
        prices_source="live",
        prices_path=None,
        resolution="quarterhour",
        forecast_resolution="hour",
        scan_interval_s=2,
    )

    rc = daemon._run_one(cfg, solve_time)

    assert rc == 0
    dispatch_path = dispatch_dir / "2026-05-07T13:00:00Z.parquet"
    current_tmp = current.with_suffix(current.suffix + ".tmp")
    assert replace_calls[-2:] == [
        (dispatch_path.with_suffix(".parquet.tmp"), dispatch_path),
        (current_tmp, current),
    ]


def test_run_one_cleans_up_staged_outputs_after_crash(tmp_path, monkeypatch):
    forecast_dir = tmp_path / "forecast"
    state_dir = tmp_path / "state"
    dispatch_dir = tmp_path / "dispatch"
    forecast_dir.mkdir()
    state_dir.mkdir()
    dispatch_dir.mkdir()

    solve_time = pd.Timestamp("2026-05-07T13:00:00Z")
    current = state_dir / "current.json"
    original_state = DispatchState.cold_start(solve_time)
    original_state.save(current)

    def fake_run_one_cycle(**kwargs):
        kwargs["dispatch_out"].write_text("staged dispatch")
        DispatchState.cold_start(solve_time + pd.Timedelta(hours=1)).save(kwargs["state_out"])
        raise RuntimeError("boom after persistence")

    monkeypatch.setattr(daemon, "run_one_cycle", fake_run_one_cycle)

    cfg = daemon.DaemonConfig(
        forecast_dir=forecast_dir,
        state_dir=state_dir,
        dispatch_dir=dispatch_dir,
        prices_source="live",
        prices_path=None,
        resolution="quarterhour",
        forecast_resolution="hour",
        scan_interval_s=2,
    )

    try:
        daemon._run_one(cfg, solve_time)
    except RuntimeError as e:
        assert str(e) == "boom after persistence"
    else:
        raise AssertionError("expected _run_one to re-raise the crash")

    assert not (dispatch_dir / "2026-05-07T13:00:00Z.parquet").exists()
    assert not (dispatch_dir / "2026-05-07T13:00:00Z.parquet.tmp").exists()
    assert not (state_dir / "2026-05-07T13:00:00Z.json").exists()
    assert DispatchState.load(current).timestamp == solve_time


def test_init_state_writes_symlink_and_heartbeat(tmp_path):
    state_out = tmp_path / "state" / "current.json"
    ts = "2026-05-07T13:00:00+00:00"

    rc = init_state.main(["--state-out", str(state_out), "--solve-time", ts])

    assert rc == 0
    assert state_out.is_symlink()
    assert state_out.resolve().name == "2026-05-07T13:00:00Z.json"
    assert DispatchState.load(state_out).timestamp == pd.Timestamp(ts)
    assert (state_out.parent / ".heartbeat").exists()


def test_init_state_backfills_missing_heartbeat_for_existing_state(tmp_path):
    state_out = tmp_path / "state" / "current.json"
    ts = pd.Timestamp("2026-05-07T13:00:00Z")
    DispatchState.cold_start(ts).save(state_out)

    rc = init_state.main(["--state-out", str(state_out)])

    assert rc == 0
    assert DispatchState.load(state_out).timestamp == ts
    assert (state_out.parent / ".heartbeat").exists()


def test_runtime_entrypoints_import():
    assert callable(daemon.run)
    assert callable(init_state.main)
