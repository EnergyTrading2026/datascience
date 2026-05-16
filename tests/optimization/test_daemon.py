"""Daemon and init-state regressions."""
from __future__ import annotations

import json
import os
import queue
from dataclasses import replace

import pandas as pd

from optimization import daemon, init_state
from optimization.config import (
    BoilerParams,
    CHPParams,
    HeatPumpParams,
    PlantConfig,
    StorageParams,
)
from optimization.state import DispatchState

_LEGACY = PlantConfig.legacy_default()


def _cs(ts):
    return DispatchState.cold_start(_LEGACY, ts)


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
    _cs(pd.Timestamp("2026-05-07T14:00:00Z")).save(current)

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
    _cs(next_solve_time).save(state_dir / "current.json")
    (forecast_dir / f"{next_solve_time.strftime('%Y-%m-%dT%H-%M-%SZ')}.parquet").write_text("")

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
    (forecast_dir / f"{solve_time.strftime('%Y-%m-%dT%H-%M-%SZ')}.parquet").write_text("")

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


def test_scan_does_not_advance_watermark_for_future_forecast(tmp_path):
    """A bogus far-future forecast must not poison the scanner watermark.

    Regression: previously _scan advanced ``last_enqueued`` before the worker
    validated timestamps. A 2099-dated file became the eternal "newest", and
    every legitimate forecast afterwards was filtered out as "already seen".
    """
    forecast_dir = tmp_path / "forecast"
    state_dir = tmp_path / "state"
    dispatch_dir = tmp_path / "dispatch"
    forecast_dir.mkdir()
    state_dir.mkdir()
    dispatch_dir.mkdir()

    now = pd.Timestamp.now(tz="UTC").floor("h")
    bogus = now + pd.Timedelta(hours=daemon.MAX_FUTURE_SKEW_HOURS + 24)
    legit = now + pd.Timedelta(hours=1)
    (forecast_dir / f"{bogus.strftime('%Y-%m-%dT%H-%M-%SZ')}.parquet").write_text("")

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

    assert work_queue.empty()
    assert scan_state.last_enqueued is None

    # A legitimate forecast arriving afterwards must still be picked up.
    (forecast_dir / f"{legit.strftime('%Y-%m-%dT%H-%M-%SZ')}.parquet").write_text("")
    daemon._scan(cfg, work_queue, scan_state)

    assert work_queue.get_nowait() == legit
    assert scan_state.last_enqueued == legit


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
    initial_state = _cs(solve_time)
    initial_state.save(current)

    seen: dict[str, object] = {}

    def fake_run_one_cycle(**kwargs):
        dispatch_out = kwargs["dispatch_out"]
        state_out = kwargs["state_out"]
        seen["dispatch_out"] = dispatch_out
        seen["state_out"] = state_out
        dispatch_out.write_text("staged dispatch")
        _cs(solve_time + pd.Timedelta(hours=1)).save(state_out)
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

    final_dispatch = dispatch_dir / "2026-05-07T13-00-00Z.parquet"
    staged_dispatch = final_dispatch.with_suffix(".parquet.tmp")
    assert rc == 0
    assert seen["dispatch_out"] == staged_dispatch
    assert final_dispatch.exists()
    assert not staged_dispatch.exists()
    assert current.is_symlink()
    assert current.resolve().name == "2026-05-07T13-00-00Z.json"


def test_run_one_publishes_dispatch_before_advancing_state(tmp_path, monkeypatch):
    forecast_dir = tmp_path / "forecast"
    state_dir = tmp_path / "state"
    dispatch_dir = tmp_path / "dispatch"
    forecast_dir.mkdir()
    state_dir.mkdir()
    dispatch_dir.mkdir()

    solve_time = pd.Timestamp("2026-05-07T13:00:00Z")
    current = state_dir / "current.json"
    _cs(solve_time).save(current)

    def fake_run_one_cycle(**kwargs):
        kwargs["dispatch_out"].write_text("staged dispatch")
        _cs(solve_time + pd.Timedelta(hours=1)).save(kwargs["state_out"])
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
    dispatch_path = dispatch_dir / "2026-05-07T13-00-00Z.parquet"
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
    original_state = _cs(solve_time)
    original_state.save(current)

    def fake_run_one_cycle(**kwargs):
        kwargs["dispatch_out"].write_text("staged dispatch")
        _cs(solve_time + pd.Timedelta(hours=1)).save(kwargs["state_out"])
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

    assert not (dispatch_dir / "2026-05-07T13-00-00Z.parquet").exists()
    assert not (dispatch_dir / "2026-05-07T13-00-00Z.parquet.tmp").exists()
    assert not (state_dir / "2026-05-07T13-00-00Z.json").exists()
    assert DispatchState.load(current).timestamp == solve_time


def test_init_state_writes_symlink_and_heartbeat(tmp_path):
    state_out = tmp_path / "state" / "current.json"
    ts = "2026-05-07T13:00:00+00:00"

    rc = init_state.main(["--state-out", str(state_out), "--solve-time", ts])

    assert rc == 0
    assert state_out.is_symlink()
    assert state_out.resolve().name == "2026-05-07T13-00-00Z.json"
    assert DispatchState.load(state_out).timestamp == pd.Timestamp(ts)
    assert (state_out.parent / ".heartbeat").exists()


def test_init_state_backfills_missing_heartbeat_for_existing_state(tmp_path):
    state_out = tmp_path / "state" / "current.json"
    ts = pd.Timestamp("2026-05-07T13:00:00Z")
    _cs(ts).save(state_out)

    rc = init_state.main(["--state-out", str(state_out)])

    assert rc == 0
    assert DispatchState.load(state_out).timestamp == ts
    assert (state_out.parent / ".heartbeat").exists()


def test_runtime_entrypoints_import():
    assert callable(daemon.run)
    assert callable(init_state.main)


def test_daemon_config_from_env_reads_config_file(monkeypatch, tmp_path):
    """CONFIG_FILE env must surface as DaemonConfig.config_file (Path)."""
    cfg_path = tmp_path / "plant_config.json"
    monkeypatch.setenv("CONFIG_FILE", str(cfg_path))
    cfg = daemon.DaemonConfig.from_env()
    assert cfg.config_file == cfg_path

    monkeypatch.delenv("CONFIG_FILE", raising=False)
    cfg2 = daemon.DaemonConfig.from_env()
    assert cfg2.config_file is None


def test_run_one_passes_plant_config_through_to_run_one_cycle(tmp_path, monkeypatch):
    """The daemon must thread its loaded PlantConfig into run_one_cycle.

    Regression for the bug where the daemon silently used legacy_default()
    even when an operator had pointed CONFIG_FILE at a custom plant config.
    """
    forecast_dir = tmp_path / "forecast"
    state_dir = tmp_path / "state"
    dispatch_dir = tmp_path / "dispatch"
    forecast_dir.mkdir()
    state_dir.mkdir()
    dispatch_dir.mkdir()

    custom_cfg = replace(
        PlantConfig.legacy_default(),
        heat_pumps=(
            HeatPumpParams(id="hp_a", p_el_min_mw=0.5, p_el_max_mw=4.0, cop=3.5),
            HeatPumpParams(id="hp_b", p_el_min_mw=0.5, p_el_max_mw=4.0, cop=3.5),
        ),
    )

    solve_time = pd.Timestamp("2026-05-07T13:00:00Z")
    current = state_dir / "current.json"
    DispatchState.cold_start(custom_cfg, solve_time).save(current)

    seen: dict[str, object] = {}

    def fake_run_one_cycle(**kwargs):
        seen["config"] = kwargs.get("config")
        kwargs["dispatch_out"].write_text("staged")
        DispatchState.cold_start(
            custom_cfg, solve_time + pd.Timedelta(hours=1),
        ).save(kwargs["state_out"])
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

    rc = daemon._run_one(cfg, solve_time, plant_config=custom_cfg)
    assert rc == 0
    assert seen["config"] is custom_cfg


# --------------------------------------------------------------------------- #
# Live config reload (Task 3): the worker re-stats cfg.config_file at every
# cycle boundary and swaps the in-memory PlantConfig when a valid, same-asset-
# set change is on disk. Failures and asset-set changes are logged and the
# daemon keeps running on the previously-loaded config.
# --------------------------------------------------------------------------- #


def _bump_mtime(path):
    """Force mtime to actually advance past whatever os.utime resolution gives.

    The validator-rejection tests rely on the watermark moving on every observed
    bump even when no swap happens, so they need real mtime changes — not a
    write that lands in the same filesystem timestamp bucket.
    """
    st = path.stat()
    os.utime(path, (st.st_atime + 1, st.st_mtime + 1))


def _write_config(path, cfg):
    path.write_text(json.dumps(cfg.to_dict(), indent=2))


def test_maybe_reload_returns_current_when_config_file_is_none():
    state = daemon._ConfigReloadState()
    cfg = PlantConfig.legacy_default()
    assert daemon._maybe_reload_plant_config(None, cfg, state) is cfg
    assert state.last_mtime_ns is None  # watermark untouched in disabled mode


def test_maybe_reload_no_swap_when_mtime_unchanged(tmp_path):
    config_path = tmp_path / "plant_config.json"
    cfg = PlantConfig.legacy_default()
    _write_config(config_path, cfg)
    state = daemon._ConfigReloadState(last_mtime_ns=config_path.stat().st_mtime_ns)

    result = daemon._maybe_reload_plant_config(config_path, cfg, state)

    assert result is cfg


def test_maybe_reload_applies_parameter_only_change(tmp_path):
    config_path = tmp_path / "plant_config.json"
    base = PlantConfig.legacy_default()
    _write_config(config_path, base)
    state = daemon._ConfigReloadState(last_mtime_ns=config_path.stat().st_mtime_ns)

    bumped = replace(base, gas_price_eur_mwh_hs=base.gas_price_eur_mwh_hs + 5.0)
    _write_config(config_path, bumped)
    _bump_mtime(config_path)

    result = daemon._maybe_reload_plant_config(config_path, base, state)

    assert result is not base
    assert result.gas_price_eur_mwh_hs == bumped.gas_price_eur_mwh_hs
    assert state.last_mtime_ns == config_path.stat().st_mtime_ns


def test_maybe_reload_rejects_asset_id_diff(tmp_path, caplog):
    """Adding an asset is Task 4 territory; reload must not swap and must say so."""
    config_path = tmp_path / "plant_config.json"
    base = PlantConfig.legacy_default()
    _write_config(config_path, base)
    state = daemon._ConfigReloadState(last_mtime_ns=config_path.stat().st_mtime_ns)

    grown = replace(
        base,
        heat_pumps=base.heat_pumps + (
            HeatPumpParams(id="hp_new", p_el_min_mw=1.0, p_el_max_mw=4.0, cop=3.5),
        ),
    )
    _write_config(config_path, grown)
    _bump_mtime(config_path)

    with caplog.at_level("ERROR", logger="optimization.daemon"):
        result = daemon._maybe_reload_plant_config(config_path, base, state)

    assert result is base  # not swapped
    assert any("asset id set changed" in r.message for r in caplog.records)
    # Watermark advanced anyway — broken file should not re-trigger every cycle.
    assert state.last_mtime_ns == config_path.stat().st_mtime_ns


def test_maybe_reload_rejects_validation_errors_with_full_diff(tmp_path, caplog):
    config_path = tmp_path / "plant_config.json"
    base = PlantConfig.legacy_default()
    _write_config(config_path, base)
    state = daemon._ConfigReloadState(last_mtime_ns=config_path.stat().st_mtime_ns)

    # Write a payload with multiple errors — the daemon must log all of them,
    # mirroring the collect-all contract from validate_plant_payload.
    broken = base.to_dict()
    broken["heat_pumps"][0]["p_el_min_mw"] = 99.0  # MIN_EXCEEDS_MAX (max=8)
    broken["co2_price_eur_per_t"] = -1.0           # NEGATIVE
    config_path.write_text(json.dumps(broken))
    _bump_mtime(config_path)

    with caplog.at_level("ERROR", logger="optimization.daemon"):
        result = daemon._maybe_reload_plant_config(config_path, base, state)

    assert result is base
    msg = "\n".join(r.message for r in caplog.records)
    assert "validation failed" in msg
    assert "MIN_EXCEEDS_MAX" in msg
    assert "NEGATIVE" in msg
    # Watermark advanced — broken file does not get re-tried until next save.
    assert state.last_mtime_ns == config_path.stat().st_mtime_ns


def test_maybe_reload_tolerates_malformed_json(tmp_path, caplog):
    config_path = tmp_path / "plant_config.json"
    base = PlantConfig.legacy_default()
    _write_config(config_path, base)
    state = daemon._ConfigReloadState(last_mtime_ns=config_path.stat().st_mtime_ns)

    config_path.write_text("{not json")
    _bump_mtime(config_path)

    with caplog.at_level("ERROR", logger="optimization.daemon"):
        result = daemon._maybe_reload_plant_config(config_path, base, state)

    assert result is base
    assert any("could not load" in r.message for r in caplog.records)


def test_maybe_reload_surfaces_warnings_when_swap_succeeds(tmp_path, caplog):
    """A config that validates with warnings still swaps — but warnings get logged."""
    config_path = tmp_path / "plant_config.json"
    base = PlantConfig.legacy_default()
    _write_config(config_path, base)
    state = daemon._ConfigReloadState(last_mtime_ns=config_path.stat().st_mtime_ns)

    # STORAGE_DISCHARGE_DISABLED warning: legal but suspect.
    with_warning = replace(
        base,
        storages=(replace(base.storages[0], discharge_max_mw_th=0.0),),
    )
    _write_config(config_path, with_warning)
    _bump_mtime(config_path)

    with caplog.at_level("WARNING", logger="optimization.daemon"):
        result = daemon._maybe_reload_plant_config(config_path, base, state)

    assert result is not base
    assert result.storages[0].discharge_max_mw_th == 0.0
    msg = "\n".join(r.message for r in caplog.records)
    assert "STORAGE_DISCHARGE_DISABLED" in msg


def test_maybe_reload_tolerates_missing_file(tmp_path, caplog):
    """File deleted between cycles: keep running on the previously-loaded config."""
    config_path = tmp_path / "plant_config.json"
    base = PlantConfig.legacy_default()
    state = daemon._ConfigReloadState(last_mtime_ns=12345)  # arbitrary seed

    with caplog.at_level("WARNING", logger="optimization.daemon"):
        result = daemon._maybe_reload_plant_config(config_path, base, state)

    assert result is base
    assert any("not statable" in r.message for r in caplog.records)


def test_worker_passes_reloaded_config_to_next_cycle(tmp_path, monkeypatch):
    """End-to-end: an mtime bump between cycles flows through to run_one_cycle."""
    forecast_dir = tmp_path / "forecast"
    state_dir = tmp_path / "state"
    dispatch_dir = tmp_path / "dispatch"
    config_path = tmp_path / "plant_config.json"
    forecast_dir.mkdir()
    state_dir.mkdir()
    dispatch_dir.mkdir()

    base = PlantConfig.legacy_default()
    _write_config(config_path, base)
    bumped = replace(base, gas_price_eur_mwh_hs=base.gas_price_eur_mwh_hs + 7.0)

    # Use a recent t0 so the next-hour forecast t1 passes _should_run's
    # stale/future guards (the reload hook is gated behind _should_run).
    t0 = pd.Timestamp.now(tz="UTC").floor("h")
    t1 = t0 + pd.Timedelta(hours=1)
    DispatchState.cold_start(base, t0).save(state_dir / "current.json")

    seen_configs: list[PlantConfig] = []

    def fake_run_one_cycle(**kwargs):
        seen_configs.append(kwargs["config"])
        kwargs["dispatch_out"].write_text("staged")
        DispatchState.cold_start(base, kwargs["solve_time"] + pd.Timedelta(hours=1)).save(
            kwargs["state_out"]
        )
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
        config_file=config_path,
    )

    work_queue: "queue.Queue[object]" = queue.Queue()
    work_queue.put(t1)
    # Edit the config between enqueue and the worker dequeue — exactly the
    # situation the reload hook is designed to catch.
    _write_config(config_path, bumped)
    _bump_mtime(config_path)
    work_queue.put(daemon.SHUTDOWN_SENTINEL)

    import threading as _t
    stop_event = _t.Event()
    reload_state = daemon._ConfigReloadState(last_mtime_ns=config_path.stat().st_mtime_ns - 1)

    daemon._worker(cfg, work_queue, stop_event, plant_config=base, reload_state=reload_state)

    assert len(seen_configs) == 1
    assert seen_configs[0].gas_price_eur_mwh_hs == bumped.gas_price_eur_mwh_hs


def test_run_one_passes_none_when_no_plant_config(tmp_path, monkeypatch):
    """When no CONFIG_FILE is set, plant_config defaults to None and
    run_one_cycle falls back to legacy_default() inside itself."""
    forecast_dir = tmp_path / "forecast"
    state_dir = tmp_path / "state"
    dispatch_dir = tmp_path / "dispatch"
    forecast_dir.mkdir()
    state_dir.mkdir()
    dispatch_dir.mkdir()

    solve_time = pd.Timestamp("2026-05-07T13:00:00Z")
    _cs(solve_time).save(state_dir / "current.json")

    seen: dict[str, object] = {}

    def fake_run_one_cycle(**kwargs):
        seen["config"] = kwargs.get("config")
        kwargs["dispatch_out"].write_text("staged")
        _cs(solve_time + pd.Timedelta(hours=1)).save(kwargs["state_out"])
        return 0

    monkeypatch.setattr(daemon, "run_one_cycle", fake_run_one_cycle)

    cfg = _make_cfg(tmp_path)
    rc = daemon._run_one(cfg, solve_time)  # no plant_config kwarg
    assert rc == 0
    assert seen["config"] is None
