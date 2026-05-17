"""Daemon and init-state regressions."""
from __future__ import annotations

import json
import os
import queue
from dataclasses import replace

import pandas as pd
import pytest

from optimization import daemon, init_state
from optimization.config import HeatPumpParams, PlantConfig
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


def test_should_run_accepts_historical_solve_time(tmp_path):
    """No wall-clock guard: a months-old solve_time must run.

    The MVP runs against a CSV-backed replay forecaster whose solve_times
    sit months in the past. The daemon must process them; monotonicity is
    the only invariant that matters.
    """
    cfg = _make_cfg(tmp_path)
    historical = pd.Timestamp("2024-08-15T09:00:00Z")
    assert daemon._should_run(cfg, historical, last_solve_time=None) is True


def test_should_run_blocks_non_monotonic_solve_time(tmp_path):
    cfg = _make_cfg(tmp_path)
    last = pd.Timestamp("2024-08-15T10:00:00Z")
    earlier = pd.Timestamp("2024-08-15T09:00:00Z")
    assert daemon._should_run(cfg, earlier, last_solve_time=last) is False
    assert daemon._should_run(cfg, last, last_solve_time=last) is False


def test_should_run_accepts_old_forecast_when_stale_grace_disabled(tmp_path, monkeypatch):
    """Default (STALE_GRACE_HOURS=0): months-old solve_times pass.

    Belt-and-suspenders against accidental re-introduction of the floor as
    a hard-coded constant — the replay producer depends on this.
    """
    monkeypatch.setattr(daemon, "STALE_GRACE_HOURS", 0)
    cfg = _make_cfg(tmp_path)
    very_old = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=90)
    assert daemon._should_run(cfg, very_old, last_solve_time=None) is True


def test_should_run_rejects_stale_forecast_when_grace_enabled(tmp_path, monkeypatch):
    """With STALE_GRACE_HOURS>0 (live-feed config): old solve_times are dropped.

    Models the live-feed cutover: a wedged upstream producer feeding
    hours-old forecasts must not silently drive the optimizer off
    real-time prices.
    """
    monkeypatch.setattr(daemon, "STALE_GRACE_HOURS", 2)
    cfg = _make_cfg(tmp_path)
    stale = pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(hours=5)
    assert daemon._should_run(cfg, stale, last_solve_time=None) is False
    # Forecast within the grace window still passes.
    fresh = pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(hours=1)
    assert daemon._should_run(cfg, fresh, last_solve_time=None) is True


def test_stale_grace_hours_module_constant_is_int():
    """Regression: STALE_GRACE_HOURS must be parsed at import time so a
    typo crashes the daemon at startup, not at the first scan tick."""
    assert hasattr(daemon, "STALE_GRACE_HOURS")
    assert isinstance(daemon.STALE_GRACE_HOURS, int)
    assert daemon.STALE_GRACE_HOURS >= 0


def test_stale_grace_hours_default_is_zero(monkeypatch):
    monkeypatch.delenv("STALE_GRACE_HOURS", raising=False)
    assert daemon._stale_grace_hours_from_env() == 0


def test_stale_grace_hours_positive_value_parses(monkeypatch):
    monkeypatch.setenv("STALE_GRACE_HOURS", "2")
    assert daemon._stale_grace_hours_from_env() == 2


def test_stale_grace_hours_rejects_non_integer(monkeypatch):
    """Fail loud on operator typo. Silent fallback would let a typo like
    '2h' disable the floor exactly when the operator wants it enabled."""
    monkeypatch.setenv("STALE_GRACE_HOURS", "two")
    with pytest.raises(ValueError, match="STALE_GRACE_HOURS"):
        daemon._stale_grace_hours_from_env()


def test_stale_grace_hours_rejects_negative(monkeypatch):
    monkeypatch.setenv("STALE_GRACE_HOURS", "-1")
    with pytest.raises(ValueError, match="STALE_GRACE_HOURS"):
        daemon._stale_grace_hours_from_env()


def test_should_run_rejects_implausibly_future_forecast(tmp_path):
    """Off-by-year filenames must not be processed.

    A solve_time far in the future (clock-skewed writer, typo in a manual
    drop, off-by-year bug) would otherwise advance state to that timestamp
    and silently block every real forecast that follows.
    """
    cfg = _make_cfg(tmp_path)
    solve_time = pd.Timestamp.now(tz="UTC").ceil("h") + pd.Timedelta(
        hours=daemon.MAX_FUTURE_SKEW_HOURS + 2
    )

    assert daemon._should_run(cfg, solve_time, last_solve_time=None) is False


def test_scan_does_not_advance_watermark_for_future_forecast(tmp_path):
    """A bogus far-future forecast must not poison the scanner watermark.

    Regression: without the cap, _scan would advance ``last_enqueued`` to the
    far-future timestamp and every legitimate forecast afterwards would be
    filtered out as "already seen".
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


def test_maybe_reload_returns_current_when_current_is_none(tmp_path):
    """legacy_default mode never loads a PlantConfig, so the worker passes
    current=None. The reload hook must short-circuit without touching the
    filesystem or the watermark — there is nothing to compare against."""
    config_path = tmp_path / "plant_config.json"  # deliberately not created
    state = daemon._ConfigReloadState()
    assert daemon._maybe_reload_plant_config(config_path, None, state) is None
    assert state.last_mtime_ns is None


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


def test_maybe_reload_rejects_family_move(tmp_path, caplog):
    """Same string id moving between HP/Boiler/CHP/Storage is a destructive
    schema change disguised as a rename; reload must refuse and tell the
    operator to do it as two separate edits (remove then add)."""
    from optimization.config import CHPParams
    config_path = tmp_path / "plant_config.json"
    base = PlantConfig.legacy_default()
    _write_config(config_path, base)
    state = daemon._ConfigReloadState(last_mtime_ns=config_path.stat().st_mtime_ns)

    moved = replace(
        base,
        boilers=(),
        chps=base.chps + (CHPParams(
            id="boiler",  # reuses the old boiler's id as a CHP
            p_el_min_mw=1.0, p_el_max_mw=4.0,
            eff_el=0.4, eff_th=0.4,
            min_up_steps=4, min_down_steps=4,
            startup_cost_eur=100.0,
        ),),
    )
    _write_config(config_path, moved)
    _bump_mtime(config_path)

    with caplog.at_level("ERROR", logger="optimization.daemon"):
        result = daemon._maybe_reload_plant_config(config_path, base, state)

    assert result is base  # not swapped
    msg = "\n".join(r.message for r in caplog.records)
    assert "changed family" in msg
    assert "'boiler'" in msg
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
    assert "validation error" in msg
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


def test_maybe_reload_no_op_when_content_identical(tmp_path, caplog):
    """``touch`` (or any mtime bump with unchanged content) must not swap and
    must not produce a 'config reloaded' log — otherwise idle saves spam logs
    and confuse the operator about what changed."""
    config_path = tmp_path / "plant_config.json"
    base = PlantConfig.legacy_default()
    _write_config(config_path, base)
    state = daemon._ConfigReloadState(last_mtime_ns=config_path.stat().st_mtime_ns)

    # Bump mtime without changing content.
    _bump_mtime(config_path)

    with caplog.at_level("INFO", logger="optimization.daemon"):
        result = daemon._maybe_reload_plant_config(config_path, base, state)

    assert result is base  # identity preserved on no-op
    # Watermark still advances so the comparison is fast next time.
    assert state.last_mtime_ns == config_path.stat().st_mtime_ns
    # And critically: nothing logged.
    assert all("config reload" not in r.message for r in caplog.records)


def test_maybe_reload_rejects_when_state_infeasible_under_new_config(tmp_path, caplog):
    """Parameter-only change that strands the stored SoC outside the new
    [floor, capacity] must be rejected — same-asset-set alone is not enough
    to guarantee the existing state is still feasible."""
    config_path = tmp_path / "plant_config.json"
    state_path = tmp_path / "current.json"
    base = PlantConfig.legacy_default()
    _write_config(config_path, base)

    # Save a state whose SoC sits comfortably above the original floor.
    stored = base.storages[0]
    state_dispatch = DispatchState.cold_start(base, pd.Timestamp.now(tz="UTC"))
    state_dispatch.storages[stored.id].soc_mwh_th = stored.floor_mwh_th + 5.0
    state_dispatch.save(state_path)

    # New config: raise floor above the current SoC. Validation passes
    # (0 <= floor <= capacity), but the state can't survive the swap.
    raised_floor = state_dispatch.storages[stored.id].soc_mwh_th + 10.0
    bumped = replace(
        base,
        storages=(replace(stored, floor_mwh_th=raised_floor),),
    )
    _write_config(config_path, bumped)
    _bump_mtime(config_path)
    reload_state = daemon._ConfigReloadState(
        last_mtime_ns=config_path.stat().st_mtime_ns - 1
    )

    with caplog.at_level("ERROR", logger="optimization.daemon"):
        result = daemon._maybe_reload_plant_config(
            config_path, base, reload_state, state_path=state_path
        )

    assert result is base  # reverted
    msg = "\n".join(r.message for r in caplog.records)
    assert "state is infeasible" in msg
    assert "below new floor_mwh_th" in msg
    # Watermark advanced — broken edit doesn't re-trigger every cycle.
    assert reload_state.last_mtime_ns == config_path.stat().st_mtime_ns


def test_maybe_reload_state_infeasibility_reject_surfaces_warnings(tmp_path, caplog):
    """When the state-feasibility gate rejects a swap, the operator should
    still see any warnings the candidate carried — same contract as the
    asset-id-diff rejection. Otherwise the only signal is the bound bust
    and the warning is lost until the operator retries."""
    config_path = tmp_path / "plant_config.json"
    state_path = tmp_path / "current.json"
    base = PlantConfig.legacy_default()
    _write_config(config_path, base)

    stored = base.storages[0]
    state_dispatch = DispatchState.cold_start(base, pd.Timestamp.now(tz="UTC"))
    state_dispatch.storages[stored.id].soc_mwh_th = stored.floor_mwh_th + 5.0
    state_dispatch.save(state_path)

    # Both: raise floor above stored SoC (state-infeasible, hard reject) AND
    # disable discharge (legal but warning-eligible — STORAGE_DISCHARGE_DISABLED).
    raised_floor = state_dispatch.storages[stored.id].soc_mwh_th + 10.0
    candidate = replace(
        base,
        storages=(
            replace(
                stored,
                floor_mwh_th=raised_floor,
                discharge_max_mw_th=0.0,
            ),
        ),
    )
    _write_config(config_path, candidate)
    _bump_mtime(config_path)
    reload_state = daemon._ConfigReloadState(
        last_mtime_ns=config_path.stat().st_mtime_ns - 1
    )

    with caplog.at_level("WARNING", logger="optimization.daemon"):
        result = daemon._maybe_reload_plant_config(
            config_path, base, reload_state, state_path=state_path
        )

    assert result is base
    msg = "\n".join(r.message for r in caplog.records)
    # The infeasibility error must be there, but so must the warning record.
    assert "state is infeasible" in msg
    assert "STORAGE_DISCHARGE_DISABLED" in msg


def test_maybe_reload_state_path_none_skips_feasibility_check(tmp_path):
    """Callers that don't have a state path (tests, init-time) must not be
    blocked. The check is skipped silently — the worker's real path always
    passes one."""
    config_path = tmp_path / "plant_config.json"
    base = PlantConfig.legacy_default()
    _write_config(config_path, base)

    bumped = replace(base, gas_price_eur_mwh_hs=base.gas_price_eur_mwh_hs + 1.0)
    _write_config(config_path, bumped)
    state = daemon._ConfigReloadState(
        last_mtime_ns=config_path.stat().st_mtime_ns - 1
    )

    # No state_path → feasibility check skipped, swap proceeds.
    result = daemon._maybe_reload_plant_config(
        config_path, base, state, state_path=None
    )

    assert result is not base
    assert result.gas_price_eur_mwh_hs == bumped.gas_price_eur_mwh_hs


def test_maybe_reload_missing_state_file_skips_feasibility_check(tmp_path):
    """If current.json doesn't exist yet, the swap goes through. The worker's
    own _run_one will error if it ever actually needs a state file."""
    config_path = tmp_path / "plant_config.json"
    state_path = tmp_path / "does_not_exist.json"
    base = PlantConfig.legacy_default()
    _write_config(config_path, base)

    bumped = replace(base, gas_price_eur_mwh_hs=base.gas_price_eur_mwh_hs + 1.0)
    _write_config(config_path, bumped)
    state = daemon._ConfigReloadState(
        last_mtime_ns=config_path.stat().st_mtime_ns - 1
    )

    result = daemon._maybe_reload_plant_config(
        config_path, base, state, state_path=state_path
    )

    assert result is not base


def test_maybe_reload_unreadable_state_still_applies_swap(tmp_path, caplog):
    """Corrupt state file: log a warning and apply the swap anyway. The state
    problem is the daemon's bigger issue — don't double-fail the operator's
    legitimate config edit on it."""
    config_path = tmp_path / "plant_config.json"
    state_path = tmp_path / "current.json"
    base = PlantConfig.legacy_default()
    _write_config(config_path, base)
    state_path.write_text("{ not json")

    bumped = replace(base, gas_price_eur_mwh_hs=base.gas_price_eur_mwh_hs + 1.0)
    _write_config(config_path, bumped)
    reload_state = daemon._ConfigReloadState(
        last_mtime_ns=config_path.stat().st_mtime_ns - 1
    )

    with caplog.at_level("WARNING", logger="optimization.daemon"):
        result = daemon._maybe_reload_plant_config(
            config_path, base, reload_state, state_path=state_path
        )

    assert result is not base
    msg = "\n".join(r.message for r in caplog.records)
    assert "could not read state" in msg


def test_load_plant_config_with_watermark_stats_before_read(tmp_path, monkeypatch):
    """Startup race: a write between stat and from_json must not be silently
    swallowed by a post-read watermark. We verify stat() returns a value
    captured BEFORE PlantConfig.from_json is called by simulating the race
    via a from_json shim that bumps mtime."""
    config_path = tmp_path / "plant_config.json"
    base = PlantConfig.legacy_default()
    _write_config(config_path, base)
    pre_mtime = config_path.stat().st_mtime_ns

    orig_from_json_with_result = PlantConfig.from_json_with_result

    def racing_from_json_with_result(path):
        # Simulate an operator write landing during the read.
        _bump_mtime(path)
        return orig_from_json_with_result(path)

    monkeypatch.setattr(
        PlantConfig,
        "from_json_with_result",
        staticmethod(racing_from_json_with_result),
    )

    cfg, watermark, _result = daemon._load_plant_config_with_watermark(config_path)

    assert cfg == base
    # Watermark is the pre-read value, not the post-read one. So when the
    # worker stats next, it will see the bumped mtime and trigger a reload.
    assert watermark == pre_mtime
    assert config_path.stat().st_mtime_ns > pre_mtime


def test_load_plant_config_with_watermark_tolerates_stat_failure(tmp_path, monkeypatch):
    """If pre-read stat fails for any reason, watermark is None and the
    next worker cycle reloads on first stat — better than aborting startup."""
    config_path = tmp_path / "plant_config.json"
    base = PlantConfig.legacy_default()
    _write_config(config_path, base)

    real_stat = type(config_path).stat
    call = {"count": 0}

    def flaky_stat(self, *args, **kwargs):
        call["count"] += 1
        if call["count"] == 1:
            raise OSError("transient")
        return real_stat(self, *args, **kwargs)

    # Patch on the concrete subclass returned by Path(...) so from_json's
    # own stat (inside Path.read_text) still works.
    monkeypatch.setattr(type(config_path), "stat", flaky_stat)

    cfg, watermark, _result = daemon._load_plant_config_with_watermark(config_path)

    assert cfg == base
    assert watermark is None


def test_load_plant_config_with_watermark_returns_warnings(tmp_path):
    """Warnings on the startup load must reach the caller. Without this the
    daemon would run a flagged config without ever logging the flag."""
    config_path = tmp_path / "plant_config.json"
    base = PlantConfig.legacy_default()
    # STORAGE_DISCHARGE_DISABLED: legal but warning-eligible.
    with_warning = replace(
        base,
        storages=(replace(base.storages[0], discharge_max_mw_th=0.0),),
    )
    _write_config(config_path, with_warning)

    _cfg, _watermark, result = daemon._load_plant_config_with_watermark(config_path)

    assert not result.errors
    assert result.warnings  # warning surfaced, not swallowed
    codes = {issue.code for issue in result.warnings}
    assert "STORAGE_DISCHARGE_DISABLED" in codes


def test_log_startup_warnings_emits_warning_record(tmp_path, caplog):
    """The startup warning logger must produce a WARNING record naming the
    issue codes — that is the operator-facing observability fix."""
    config_path = tmp_path / "plant_config.json"
    base = PlantConfig.legacy_default()
    with_warning = replace(
        base,
        storages=(replace(base.storages[0], discharge_max_mw_th=0.0),),
    )
    _write_config(config_path, with_warning)
    _cfg, _watermark, result = daemon._load_plant_config_with_watermark(config_path)

    with caplog.at_level("WARNING", logger="optimization.daemon"):
        daemon._log_startup_warnings(config_path, result)

    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert warnings, "expected at least one WARNING record"
    msg = "\n".join(r.message for r in warnings)
    assert "STORAGE_DISCHARGE_DISABLED" in msg
    assert str(config_path) in msg


def test_load_plant_config_raises_with_full_result_on_multi_error_payload(tmp_path):
    """``ConfigValidationError`` carries the full ValidationResult, not just
    the first message. Without this the startup error log can only ever
    surface one issue per boot attempt."""
    config_path = tmp_path / "plant_config.json"
    # Build a payload with independent errors in different places.
    base_payload = PlantConfig.legacy_default().to_dict()
    base_payload["gas_price_eur_mwh_hs"] = -1.0          # NEGATIVE
    base_payload["heat_pumps"][0]["p_el_min_mw"] = 99.0  # MIN_EXCEEDS_MAX
    base_payload["boilers"][0]["eff"] = 2.0              # EFF_OUT_OF_RANGE
    config_path.write_text(json.dumps(base_payload, indent=2))

    from optimization.config_validation import ConfigValidationError
    try:
        daemon._load_plant_config_with_watermark(config_path)
    except ConfigValidationError as e:
        codes = {i.code for i in e.result.errors}
        assert {"NEGATIVE", "MIN_EXCEEDS_MAX", "EFF_OUT_OF_RANGE"} <= codes
    else:
        raise AssertionError("expected ConfigValidationError")


def test_log_startup_validation_failure_logs_every_error(tmp_path, caplog):
    """The startup error logger must enumerate every error, not just the
    first one ``str(exc)`` would carry. This is the operator-facing fix on
    the error side of the boot path."""
    config_path = tmp_path / "plant_config.json"
    base_payload = PlantConfig.legacy_default().to_dict()
    base_payload["gas_price_eur_mwh_hs"] = -1.0
    base_payload["heat_pumps"][0]["p_el_min_mw"] = 99.0
    base_payload["boilers"][0]["eff"] = 2.0
    config_path.write_text(json.dumps(base_payload, indent=2))

    from optimization.config_validation import ConfigValidationError
    try:
        daemon._load_plant_config_with_watermark(config_path)
    except ConfigValidationError as e:
        with caplog.at_level("ERROR", logger="optimization.daemon"):
            daemon._log_startup_validation_failure(config_path, e.result)

    msg = "\n".join(r.message for r in caplog.records if r.levelname == "ERROR")
    assert "NEGATIVE" in msg
    assert "MIN_EXCEEDS_MAX" in msg
    assert "EFF_OUT_OF_RANGE" in msg
    assert str(config_path) in msg


def test_log_startup_warnings_silent_when_no_warnings(tmp_path, caplog):
    """Clean config: no spurious log record. Operators should only hear from
    this logger when there is something to look at."""
    config_path = tmp_path / "plant_config.json"
    base = PlantConfig.legacy_default()
    _write_config(config_path, base)
    _cfg, _watermark, result = daemon._load_plant_config_with_watermark(config_path)

    with caplog.at_level("WARNING", logger="optimization.daemon"):
        daemon._log_startup_warnings(config_path, result)

    assert not result.warnings
    assert not [r for r in caplog.records if r.levelname == "WARNING"]


# --------------------------------------------------------------------------- #
# Live add/remove of assets (Task 4): the reload path runs
# DispatchState.migrate_to on the candidate config, retargets current.json
# to the migrated state, and backs up the pre-migration state on any
# destructive (remove) edit.
# --------------------------------------------------------------------------- #


def _seed_state_and_link(state_dir, state):
    """Helper: write a dated state and point current.json at it via symlink.
    The reload code retargets the symlink; tests using --state-path on a
    plain file would skip that retarget path entirely."""
    state_dir.mkdir(parents=True, exist_ok=True)
    dated = state_dir / "seed.json"
    state.save(dated)
    current = state_dir / "current.json"
    if current.exists() or current.is_symlink():
        current.unlink()
    current.symlink_to(dated.name)
    return current


def test_maybe_reload_applies_add_unit_and_migrates_state(tmp_path, caplog):
    """Adding an HP triggers state migration: current.json points to a new
    file that contains the original entries plus an off/TIS_LONG entry for
    the new HP. The original dated state file remains on disk."""
    config_path = tmp_path / "plant_config.json"
    state_dir = tmp_path / "state"
    base = PlantConfig.legacy_default()
    _write_config(config_path, base)
    seed_ts = pd.Timestamp("2026-05-07T13:00:00Z")
    _seed_state_and_link(state_dir, _cs(seed_ts))

    grown = replace(
        base,
        heat_pumps=base.heat_pumps + (
            HeatPumpParams(id="hp_new", p_el_min_mw=0.5, p_el_max_mw=4.0, cop=3.4),
        ),
    )
    _write_config(config_path, grown)
    _bump_mtime(config_path)

    reload_state = daemon._ConfigReloadState()
    with caplog.at_level("INFO", logger="optimization.daemon"):
        result = daemon._maybe_reload_plant_config(
            config_path, base, reload_state,
            state_path=state_dir / "current.json",
            state_dir=state_dir,
        )

    assert result == grown
    # current.json now points at a migrate-* file with the new asset entry.
    loaded = DispatchState.load(state_dir / "current.json")
    assert "hp_new" in loaded.units
    assert loaded.units["hp_new"].on == 0
    # Existing entries untouched.
    for uid in ("hp", "boiler", "chp"):
        assert uid in loaded.units
    # Pre-migration file is still on disk (we point at the new one).
    assert (state_dir / "seed.json").exists()
    # No destructive backup written for a pure add.
    assert not any(p.name.startswith("before-migrate-") for p in state_dir.iterdir())


def test_maybe_reload_applies_remove_unit_and_writes_backup(tmp_path, caplog):
    """Removing a boiler drops the boiler state and writes a before-migrate
    backup; current.json is retargeted to the post-migration state."""
    config_path = tmp_path / "plant_config.json"
    state_dir = tmp_path / "state"
    base = PlantConfig.legacy_default()
    _write_config(config_path, base)
    seed_ts = pd.Timestamp("2026-05-07T13:00:00Z")
    _seed_state_and_link(state_dir, _cs(seed_ts))

    shrunk = replace(base, boilers=())
    _write_config(config_path, shrunk)
    _bump_mtime(config_path)

    reload_state = daemon._ConfigReloadState()
    with caplog.at_level("WARNING", logger="optimization.daemon"):
        result = daemon._maybe_reload_plant_config(
            config_path, base, reload_state,
            state_path=state_dir / "current.json",
            state_dir=state_dir,
        )

    assert result == shrunk
    loaded = DispatchState.load(state_dir / "current.json")
    assert "boiler" not in loaded.units
    # Backup written, contains the original boiler entry.
    backups = sorted(p for p in state_dir.iterdir() if p.name.startswith("before-migrate-"))
    assert len(backups) == 1
    pre = DispatchState.load(backups[0])
    assert "boiler" in pre.units
    # Operator-visible log line ("backed up to ...") fires on destructive edit.
    assert any("backed up to" in r.message for r in caplog.records)


def test_maybe_reload_param_only_change_does_not_retarget_or_backup(tmp_path):
    """A parameter-only change must not touch the state file or write a backup."""
    config_path = tmp_path / "plant_config.json"
    state_dir = tmp_path / "state"
    base = PlantConfig.legacy_default()
    _write_config(config_path, base)
    seed_ts = pd.Timestamp("2026-05-07T13:00:00Z")
    _seed_state_and_link(state_dir, _cs(seed_ts))
    original_target = (state_dir / "current.json").readlink()

    bumped = replace(base, gas_price_eur_mwh_hs=base.gas_price_eur_mwh_hs + 5.0)
    _write_config(config_path, bumped)
    _bump_mtime(config_path)

    reload_state = daemon._ConfigReloadState()
    result = daemon._maybe_reload_plant_config(
        config_path, base, reload_state,
        state_path=state_dir / "current.json",
        state_dir=state_dir,
    )

    assert result.gas_price_eur_mwh_hs == bumped.gas_price_eur_mwh_hs
    # Symlink target unchanged.
    assert (state_dir / "current.json").readlink() == original_target
    # No migrate-* or before-migrate-* files created.
    assert not any(
        p.name.startswith("migrate-") or p.name.startswith("before-migrate-")
        for p in state_dir.iterdir()
    )


def test_maybe_reload_rejects_when_migrated_state_infeasible(tmp_path, caplog):
    """Param change on a kept storage that strands its SoC after migration:
    feasibility check on the migrated state rejects, no swap, no backup."""
    config_path = tmp_path / "plant_config.json"
    state_dir = tmp_path / "state"
    base = PlantConfig.legacy_default()
    _write_config(config_path, base)
    seed_ts = pd.Timestamp("2026-05-07T13:00:00Z")
    seed_state = _cs(seed_ts)
    # Shift SoC well below the proposed new floor so feasible_against trips.
    seed_state.storages["storage"].soc_mwh_th = 60.0
    _seed_state_and_link(state_dir, seed_state)

    candidate = replace(
        base,
        storages=(replace(base.storages[0], floor_mwh_th=80.0),),
    )
    _write_config(config_path, candidate)
    _bump_mtime(config_path)

    reload_state = daemon._ConfigReloadState()
    with caplog.at_level("ERROR", logger="optimization.daemon"):
        result = daemon._maybe_reload_plant_config(
            config_path, base, reload_state,
            state_path=state_dir / "current.json",
            state_dir=state_dir,
        )

    assert result is base
    msg = "\n".join(r.message for r in caplog.records)
    assert "infeasible" in msg
    assert not any(p.name.startswith("before-migrate-") for p in state_dir.iterdir())


def test_maybe_reload_disable_toggle_does_not_touch_state(tmp_path):
    """Disable is a parameter-level change for the daemon: no add/remove,
    no state migration, no backup — same code path as any other param edit."""
    config_path = tmp_path / "plant_config.json"
    state_dir = tmp_path / "state"
    base = PlantConfig.legacy_default()
    _write_config(config_path, base)
    seed_ts = pd.Timestamp("2026-05-07T13:00:00Z")
    _seed_state_and_link(state_dir, _cs(seed_ts))
    original_target = (state_dir / "current.json").readlink()

    toggled = replace(base, disabled_asset_ids=("hp",))
    _write_config(config_path, toggled)
    _bump_mtime(config_path)

    reload_state = daemon._ConfigReloadState()
    result = daemon._maybe_reload_plant_config(
        config_path, base, reload_state,
        state_path=state_dir / "current.json",
        state_dir=state_dir,
    )

    assert result.disabled_asset_ids == ("hp",)
    assert (state_dir / "current.json").readlink() == original_target


def test_maybe_reload_mixed_add_remove_param_writes_backup_once(tmp_path, caplog):
    """End-to-end: an edit that adds an HP, removes a boiler, changes a param
    on a kept asset and bumps a global. The reload must apply everything in
    one go, write a single backup, retarget the symlink to a migrated state
    containing the right unit set, and log a structured diff summary."""
    config_path = tmp_path / "plant_config.json"
    state_dir = tmp_path / "state"
    base = PlantConfig.legacy_default()
    _write_config(config_path, base)
    seed_ts = pd.Timestamp("2026-05-07T13:00:00Z")
    _seed_state_and_link(state_dir, _cs(seed_ts))

    mixed = replace(
        base,
        heat_pumps=base.heat_pumps + (
            HeatPumpParams(id="hp_new", p_el_min_mw=0.5, p_el_max_mw=4.0, cop=3.4),
        ),
        boilers=(),
        chps=(replace(base.chps[0], startup_cost_eur=800.0),),
        gas_price_eur_mwh_hs=base.gas_price_eur_mwh_hs + 5.0,
    )
    _write_config(config_path, mixed)
    _bump_mtime(config_path)

    reload_state = daemon._ConfigReloadState()
    with caplog.at_level("INFO", logger="optimization.daemon"):
        result = daemon._maybe_reload_plant_config(
            config_path, base, reload_state,
            state_path=state_dir / "current.json",
            state_dir=state_dir,
        )

    assert result == mixed
    loaded = DispatchState.load(state_dir / "current.json")
    assert set(loaded.units) == {"hp", "hp_new", "chp"}  # boiler dropped, hp_new added
    backups = [p for p in state_dir.iterdir() if p.name.startswith("before-migrate-")]
    assert len(backups) == 1
    msg = "\n".join(r.message for r in caplog.records)
    # Structured diff summary reflects all four changes.
    assert "added units ['hp_new']" in msg
    assert "removed units ['boiler']" in msg
    assert "param-changed units ['chp']" in msg
    assert "globals changed" in msg


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
