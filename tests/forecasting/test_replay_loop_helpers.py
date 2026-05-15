"""Tests for the non-state helpers in replay_loop: _tick, _build_model,
and _cleanup_forecast_dir.

ReplayState walk-mechanics live in test_replay_state.py; this file covers
the side-effecting helpers that need a real filesystem and a stub model.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List

import pandas as pd
import pytest

from forecasting.base_model import BaseForecaster
from forecasting.replay_loop import (
    Config,
    FORECAST_FILENAME_SUFFIX,
    HEARTBEAT_NAME,
    REPLAY_STATE_NAME,
    ReplayState,
    _build_model,
    _cleanup_forecast_dir,
    _tick,
)


CSV_END = pd.Timestamp("2025-08-15T12:00:00Z")


class _StubForecaster(BaseForecaster):
    """Records every predict() call so tests can assert on invocations."""

    def __init__(self) -> None:
        self.calls: List[pd.Timestamp] = []

    def fit(self, X, y=None):
        return self

    def predict(self, history: pd.DataFrame, solve_time: pd.Timestamp, horizon: int = 35) -> pd.Series:
        self.calls.append(solve_time)
        idx = pd.date_range(start=solve_time, periods=horizon, freq="h")
        return pd.Series([1.0] * horizon, index=idx, name="demand_mw_th")


class _RaisingForecaster(BaseForecaster):
    """Predict always blows up — used to test state-rollback on failure."""

    def fit(self, X, y=None):
        return self

    def predict(self, history: pd.DataFrame, solve_time: pd.Timestamp, horizon: int = 35) -> pd.Series:
        raise RuntimeError("simulated predict failure")


def _make_cfg(tmp_path: Path) -> Config:
    forecast_dir = tmp_path / "forecast"
    return Config(
        csv_path=tmp_path / "missing.csv",
        forecast_dir=forecast_dir,
        heartbeat_path=forecast_dir / HEARTBEAT_NAME,
        replay_state_path=forecast_dir / REPLAY_STATE_NAME,
        model_name="stub",
        horizon_hours=4,
        lookback_months=1,
        tick_interval_s=3600,
    )


def _history_covering(end: pd.Timestamp, hours: int = 24 * 90) -> pd.DataFrame:
    """Spans 90 days back from ``end`` — covers a 1-month lookback comfortably."""
    idx = pd.date_range(end=end, periods=hours, freq="h", tz="UTC")
    return pd.DataFrame({"heat_demand_W": [1_000_000.0] * hours}, index=idx)


# --- _tick -------------------------------------------------------------------


def test_tick_idle_bumps_heartbeat_writes_no_parquet(tmp_path):
    cfg = _make_cfg(tmp_path)
    cfg.forecast_dir.mkdir(parents=True)
    state = ReplayState(csv_end=CSV_END, lookback_months=1)
    state.virt_solve_time = CSV_END  # force exhaustion
    model = _StubForecaster()
    history = _history_covering(CSV_END)

    assert state.peek() is None  # sanity: state is exhausted

    _tick(cfg, history, model, state)

    # No parquet was written.
    assert not list(cfg.forecast_dir.glob(f"*{FORECAST_FILENAME_SUFFIX}"))
    # Forecaster was NOT called.
    assert model.calls == []
    # Heartbeat exists and is fresh.
    assert cfg.heartbeat_path.exists()
    assert time.time() - cfg.heartbeat_path.stat().st_mtime < 5


def test_tick_active_writes_parquet_and_bumps_heartbeat(tmp_path):
    cfg = _make_cfg(tmp_path)
    cfg.forecast_dir.mkdir(parents=True)
    state = ReplayState(
        csv_end=CSV_END, lookback_months=1, state_path=cfg.replay_state_path,
    )
    model = _StubForecaster()
    history = _history_covering(CSV_END)

    initial = state.virt_solve_time
    _tick(cfg, history, model, state)

    parquets = list(cfg.forecast_dir.glob(f"*{FORECAST_FILENAME_SUFFIX}"))
    assert len(parquets) == 1
    assert len(model.calls) == 1
    assert cfg.heartbeat_path.exists()
    # State advanced exactly one hour after the successful tick.
    assert state.virt_solve_time == initial + pd.Timedelta(hours=1)
    # State was persisted to disk.
    assert cfg.replay_state_path.exists()
    persisted = pd.Timestamp(json.loads(cfg.replay_state_path.read_text())["virt_solve_time"])
    assert persisted == state.virt_solve_time


def test_tick_does_not_advance_state_on_predict_failure(tmp_path):
    """Atomicity guarantee: if predict() raises, virt_solve_time stays put.

    Without this, a transient model failure would silently drop one forecast
    (the next tick would pull the *next* solve_time).
    """
    cfg = _make_cfg(tmp_path)
    cfg.forecast_dir.mkdir(parents=True)
    state = ReplayState(
        csv_end=CSV_END, lookback_months=1, state_path=cfg.replay_state_path,
    )
    initial = state.virt_solve_time
    model = _RaisingForecaster()
    history = _history_covering(CSV_END)

    with pytest.raises(RuntimeError):
        _tick(cfg, history, model, state)

    # State did NOT advance — the same solve_time gets retried next tick.
    assert state.virt_solve_time == initial
    # No parquet was written.
    assert not list(cfg.forecast_dir.glob(f"*{FORECAST_FILENAME_SUFFIX}"))
    # No state file was written either (commit never reached).
    assert not cfg.replay_state_path.exists()


def test_tick_advances_past_unusable_solve_time_with_empty_history(tmp_path):
    """If a slot has no usable history, advance past it — otherwise we'd
    spin forever on the same broken solve_time."""
    cfg = _make_cfg(tmp_path)
    cfg.forecast_dir.mkdir(parents=True)
    state = ReplayState(csv_end=CSV_END, lookback_months=1)
    initial = state.virt_solve_time
    model = _StubForecaster()
    # History entirely AFTER the current solve_time → cycle_history empty.
    history = _history_covering(CSV_END + pd.Timedelta(days=365))

    _tick(cfg, history, model, state)

    assert state.virt_solve_time == initial + pd.Timedelta(hours=1)
    assert model.calls == []
    assert not list(cfg.forecast_dir.glob(f"*{FORECAST_FILENAME_SUFFIX}"))


# --- _build_model -----------------------------------------------------------


def test_build_model_known_names():
    for name in ("daily_naive", "weekly_naive", "combined_seasonal"):
        model = _build_model(name)
        assert isinstance(model, BaseForecaster)


def test_build_model_unknown_name_lists_choices():
    with pytest.raises(ValueError, match="unknown MODEL"):
        _build_model("does_not_exist")
    with pytest.raises(ValueError, match="choices"):
        _build_model("does_not_exist")


# --- _cleanup_forecast_dir --------------------------------------------------


def test_cleanup_removes_parquets_and_heartbeat(tmp_path):
    forecast_dir = tmp_path / "forecast"
    forecast_dir.mkdir()
    (forecast_dir / "2024-08-15T09-00-00Z.parquet").write_text("old")
    (forecast_dir / "2024-08-15T10-00-00Z.parquet").write_text("old")
    (forecast_dir / HEARTBEAT_NAME).write_text("")

    _cleanup_forecast_dir(forecast_dir)

    assert list(forecast_dir.iterdir()) == []


def test_cleanup_preserves_replay_state_file(tmp_path):
    """The replay-state file enables resume-after-restart; cleanup must
    leave it alone, otherwise the container would always restart from
    replay_start."""
    forecast_dir = tmp_path / "forecast"
    forecast_dir.mkdir()
    (forecast_dir / "2024-08-15T09-00-00Z.parquet").write_text("old")
    (forecast_dir / HEARTBEAT_NAME).write_text("")
    (forecast_dir / REPLAY_STATE_NAME).write_text(
        json.dumps({"virt_solve_time": "2024-08-15T10:00:00+00:00"})
    )

    _cleanup_forecast_dir(forecast_dir)

    survivors = sorted(p.name for p in forecast_dir.iterdir())
    assert survivors == [REPLAY_STATE_NAME]


def test_cleanup_leaves_unrelated_files(tmp_path):
    forecast_dir = tmp_path / "forecast"
    forecast_dir.mkdir()
    (forecast_dir / "2024-08-15T09-00-00Z.parquet").write_text("old")
    (forecast_dir / "README.md").write_text("keep me")
    (forecast_dir / "operator_notes.txt").write_text("keep me too")

    _cleanup_forecast_dir(forecast_dir)

    survivors = sorted(p.name for p in forecast_dir.iterdir())
    assert survivors == ["README.md", "operator_notes.txt"]


def test_cleanup_on_missing_dir_is_noop(tmp_path):
    missing = tmp_path / "does_not_exist"
    # Must not raise.
    _cleanup_forecast_dir(missing)
    assert not missing.exists()


def test_cleanup_leaves_subdirectories_alone(tmp_path):
    forecast_dir = tmp_path / "forecast"
    forecast_dir.mkdir()
    sub = forecast_dir / ".archive"
    sub.mkdir()
    (sub / "2024-08-15T09-00-00Z.parquet").write_text("nested")

    _cleanup_forecast_dir(forecast_dir)

    # Subdir + its content untouched (cleanup only touches files at top level).
    assert sub.is_dir()
    assert (sub / "2024-08-15T09-00-00Z.parquet").exists()
