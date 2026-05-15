"""Tests for the non-state helpers in replay_loop: _tick idle behavior and
_cleanup_forecast_dir.

ReplayState walk-mechanics live in test_replay_state.py; this file covers
the side-effecting helpers that need a real filesystem and a stub model.
"""
from __future__ import annotations

import os
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
    ReplayState,
    _cleanup_forecast_dir,
    _tick,
)


CSV_END = pd.Timestamp("2025-08-15T12:00:00Z")


class _StubForecaster(BaseForecaster):
    """Records every predict() call so the test can assert it was NOT invoked."""

    def __init__(self) -> None:
        self.calls: List[pd.Timestamp] = []

    def fit(self, X, y=None):
        return self

    def predict(self, history: pd.DataFrame, solve_time: pd.Timestamp, horizon: int = 35) -> pd.Series:
        self.calls.append(solve_time)
        idx = pd.date_range(start=solve_time, periods=horizon, freq="h")
        return pd.Series([1.0] * horizon, index=idx, name="demand_mw_th")


def _make_cfg(tmp_path: Path) -> Config:
    forecast_dir = tmp_path / "forecast"
    return Config(
        csv_path=tmp_path / "missing.csv",
        forecast_dir=forecast_dir,
        heartbeat_path=forecast_dir / HEARTBEAT_NAME,
        model_name="stub",
        horizon_hours=4,
        lookback_months=1,
        tick_interval_s=3600,
    )


def _history_covering(end: pd.Timestamp, hours: int = 24 * 90) -> pd.DataFrame:
    """Spans 90 days back from ``end`` — covers a 1-month lookback comfortably."""
    idx = pd.date_range(end=end, periods=hours, freq="h", tz="UTC")
    return pd.DataFrame({"heat_demand_W": [1_000_000.0] * hours}, index=idx)


def test_tick_idle_bumps_heartbeat_writes_no_parquet(tmp_path):
    cfg = _make_cfg(tmp_path)
    cfg.forecast_dir.mkdir(parents=True)
    state = ReplayState(csv_end=CSV_END, lookback_months=1)
    state.virt_solve_time = CSV_END  # force exhaustion
    model = _StubForecaster()
    history = _history_covering(CSV_END)

    assert state.advance() is None  # sanity: state is exhausted
    state.virt_solve_time = CSV_END  # advance() does not advance once exhausted, but be explicit

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
    state = ReplayState(csv_end=CSV_END, lookback_months=1)
    model = _StubForecaster()
    history = _history_covering(CSV_END)

    _tick(cfg, history, model, state)

    parquets = list(cfg.forecast_dir.glob(f"*{FORECAST_FILENAME_SUFFIX}"))
    assert len(parquets) == 1
    assert len(model.calls) == 1
    assert cfg.heartbeat_path.exists()


def test_cleanup_removes_parquets_and_heartbeat(tmp_path):
    forecast_dir = tmp_path / "forecast"
    forecast_dir.mkdir()
    (forecast_dir / "2024-08-15T09-00-00Z.parquet").write_text("old")
    (forecast_dir / "2024-08-15T10-00-00Z.parquet").write_text("old")
    (forecast_dir / HEARTBEAT_NAME).write_text("")

    _cleanup_forecast_dir(forecast_dir)

    assert list(forecast_dir.iterdir()) == []


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
