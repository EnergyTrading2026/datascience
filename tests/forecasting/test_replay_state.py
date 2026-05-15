"""Unit tests for ReplayState.

Covers the linear-no-wrap walk: first tick returns replay_start, each
advance() steps forward one hour, exhaustion when virt_solve_time reaches
csv_end, and idempotent None-return after exhaustion.
"""
from __future__ import annotations

import pandas as pd
import pytest

from forecasting.replay_loop import ReplayState


CSV_END = pd.Timestamp("2025-08-15T12:00:00Z")


def test_first_advance_returns_replay_start():
    state = ReplayState(csv_end=CSV_END, lookback_months=3)
    expected_start = CSV_END - pd.DateOffset(months=3)
    assert state.advance() == expected_start


def test_advance_steps_one_hour_forward():
    state = ReplayState(csv_end=CSV_END, lookback_months=1)
    first = state.advance()
    second = state.advance()
    third = state.advance()
    assert second - first == pd.Timedelta(hours=1)
    assert third - second == pd.Timedelta(hours=1)


def test_exhaustion_when_reaching_csv_end():
    # 2h lookback: replay_start = csv_end - 2 months. Walk a short window.
    csv_end = pd.Timestamp("2025-08-15T03:00:00Z")
    replay_start = csv_end - pd.DateOffset(months=1)
    state = ReplayState(csv_end=csv_end, lookback_months=1)
    # Manually fast-forward to one hour before csv_end.
    state.virt_solve_time = csv_end - pd.Timedelta(hours=1)
    last = state.advance()
    assert last == csv_end - pd.Timedelta(hours=1)
    assert state.advance() is None


def test_advance_returns_none_repeatedly_after_exhaustion():
    state = ReplayState(csv_end=CSV_END, lookback_months=1)
    state.virt_solve_time = CSV_END
    for _ in range(5):
        assert state.advance() is None


def test_exhausted_message_logged_once(caplog):
    state = ReplayState(csv_end=CSV_END, lookback_months=1)
    state.virt_solve_time = CSV_END
    with caplog.at_level("INFO", logger="forecasting.replay"):
        state.advance()
        state.advance()
        state.advance()
    exhausted_logs = [r for r in caplog.records if "exhausted" in r.message]
    assert len(exhausted_logs) == 1


def test_zero_lookback_rejected():
    with pytest.raises(ValueError, match="lookback_months must be positive"):
        ReplayState(csv_end=CSV_END, lookback_months=0)


def test_negative_lookback_rejected():
    with pytest.raises(ValueError, match="lookback_months must be positive"):
        ReplayState(csv_end=CSV_END, lookback_months=-1)
