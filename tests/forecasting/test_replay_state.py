"""Unit tests for ReplayState.

Covers the linear-no-wrap walk via peek()/commit(), exhaustion semantics,
and persistence-to-disk (write on commit, read on construction, fallback
on missing/corrupt/out-of-window state files).
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from forecasting.replay_loop import ReplayState


CSV_END = pd.Timestamp("2025-08-15T12:00:00Z")


# --- Walk mechanics ----------------------------------------------------------


def test_first_peek_returns_replay_start():
    state = ReplayState(csv_end=CSV_END, lookback_months=3)
    expected_start = CSV_END - pd.DateOffset(months=3)
    assert state.peek() == expected_start


def test_peek_does_not_mutate():
    state = ReplayState(csv_end=CSV_END, lookback_months=1)
    first = state.peek()
    second = state.peek()
    assert first == second


def test_commit_steps_one_hour_forward():
    state = ReplayState(csv_end=CSV_END, lookback_months=1)
    first = state.peek()
    state.commit()
    second = state.peek()
    state.commit()
    third = state.peek()
    assert second - first == pd.Timedelta(hours=1)
    assert third - second == pd.Timedelta(hours=1)


def test_exhaustion_when_reaching_csv_end():
    csv_end = pd.Timestamp("2025-08-15T03:00:00Z")
    state = ReplayState(csv_end=csv_end, lookback_months=1)
    state.virt_solve_time = csv_end - pd.Timedelta(hours=1)
    last = state.peek()
    assert last == csv_end - pd.Timedelta(hours=1)
    state.commit()
    assert state.peek() is None


def test_peek_returns_none_repeatedly_after_exhaustion():
    state = ReplayState(csv_end=CSV_END, lookback_months=1)
    state.virt_solve_time = CSV_END
    for _ in range(5):
        assert state.peek() is None


def test_commit_is_no_op_after_exhaustion():
    state = ReplayState(csv_end=CSV_END, lookback_months=1)
    state.virt_solve_time = CSV_END
    state.commit()
    state.commit()
    # Must never advance past csv_end — that would land in the invalid
    # range and break a subsequent resume.
    assert state.virt_solve_time == CSV_END


def test_exhausted_message_logged_once(caplog):
    state = ReplayState(csv_end=CSV_END, lookback_months=1)
    state.virt_solve_time = CSV_END
    with caplog.at_level("INFO", logger="forecasting.replay_loop"):
        state.peek()
        state.peek()
        state.peek()
    exhausted_logs = [r for r in caplog.records if "exhausted" in r.message]
    assert len(exhausted_logs) == 1


def test_zero_lookback_rejected():
    with pytest.raises(ValueError, match="lookback_months must be positive"):
        ReplayState(csv_end=CSV_END, lookback_months=0)


def test_negative_lookback_rejected():
    with pytest.raises(ValueError, match="lookback_months must be positive"):
        ReplayState(csv_end=CSV_END, lookback_months=-1)


# --- Persistence -------------------------------------------------------------


def test_commit_persists_to_disk(tmp_path):
    state_path = tmp_path / ".replay-state.json"
    state = ReplayState(csv_end=CSV_END, lookback_months=1, state_path=state_path)
    start = state.peek()
    state.commit()

    data = json.loads(state_path.read_text())
    persisted = pd.Timestamp(data["virt_solve_time"])
    assert persisted == start + pd.Timedelta(hours=1)


def test_resume_from_persisted_state(tmp_path):
    state_path = tmp_path / ".replay-state.json"
    s1 = ReplayState(csv_end=CSV_END, lookback_months=1, state_path=state_path)
    s1.commit()
    s1.commit()
    expected = s1.virt_solve_time

    # Fresh ReplayState reads the same file and resumes there.
    s2 = ReplayState(csv_end=CSV_END, lookback_months=1, state_path=state_path)
    assert s2.virt_solve_time == expected


def test_missing_state_file_falls_back_silently(tmp_path, caplog):
    state_path = tmp_path / ".replay-state.json"
    with caplog.at_level("WARNING", logger="forecasting.replay_loop"):
        state = ReplayState(csv_end=CSV_END, lookback_months=1, state_path=state_path)
    assert state.virt_solve_time == CSV_END - pd.DateOffset(months=1)
    # Missing-file is the normal first-boot path; must not warn.
    assert not [r for r in caplog.records if r.levelname == "WARNING"]


def test_corrupt_state_file_falls_back_to_replay_start(tmp_path, caplog):
    state_path = tmp_path / ".replay-state.json"
    state_path.write_text("{not valid json")

    with caplog.at_level("WARNING", logger="forecasting.replay_loop"):
        state = ReplayState(csv_end=CSV_END, lookback_months=1, state_path=state_path)

    assert state.virt_solve_time == CSV_END - pd.DateOffset(months=1)
    assert any("could not read persisted" in r.message for r in caplog.records)


def test_missing_key_in_state_file_falls_back(tmp_path, caplog):
    state_path = tmp_path / ".replay-state.json"
    state_path.write_text(json.dumps({"unexpected_key": "value"}))

    with caplog.at_level("WARNING", logger="forecasting.replay_loop"):
        state = ReplayState(csv_end=CSV_END, lookback_months=1, state_path=state_path)

    assert state.virt_solve_time == CSV_END - pd.DateOffset(months=1)
    assert any("could not read persisted" in r.message for r in caplog.records)


def test_before_window_persisted_state_rejected(tmp_path, caplog):
    state_path = tmp_path / ".replay-state.json"
    state_path.write_text(json.dumps({"virt_solve_time": "2020-01-01T00:00:00Z"}))

    with caplog.at_level("WARNING", logger="forecasting.replay_loop"):
        state = ReplayState(csv_end=CSV_END, lookback_months=1, state_path=state_path)

    assert state.virt_solve_time == CSV_END - pd.DateOffset(months=1)
    assert any("outside replay window" in r.message for r in caplog.records)


def test_after_window_persisted_state_rejected(tmp_path, caplog):
    state_path = tmp_path / ".replay-state.json"
    future = (CSV_END + pd.Timedelta(hours=1)).isoformat()
    state_path.write_text(json.dumps({"virt_solve_time": future}))

    with caplog.at_level("WARNING", logger="forecasting.replay_loop"):
        state = ReplayState(csv_end=CSV_END, lookback_months=1, state_path=state_path)

    assert state.virt_solve_time == CSV_END - pd.DateOffset(months=1)
    assert any("outside replay window" in r.message for r in caplog.records)


def test_persisted_state_at_csv_end_resumes_exhausted(tmp_path):
    """virt_solve_time == csv_end is the terminal state; resume should idle."""
    state_path = tmp_path / ".replay-state.json"
    state_path.write_text(json.dumps({"virt_solve_time": CSV_END.isoformat()}))

    state = ReplayState(csv_end=CSV_END, lookback_months=1, state_path=state_path)
    assert state.virt_solve_time == CSV_END
    assert state.peek() is None


def test_persist_is_atomic_no_partial_file(tmp_path):
    """The temp file used by atomic rename must not survive a normal commit."""
    state_path = tmp_path / ".replay-state.json"
    state = ReplayState(csv_end=CSV_END, lookback_months=1, state_path=state_path)
    state.commit()

    siblings = sorted(p.name for p in tmp_path.iterdir())
    assert siblings == [".replay-state.json"]
