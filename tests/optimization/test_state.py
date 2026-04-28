"""State persistence — round-trip + edge cases."""
from __future__ import annotations

import pandas as pd
import pytest

from optimization.state import TIS_LONG, DispatchState


def test_roundtrip(tmp_path):
    s = DispatchState(
        timestamp=pd.Timestamp("2026-01-08 14:00:00", tz="Europe/Berlin"),
        sto_soc_mwh_th=137.5,
        hp_on=1, boiler_on=1, boiler_time_in_state_steps=12,
        chp_on=0, chp_time_in_state_steps=TIS_LONG,
    )
    p = tmp_path / "state.json"
    s.save(p)
    loaded = DispatchState.load(p)
    assert loaded == s


def test_cold_start_default_soc():
    ts = pd.Timestamp("2026-01-01 00:00:00", tz="Europe/Berlin")
    s = DispatchState.cold_start(ts)
    assert s.sto_soc_mwh_th == 200.0
    assert s.hp_on == s.boiler_on == s.chp_on == 0
    assert s.boiler_time_in_state_steps == TIS_LONG
    assert s.chp_time_in_state_steps == TIS_LONG


def test_atomic_write_does_not_leave_tmp(tmp_path):
    s = DispatchState.cold_start(pd.Timestamp("2026-01-01 00:00:00", tz="Europe/Berlin"))
    p = tmp_path / "state.json"
    s.save(p)
    assert p.exists()
    assert not (tmp_path / "state.json.tmp").exists()


def test_load_rejects_naive_timestamp(tmp_path):
    """Naive (no-tz) timestamps would silently pick local-machine TZ — refuse."""
    p = tmp_path / "state.json"
    p.write_text(
        '{"timestamp":"2026-01-01T00:00:00","sto_soc_mwh_th":200.0,'
        '"hp_on":0,"boiler_on":0,"boiler_time_in_state_steps":999,'
        '"chp_on":0,"chp_time_in_state_steps":999}'
    )
    with pytest.raises(ValueError, match="tz-aware"):
        DispatchState.load(p)
