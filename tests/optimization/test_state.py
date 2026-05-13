"""State persistence — round-trip + edge cases."""
from __future__ import annotations

import pandas as pd
import pytest

from optimization.config import PlantConfig
from optimization.state import (
    SCHEMA_VERSION,
    TIS_LONG,
    DispatchState,
    StorageState,
    UnitState,
)


def test_roundtrip(tmp_path):
    s = DispatchState(
        timestamp=pd.Timestamp("2026-01-08 14:00:00", tz="Europe/Berlin"),
        units={
            "hp": UnitState(on=1, time_in_state_steps=TIS_LONG),
            "boiler": UnitState(on=1, time_in_state_steps=12),
            "chp": UnitState(on=0, time_in_state_steps=TIS_LONG),
        },
        storages={"storage": StorageState(soc_mwh_th=137.5)},
    )
    p = tmp_path / "state.json"
    s.save(p)
    loaded = DispatchState.load(p)
    assert loaded == s


def test_cold_start_default_soc():
    cfg = PlantConfig.legacy_default()
    ts = pd.Timestamp("2026-01-01 00:00:00", tz="Europe/Berlin")
    s = DispatchState.cold_start(cfg, ts)
    assert s.storages["storage"].soc_mwh_th == 200.0
    for unit in s.units.values():
        assert unit.on == 0
        assert unit.time_in_state_steps == TIS_LONG


def test_atomic_write_does_not_leave_tmp(tmp_path):
    cfg = PlantConfig.legacy_default()
    s = DispatchState.cold_start(cfg, pd.Timestamp("2026-01-01 00:00:00", tz="Europe/Berlin"))
    p = tmp_path / "state.json"
    s.save(p)
    assert p.exists()
    assert not (tmp_path / "state.json.tmp").exists()


def test_load_rejects_naive_timestamp(tmp_path):
    """Naive (no-tz) timestamps would silently pick local-machine TZ — refuse."""
    p = tmp_path / "state.json"
    p.write_text(
        '{"schema_version":2,"timestamp":"2026-01-01T00:00:00",'
        '"units":{"hp":{"on":0,"time_in_state_steps":999}},'
        '"storages":{"storage":{"soc_mwh_th":200.0}}}'
    )
    with pytest.raises(ValueError, match="tz-aware"):
        DispatchState.load(p)


def test_load_rejects_v1_flat_schema(tmp_path):
    """Old flat-field (v1) schema must fail loud with migration guidance."""
    p = tmp_path / "state.json"
    p.write_text(
        '{"timestamp":"2026-01-01T00:00:00+01:00","sto_soc_mwh_th":200.0,'
        '"hp_on":0,"boiler_on":0}'
    )
    with pytest.raises(ValueError, match="missing schema_version"):
        DispatchState.load(p)


def test_load_rejects_unknown_schema_version(tmp_path):
    """Future or unrecognized schema_version must fail loud."""
    p = tmp_path / "state.json"
    p.write_text(
        '{"schema_version":99,"timestamp":"2026-01-01T00:00:00+01:00",'
        '"units":{},"storages":{}}'
    )
    with pytest.raises(ValueError, match="schema_version=99"):
        DispatchState.load(p)


def test_save_writes_current_schema_version(tmp_path):
    cfg = PlantConfig.legacy_default()
    s = DispatchState.cold_start(cfg, pd.Timestamp("2026-01-01 00:00:00", tz="Europe/Berlin"))
    p = tmp_path / "state.json"
    s.save(p)
    import json
    payload = json.loads(p.read_text())
    assert payload["schema_version"] == SCHEMA_VERSION


def test_covers_detects_missing_assets():
    cfg = PlantConfig.legacy_default()
    s = DispatchState(
        timestamp=pd.Timestamp("2026-01-01 00:00:00", tz="Europe/Berlin"),
        units={},  # missing every unit
        storages={"storage": StorageState(soc_mwh_th=100.0)},
    )
    with pytest.raises(ValueError, match="missing unit state"):
        s.covers(cfg)
