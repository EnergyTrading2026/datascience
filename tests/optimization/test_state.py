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


# --------------------------------------------------------------------------- #
# migrate_to — produce a state aligned to a different PlantConfig. Used by
# the live add/remove reload path: kept assets pass through unchanged, removed
# assets are dropped, added assets get cold-start values.
# --------------------------------------------------------------------------- #


def _ts():
    return pd.Timestamp("2026-01-08 14:00:00", tz="Europe/Berlin")


def test_migrate_to_identity_returns_equal_state():
    """No structural change → migrated state equals original."""
    cfg = PlantConfig.legacy_default()
    s = DispatchState.cold_start(cfg, _ts())
    s.units["boiler"] = UnitState(on=1, time_in_state_steps=12)
    s.storages["storage"] = StorageState(soc_mwh_th=137.5)
    assert s.migrate_to(cfg) == s


def test_migrate_to_drops_removed_assets():
    from dataclasses import replace
    cfg = PlantConfig.legacy_default()
    s = DispatchState.cold_start(cfg, _ts())
    s.units["boiler"] = UnitState(on=1, time_in_state_steps=12)
    candidate = replace(cfg, boilers=())  # remove the boiler
    migrated = s.migrate_to(candidate)
    assert "boiler" not in migrated.units
    assert "hp" in migrated.units  # untouched
    assert "storage" in migrated.storages


def test_migrate_to_inits_added_unit_with_tis_long_off():
    """A new unit starts off with TIS_LONG so no min-up/down binds at t=1."""
    from dataclasses import replace
    from optimization.config import HeatPumpParams
    base = PlantConfig.legacy_default()
    s = DispatchState.cold_start(base, _ts())
    candidate = replace(
        base,
        heat_pumps=base.heat_pumps + (
            HeatPumpParams(id="hp_new", p_el_min_mw=0.5, p_el_max_mw=3.0, cop=3.4),
        ),
    )
    migrated = s.migrate_to(candidate)
    assert migrated.units["hp_new"] == UnitState(on=0, time_in_state_steps=TIS_LONG)
    # Existing entry untouched (regression guard against init_state-style reset).
    assert migrated.units["hp"] == s.units["hp"]


def test_migrate_to_inits_added_storage_from_candidate_soc_init():
    """Added storage starts at its candidate's ``soc_init_mwh_th``, not 0 or
    the legacy default — operator-facing contract."""
    from dataclasses import replace
    from optimization.config import StorageParams
    base = PlantConfig.legacy_default()
    s = DispatchState.cold_start(base, _ts())
    candidate = replace(
        base,
        storages=base.storages + (
            StorageParams(
                id="tank_b",
                capacity_mwh_th=80.0,
                floor_mwh_th=10.0,
                charge_max_mw_th=5.0,
                discharge_max_mw_th=5.0,
                loss_mwh_per_step=0.0,
                soc_init_mwh_th=42.5,
            ),
        ),
    )
    migrated = s.migrate_to(candidate)
    assert migrated.storages["tank_b"].soc_mwh_th == 42.5
    assert migrated.storages["storage"].soc_mwh_th == s.storages["storage"].soc_mwh_th


def test_migrate_to_preserves_timestamp():
    """Migration is a re-registration, not a time step."""
    from dataclasses import replace
    base = PlantConfig.legacy_default()
    s = DispatchState.cold_start(base, _ts())
    candidate = replace(base, chps=())
    migrated = s.migrate_to(candidate)
    assert migrated.timestamp == s.timestamp


def test_migrate_to_then_covers_passes():
    """The whole point: post-migration, the new state covers the candidate."""
    from dataclasses import replace
    from optimization.config import HeatPumpParams
    base = PlantConfig.legacy_default()
    s = DispatchState.cold_start(base, _ts())
    candidate = replace(
        base,
        boilers=(),
        heat_pumps=base.heat_pumps + (
            HeatPumpParams(id="hp_extra", p_el_min_mw=0.5, p_el_max_mw=4.0, cop=3.4),
        ),
    )
    migrated = s.migrate_to(candidate)
    migrated.covers(candidate)  # raises if mismatched
    assert set(migrated.units) == {"hp", "hp_extra", "chp"}
    assert set(migrated.storages) == {"storage"}


def test_migrate_to_is_id_only_does_not_inspect_params():
    """Sanity pin: even if the same id has different params in candidate,
    migrate_to keeps the existing state entry. Param-only changes don't
    perturb state by design — that's what makes hot reload safe."""
    from dataclasses import replace
    base = PlantConfig.legacy_default()
    s = DispatchState.cold_start(base, _ts())
    s.units["boiler"] = UnitState(on=1, time_in_state_steps=12)
    # Same boiler id, very different params — migrate_to should be a no-op.
    candidate = replace(
        base,
        boilers=(replace(base.boilers[0], q_max_mw_th=20.0, eff=0.85),),
    )
    migrated = s.migrate_to(candidate)
    assert migrated.units["boiler"] == UnitState(on=1, time_in_state_steps=12)
