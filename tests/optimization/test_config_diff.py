"""Tests for ``optimization.config_diff``.

Covers ``ConfigDiff.between`` and the ``asset_param_changes`` helper used
by the dry-run CLI. These are pure data-mechanics tests; the daemon-level
integration (reload path acting on the diff) lives in ``test_daemon.py``.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from optimization.config import (
    BoilerParams,
    CHPParams,
    HeatPumpParams,
    PlantConfig,
    StorageParams,
)
from optimization.config_diff import ConfigDiff, asset_param_changes


def _base() -> PlantConfig:
    return PlantConfig.legacy_default()


def test_identity_diff_is_noop():
    cfg = _base()
    d = ConfigDiff.between(cfg, cfg)
    assert d.is_noop
    assert not d.changes_asset_set
    assert not d.is_destructive
    assert not d.globals_changed


def test_disable_toggle_is_not_noop_but_does_not_change_asset_set():
    """Operator-facing contract: disable is a parameter-level edit that
    flows through live reload, but ConfigDiff must still surface it so the
    daemon can re-pin MILP variables."""
    cfg = _base()
    toggled = replace(cfg, disabled_asset_ids=("hp",))
    d = ConfigDiff.between(cfg, toggled)
    assert not d.is_noop
    assert not d.changes_asset_set
    assert d.disabled_added == frozenset({"hp"})
    assert d.disabled_removed == frozenset()


def test_disable_removed_when_id_re_enabled():
    cfg = replace(_base(), disabled_asset_ids=("hp", "boiler"))
    candidate = replace(_base(), disabled_asset_ids=("boiler",))
    d = ConfigDiff.between(cfg, candidate)
    assert d.disabled_removed == frozenset({"hp"})
    assert d.disabled_added == frozenset()


def test_added_unit_classified_correctly():
    cfg = _base()
    candidate = replace(
        cfg,
        heat_pumps=cfg.heat_pumps + (
            HeatPumpParams(id="hp2", p_el_min_mw=0.5, p_el_max_mw=4.0, cop=3.4),
        ),
    )
    d = ConfigDiff.between(cfg, candidate)
    assert d.added_unit_ids == frozenset({"hp2"})
    assert d.removed_unit_ids == frozenset()
    assert d.changes_asset_set
    assert not d.is_destructive


def test_removed_unit_classified_destructive():
    cfg = _base()
    candidate = replace(cfg, boilers=())
    d = ConfigDiff.between(cfg, candidate)
    assert d.removed_unit_ids == frozenset({"boiler"})
    assert d.is_destructive
    assert d.changes_asset_set


def test_param_change_on_kept_asset():
    cfg = _base()
    candidate = replace(
        cfg,
        heat_pumps=(replace(cfg.heat_pumps[0], cop=4.0),),
    )
    d = ConfigDiff.between(cfg, candidate)
    assert d.param_changed_unit_ids == frozenset({"hp"})
    assert d.added_unit_ids == frozenset()
    assert d.removed_unit_ids == frozenset()
    assert not d.changes_asset_set


def test_added_storage_with_init_carries_through():
    cfg = _base()
    candidate = replace(
        cfg,
        storages=cfg.storages + (
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
    d = ConfigDiff.between(cfg, candidate)
    assert d.added_storage_ids == frozenset({"tank_b"})
    assert d.removed_storage_ids == frozenset()


def test_param_change_on_storage():
    cfg = _base()
    candidate = replace(
        cfg,
        storages=(replace(cfg.storages[0], capacity_mwh_th=300.0),),
    )
    d = ConfigDiff.between(cfg, candidate)
    assert d.param_changed_storage_ids == frozenset({"storage"})


def test_globals_changed_flag_set_for_gas_price_delta():
    cfg = _base()
    candidate = replace(cfg, gas_price_eur_mwh_hs=cfg.gas_price_eur_mwh_hs + 1.0)
    d = ConfigDiff.between(cfg, candidate)
    assert d.globals_changed
    assert not d.changes_asset_set


def test_family_move_reported_as_remove_plus_add():
    """Same string id in a different family is *not* a param change — per-family
    state schemas differ, and the daemon must drop+re-init."""
    cfg = _base()
    candidate = replace(
        cfg,
        boilers=(),
        chps=cfg.chps + (CHPParams(
            id="boiler",
            p_el_min_mw=1.0, p_el_max_mw=4.0,
            eff_el=0.4, eff_th=0.4,
            min_up_steps=4, min_down_steps=4,
            startup_cost_eur=100.0,
        ),),
    )
    d = ConfigDiff.between(cfg, candidate)
    assert "boiler" in d.removed_unit_ids
    assert "boiler" in d.added_unit_ids
    assert "boiler" not in d.param_changed_unit_ids


def test_mixed_add_remove_param_change_diff_captures_all_three():
    cfg = _base()
    candidate = replace(
        cfg,
        heat_pumps=(
            replace(cfg.heat_pumps[0], cop=4.1),  # param change on kept
            HeatPumpParams(id="hp2", p_el_min_mw=0.5, p_el_max_mw=4.0, cop=3.4),  # add
        ),
        boilers=(),                                                                # remove
        gas_price_eur_mwh_hs=cfg.gas_price_eur_mwh_hs + 5.0,                       # globals
        disabled_asset_ids=("chp",),                                               # disable
    )
    d = ConfigDiff.between(cfg, candidate)
    assert d.added_unit_ids == frozenset({"hp2"})
    assert d.removed_unit_ids == frozenset({"boiler"})
    assert d.param_changed_unit_ids == frozenset({"hp"})
    assert d.disabled_added == frozenset({"chp"})
    assert d.globals_changed
    assert d.is_destructive
    assert d.changes_asset_set


def test_asset_param_changes_lists_only_differing_fields():
    cfg = _base()
    candidate = replace(
        cfg,
        heat_pumps=(replace(cfg.heat_pumps[0], cop=4.0, p_el_max_mw=10.0),),
    )
    changes = asset_param_changes(cfg, candidate, "hp")
    assert set(changes) == {"cop", "p_el_max_mw"}
    assert changes["cop"] == (3.5, 4.0)
    assert changes["p_el_max_mw"] == (8.0, 10.0)


def test_asset_param_changes_for_storage():
    cfg = _base()
    candidate = replace(
        cfg,
        storages=(replace(cfg.storages[0], floor_mwh_th=75.0),),
    )
    changes = asset_param_changes(cfg, candidate, "storage")
    assert changes == {"floor_mwh_th": (50.0, 75.0)}


def test_asset_param_changes_empty_for_unknown_id():
    cfg = _base()
    assert asset_param_changes(cfg, cfg, "does-not-exist") == {}


def test_asset_param_changes_empty_when_id_moved_families():
    """Family move => no in-place param diff (the caller already has the
    full asset entries from ConfigDiff's added/removed sets)."""
    cfg = _base()
    candidate = replace(
        cfg,
        boilers=(),
        chps=cfg.chps + (CHPParams(
            id="boiler",
            p_el_min_mw=1.0, p_el_max_mw=4.0,
            eff_el=0.4, eff_th=0.4,
            min_up_steps=4, min_down_steps=4,
            startup_cost_eur=100.0,
        ),),
    )
    assert asset_param_changes(cfg, candidate, "boiler") == {}
