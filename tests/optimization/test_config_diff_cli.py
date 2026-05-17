"""Tests for the ``optimization-config-diff`` dry-run CLI.

The CLI shares its checks with the daemon's reload path (same validator,
same ConfigDiff, same migrate_to + feasible_against), so these tests focus
on the *user-facing surface*: argv parsing, exit codes, and the rendered
output. The underlying logic is covered in ``test_config_validation.py``,
``test_config_diff.py``, ``test_state.py`` and ``test_daemon.py``.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from optimization.config import (
    HeatPumpParams,
    PlantConfig,
    StorageParams,
)
from optimization.config_diff_cli import main
from optimization.state import DispatchState


# --------------------------------------------------------------------------- #
# Helpers — mirror the daemon test helpers; not shared via conftest to keep
# this file self-contained for newcomers reading just the CLI tests.
# --------------------------------------------------------------------------- #


def _write_cfg(path: Path, cfg: PlantConfig) -> None:
    path.write_text(json.dumps(cfg.to_dict(), indent=2))


def _seed_state(path: Path, cfg: PlantConfig | None = None) -> DispatchState:
    cfg = cfg or PlantConfig.legacy_default()
    state = DispatchState.cold_start(
        cfg, pd.Timestamp("2026-05-07T13:00:00Z"),
    )
    state.save(path)
    return state


def _run(argv, capsys):
    rc = main(argv)
    out = capsys.readouterr()
    return rc, out.out, out.err


# --------------------------------------------------------------------------- #
# Exit code semantics
# --------------------------------------------------------------------------- #


def test_noop_diff_exits_zero(tmp_path, capsys):
    """Identical proposed and current configs: exit 0 with a no-op message."""
    cfg = PlantConfig.legacy_default()
    p = tmp_path / "proposed.json"
    c = tmp_path / "current.json"
    _write_cfg(p, cfg)
    _write_cfg(c, cfg)
    rc, stdout, _ = _run(["--proposed", str(p), "--current", str(c)], capsys)
    assert rc == 0
    assert "no observable change" in stdout


def test_apply_able_add_exits_zero(tmp_path, capsys):
    """Adding an HP is apply-able and the dry-run says so."""
    base = PlantConfig.legacy_default()
    grown = replace(
        base,
        heat_pumps=base.heat_pumps + (
            HeatPumpParams(id="hp_new", p_el_min_mw=0.5, p_el_max_mw=4.0, cop=3.4),
        ),
    )
    p, c, s = (tmp_path / "p.json", tmp_path / "c.json", tmp_path / "state.json")
    _write_cfg(p, grown)
    _write_cfg(c, base)
    _seed_state(s)
    rc, stdout, _ = _run(
        ["--proposed", str(p), "--current", str(c), "--state", str(s)],
        capsys,
    )
    assert rc == 0
    assert "RESULT: apply-able" in stdout
    assert "'hp_new'" in stdout
    assert "init:    unit 'hp_new'" in stdout
    assert "feasibility under proposed: OK" in stdout


def test_validation_errors_exit_one(tmp_path, capsys):
    """A proposed file with hard errors must exit 1 and surface every error."""
    cfg = PlantConfig.legacy_default()
    bad = cfg.to_dict()
    bad["heat_pumps"][0]["p_el_min_mw"] = 99.0  # MIN_EXCEEDS_MAX (max=8)
    bad["co2_price_eur_per_t"] = -1.0           # NEGATIVE
    p = tmp_path / "proposed.json"
    p.write_text(json.dumps(bad))
    rc, stdout, _ = _run(["--proposed", str(p)], capsys)
    assert rc == 1
    assert "MIN_EXCEEDS_MAX" in stdout
    assert "NEGATIVE" in stdout
    assert "cannot apply" in stdout


def test_family_move_blocks_apply(tmp_path, capsys):
    """Same id moved between families: dry-run flags REJECT in the migration
    block and exits 1, mirroring what the daemon would do."""
    from optimization.config import CHPParams
    base = PlantConfig.legacy_default()
    moved = replace(
        base,
        boilers=(),
        chps=base.chps + (CHPParams(
            id="boiler",
            p_el_min_mw=1.0, p_el_max_mw=4.0,
            eff_el=0.4, eff_th=0.4,
            min_up_steps=4, min_down_steps=4,
            startup_cost_eur=100.0,
        ),),
    )
    p, c, s = (tmp_path / "p.json", tmp_path / "c.json", tmp_path / "state.json")
    _write_cfg(p, moved)
    _write_cfg(c, base)
    _seed_state(s)
    rc, stdout, _ = _run(
        ["--proposed", str(p), "--current", str(c), "--state", str(s)],
        capsys,
    )
    assert rc == 1
    assert "family-move check: REJECT" in stdout
    assert "'boiler'" in stdout


def test_infeasibility_blocks_apply(tmp_path, capsys):
    """SoC outside the proposed [floor, capacity] is flagged in the migration
    block and exits 1."""
    base = PlantConfig.legacy_default()
    candidate = replace(
        base,
        storages=(replace(base.storages[0], floor_mwh_th=180.0, soc_init_mwh_th=190.0),),
    )
    p, c, s = (tmp_path / "p.json", tmp_path / "c.json", tmp_path / "state.json")
    _write_cfg(p, candidate)
    _write_cfg(c, base)
    state = _seed_state(s)
    # Knock the SoC well below the new floor.
    state.storages["storage"].soc_mwh_th = 100.0
    state.save(s)
    rc, stdout, _ = _run(
        ["--proposed", str(p), "--current", str(c), "--state", str(s)],
        capsys,
    )
    assert rc == 1
    assert "feasibility under proposed: REJECT" in stdout
    assert "below new floor_mwh_th" in stdout


# --------------------------------------------------------------------------- #
# Output content
# --------------------------------------------------------------------------- #


def test_removed_asset_lists_existing_state_for_drop(tmp_path, capsys):
    """``drop:    unit 'boiler' (on=..., tis=...)`` so the operator sees
    exactly what state would be lost."""
    base = PlantConfig.legacy_default()
    shrunk = replace(base, boilers=())
    p, c, s = (tmp_path / "p.json", tmp_path / "c.json", tmp_path / "state.json")
    _write_cfg(p, shrunk)
    _write_cfg(c, base)
    state = _seed_state(s)
    # Bump boiler state so the line carries non-default values that prove
    # the CLI is reading the actual state file (not a placeholder).
    from optimization.state import UnitState
    state.units["boiler"] = UnitState(on=1, time_in_state_steps=14)
    state.save(s)
    rc, stdout, _ = _run(
        ["--proposed", str(p), "--current", str(c), "--state", str(s)],
        capsys,
    )
    assert rc == 0
    assert "drop:    unit 'boiler'" in stdout
    assert "on=1, tis=14" in stdout


def test_param_change_lists_per_field_old_new(tmp_path, capsys):
    """``~ changed heat_pump 'hp': cop: 3.5 → 4.0`` per modified field."""
    base = PlantConfig.legacy_default()
    bumped = replace(
        base,
        heat_pumps=(replace(base.heat_pumps[0], cop=4.0, p_el_max_mw=10.0),),
    )
    p, c = (tmp_path / "p.json", tmp_path / "c.json")
    _write_cfg(p, bumped)
    _write_cfg(c, base)
    rc, stdout, _ = _run(["--proposed", str(p), "--current", str(c)], capsys)
    assert rc == 0
    assert "~ changed heat_pump 'hp'" in stdout
    assert "cop: 3.5 → 4.0" in stdout
    assert "p_el_max_mw: 8.0 → 10.0" in stdout


def test_disable_toggle_lines_appear(tmp_path, capsys):
    base = PlantConfig.legacy_default()
    toggled = replace(base, disabled_asset_ids=("hp",))
    p, c = (tmp_path / "p.json", tmp_path / "c.json")
    _write_cfg(p, toggled)
    _write_cfg(c, base)
    rc, stdout, _ = _run(["--proposed", str(p), "--current", str(c)], capsys)
    assert rc == 0
    assert "disabled: + 'hp'" in stdout
    # Pure disable toggle without state argument: no feasibility section,
    # but the diff itself is the load-bearing line.


def test_warnings_surface_without_blocking(tmp_path, capsys):
    """A config with warnings only is apply-able; warnings are still printed."""
    base = PlantConfig.legacy_default()
    with_warning = replace(
        base,
        storages=(replace(base.storages[0], discharge_max_mw_th=0.0),),
    )
    p, c = (tmp_path / "p.json", tmp_path / "c.json")
    _write_cfg(p, with_warning)
    _write_cfg(c, base)
    rc, stdout, _ = _run(["--proposed", str(p), "--current", str(c)], capsys)
    assert rc == 0
    assert "STORAGE_DISCHARGE_DISABLED" in stdout
    assert "RESULT: apply-able" in stdout


def test_unreadable_proposed_exits_one(tmp_path, capsys):
    p = tmp_path / "does-not-exist.json"
    rc, _, stderr = _run(["--proposed", str(p)], capsys)
    assert rc == 1
    assert "could not read --proposed" in stderr


def test_invalid_json_exits_one(tmp_path, capsys):
    p = tmp_path / "broken.json"
    p.write_text("{not json")
    rc, _, stderr = _run(["--proposed", str(p)], capsys)
    assert rc == 1
    assert "could not read --proposed" in stderr


def test_missing_state_file_does_not_block_apply(tmp_path, capsys):
    """No --state argument: feasibility check is skipped, but the diff still
    apply-able when there's nothing else blocking it."""
    base = PlantConfig.legacy_default()
    p, c = (tmp_path / "p.json", tmp_path / "c.json")
    _write_cfg(p, replace(base, gas_price_eur_mwh_hs=40.0))
    _write_cfg(c, base)
    rc, stdout, _ = _run(["--proposed", str(p), "--current", str(c)], capsys)
    assert rc == 0
    assert "feasibility check skipped" in stdout


def test_default_current_is_legacy_default(tmp_path, capsys):
    """Without --current, the CLI compares the proposed against
    PlantConfig.legacy_default(). Useful in legacy_default deployments."""
    base = PlantConfig.legacy_default()
    grown = replace(
        base,
        heat_pumps=base.heat_pumps + (
            HeatPumpParams(id="hp_x", p_el_min_mw=0.5, p_el_max_mw=4.0, cop=3.4),
        ),
    )
    p = tmp_path / "p.json"
    _write_cfg(p, grown)
    rc, stdout, _ = _run(["--proposed", str(p)], capsys)
    assert rc == 0
    assert "+ added heat_pump 'hp_x'" in stdout
