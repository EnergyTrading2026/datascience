"""Multi-asset coverage for the modular plant config.

The regression pin (`test_regression_pin.py`) locks the 1-of-each legacy default.
This file exercises the *modular* code paths so the abstraction does not bit-rot:

  - 2 HPs with halved limits behave equivalent to 1 HP at full limits (sanity)
  - zero-CHP plant builds and solves
  - duplicate asset ids fail at config construction
  - state/config mismatch raises with a useful message
  - PlantConfig JSON round-trip preserves equality
  - init_state seed mode honors --config-file

If the user later configures a real multi-asset plant via plant_config.json,
these tests are what gives confidence that nothing silently breaks.
"""
from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from optimization import init_state
from optimization.config import (
    BoilerParams,
    CHPParams,
    HeatPumpParams,
    PlantConfig,
    StorageParams,
)
from optimization.dispatch import extract_dispatch
from optimization.model import build_model
from optimization.solve import solve
from optimization.state import (
    TIS_LONG,
    DispatchState,
    UnitState,
)


# ---------------------- Multi-asset MILP equivalence ----------------------

def _two_hp_no_floor_config() -> PlantConfig:
    """Legacy default but with two HPs (halved limits) and no storage floor.
    No-floor lets the storage absorb / release freely so the only HP-related
    decision is total p_el; with two HPs the solver is free to split, and the
    minimum total cost should match the 1-HP case at the prod MIP gap.
    """
    base = PlantConfig.legacy_default()
    storages = tuple(replace(s, floor_mwh_th=0.0) for s in base.storages)
    return replace(
        base,
        heat_pumps=(
            HeatPumpParams(id="hp_a", p_el_min_mw=0.5, p_el_max_mw=4.0, cop=3.5),
            HeatPumpParams(id="hp_b", p_el_min_mw=0.5, p_el_max_mw=4.0, cop=3.5),
        ),
        storages=storages,
    )


def _one_hp_no_floor_config() -> PlantConfig:
    base = PlantConfig.legacy_default()
    storages = tuple(replace(s, floor_mwh_th=0.0) for s in base.storages)
    return replace(
        base,
        heat_pumps=(
            HeatPumpParams(id="hp_solo", p_el_min_mw=1.0, p_el_max_mw=8.0, cop=3.5),
        ),
        storages=storages,
    )


@pytest.fixture
def constant_inputs():
    idx = pd.date_range("2026-01-01 00:00:00", periods=35, freq="1h", tz="Europe/Berlin")
    demand = pd.Series(10.0, index=idx, name="demand_mw_th")
    price = pd.Series(60.0, index=idx, name="price_eur_mwh")
    return demand, price


def test_two_hps_match_one_hp_total_objective(constant_inputs):
    """Two HPs with half the p_el_max each should hit the same total optimum.

    Symmetric assets + linear cost on P_el => splitting is cost-neutral.
    Allows ~1% drift for MIP-gap noise (different var ordering between configs).
    """
    demand, price = constant_inputs
    cfg2 = _two_hp_no_floor_config()
    cfg1 = _one_hp_no_floor_config()
    state2 = DispatchState.cold_start(cfg2, demand.index[0])
    state1 = DispatchState.cold_start(cfg1, demand.index[0])

    m2 = build_model(demand, price, state2, cfg2, demand_safety_factor=1.0)
    m1 = build_model(demand, price, state1, cfg1, demand_safety_factor=1.0)
    r2 = solve(m2, time_limit_s=30, mip_gap=0.005)
    r1 = solve(m1, time_limit_s=30, mip_gap=0.005)
    assert r2.feasible and r1.feasible
    assert r2.objective_eur == pytest.approx(r1.objective_eur, rel=0.01)


def test_two_hps_dispatch_sums_to_total_demand_assist(constant_inputs):
    """Sum of per-HP commit dispatch should be a valid plant-level number."""
    demand, price = constant_inputs
    cfg = _two_hp_no_floor_config()
    state = DispatchState.cold_start(cfg, demand.index[0])
    m = build_model(demand, price, state, cfg, demand_safety_factor=1.0)
    solve(m, time_limit_s=30, mip_gap=0.005)
    d = extract_dispatch(m, n_intervals=4, solve_time=demand.index[0])
    assert set(d.hp_p_el_mw) == {"hp_a", "hp_b"}
    for hp_id, vals in d.hp_p_el_mw.items():
        assert len(vals) == 4
        assert all(0 <= v <= 4.0 + 1e-6 for v in vals), (hp_id, vals)


def test_zero_chp_config_builds_and_solves(constant_inputs):
    """Plant without any CHP must still produce a feasible dispatch."""
    demand, price = constant_inputs
    base = PlantConfig.legacy_default()
    cfg = replace(base, chps=())
    state = DispatchState.cold_start(cfg, demand.index[0])
    assert "chp" not in state.units  # no CHP entry expected

    m = build_model(demand, price, state, cfg, demand_safety_factor=1.0)
    r = solve(m, time_limit_s=30, mip_gap=0.005)
    assert r.feasible
    d = extract_dispatch(m, n_intervals=4, solve_time=demand.index[0])
    assert d.chp_p_el_mw == {}


def test_zero_boiler_config_builds_and_solves(constant_inputs):
    """Plant without any boiler must still produce a feasible dispatch."""
    demand, price = constant_inputs
    base = PlantConfig.legacy_default()
    cfg = replace(base, boilers=())
    state = DispatchState.cold_start(cfg, demand.index[0])
    assert not any(b.id in state.units for b in base.boilers)

    m = build_model(demand, price, state, cfg, demand_safety_factor=1.0)
    r = solve(m, time_limit_s=30, mip_gap=0.005)
    assert r.feasible
    d = extract_dispatch(m, n_intervals=4, solve_time=demand.index[0])
    assert d.boiler_q_th_mw == {}


def test_zero_storage_config_builds_and_solves(constant_inputs):
    """Plant without any thermal storage must still produce a feasible dispatch.

    Without storage the demand has to be met at every step from generation;
    the model must be feasible as long as HP+boiler+CHP can cover demand.
    """
    demand, price = constant_inputs
    base = PlantConfig.legacy_default()
    cfg = replace(base, storages=())
    state = DispatchState.cold_start(cfg, demand.index[0])
    assert state.storages == {}

    m = build_model(demand, price, state, cfg, demand_safety_factor=1.0)
    r = solve(m, time_limit_s=30, mip_gap=0.005)
    assert r.feasible
    d = extract_dispatch(m, n_intervals=4, solve_time=demand.index[0])
    assert d.sto_charge_mw_th == {}
    assert d.sto_discharge_mw_th == {}
    assert d.soc_trajectory_mwh_th == {}


def test_single_family_only_builds_and_solves(constant_inputs):
    """A boiler-only plant (no HP, no CHP, no storage) must still solve.

    Edge case: only one heat producer family is present. Pyomo Sets for the
    other families are empty, energy balance reduces to one term.
    """
    demand, price = constant_inputs
    base = PlantConfig.legacy_default()
    cfg = replace(base, heat_pumps=(), chps=(), storages=())
    state = DispatchState.cold_start(cfg, demand.index[0])
    assert set(state.units) == {b.id for b in base.boilers}
    assert state.storages == {}

    m = build_model(demand, price, state, cfg, demand_safety_factor=1.0)
    r = solve(m, time_limit_s=30, mip_gap=0.005)
    assert r.feasible


def test_zero_hp_config_builds_and_solves(constant_inputs):
    """Plant without any heat pump must still produce a feasible dispatch.

    Symmetry with zero-CHP / zero-boiler / zero-storage: the modular code
    must not assume HPs exist. (The legacy plant always had one.)
    """
    demand, price = constant_inputs
    base = PlantConfig.legacy_default()
    cfg = replace(base, heat_pumps=())
    state = DispatchState.cold_start(cfg, demand.index[0])
    assert not any(hp.id in state.units for hp in base.heat_pumps)

    m = build_model(demand, price, state, cfg, demand_safety_factor=1.0)
    r = solve(m, time_limit_s=30, mip_gap=0.005)
    assert r.feasible
    d = extract_dispatch(m, n_intervals=4, solve_time=demand.index[0])
    assert d.hp_p_el_mw == {}


# ---------------------- Heterogeneous multi-asset correctness ----------------------
#
# These tests are the load-bearing proof that the modular MILP actually
# discriminates between non-identical assets in the same family. Previously
# the only multi-asset tests used identical halved copies — which would
# pass even if the model wired every asset to a single shared param
# (an easy refactor bug). The tests below force the solver to make a
# choice that depends on per-asset parameters.


def test_two_hps_with_different_cop_solver_prefers_efficient(constant_inputs):
    """Two HPs at the same electrical limits but different COPs.

    A perfectly efficient MILP, given electrical-cost-only objective, must
    saturate the more efficient HP before using the less efficient one
    whenever total demand can be met within p_el_max of the efficient HP.

    Storage is removed entirely (not just floor-relaxed) so demand must be
    met live each step — otherwise a half-full storage would simply drain
    and both HPs would sit at zero, making the assertion pass vacuously.
    """
    demand, price = constant_inputs
    base = PlantConfig.legacy_default()
    cfg = replace(
        base,
        heat_pumps=(
            HeatPumpParams(id="hp_eff", p_el_min_mw=0.0, p_el_max_mw=4.0, cop=4.5),
            HeatPumpParams(id="hp_inef", p_el_min_mw=0.0, p_el_max_mw=4.0, cop=2.5),
        ),
        boilers=(),  # remove cheaper boiler so HPs actually carry demand
        chps=(),
        storages=(),  # no buffer: HPs must meet 10 MW thermal every step
    )
    state = DispatchState.cold_start(cfg, demand.index[0])
    m = build_model(demand, price, state, cfg, demand_safety_factor=1.0)
    r = solve(m, time_limit_s=30, mip_gap=0.005)
    assert r.feasible
    d = extract_dispatch(m, n_intervals=4, solve_time=demand.index[0])
    eff_total = sum(d.hp_p_el_mw["hp_eff"])
    inef_total = sum(d.hp_p_el_mw["hp_inef"])
    # Sanity: HPs are the only producer, so the efficient one must be ON.
    # Without this, the >= assertion below could pass with both at zero.
    assert eff_total > 0, eff_total
    # Solver should prefer the cheap (high-COP) HP.
    assert eff_total >= inef_total - 1e-6, (eff_total, inef_total)


def test_heterogeneous_storages_each_respects_own_capacity(constant_inputs):
    """Two storages with different capacities and floors must each stay in
    their own bounds. Catches accidental param sharing across storages."""
    demand, price = constant_inputs
    base = PlantConfig.legacy_default()
    cfg = replace(
        base,
        storages=(
            StorageParams(
                id="big",
                capacity_mwh_th=300.0, floor_mwh_th=0.0,
                charge_max_mw_th=20.0, discharge_max_mw_th=20.0,
                loss_mwh_per_step=0.0, soc_init_mwh_th=150.0,
            ),
            StorageParams(
                id="small",
                capacity_mwh_th=40.0, floor_mwh_th=10.0,
                charge_max_mw_th=4.0, discharge_max_mw_th=4.0,
                loss_mwh_per_step=0.0, soc_init_mwh_th=30.0,
            ),
        ),
    )
    state = DispatchState.cold_start(cfg, demand.index[0])
    m = build_model(demand, price, state, cfg, demand_safety_factor=1.0)
    r = solve(m, time_limit_s=30, mip_gap=0.005)
    assert r.feasible
    d = extract_dispatch(m, n_intervals=4, solve_time=demand.index[0])
    # Per-asset SoC trajectories present, sized correctly.
    assert set(d.soc_trajectory_mwh_th) == {"big", "small"}
    big_traj = d.soc_trajectory_mwh_th["big"]
    small_traj = d.soc_trajectory_mwh_th["small"]
    # Big storage stays within (0, 300]; small stays within [10, 40].
    assert all(-1e-6 <= v <= 300 + 1e-6 for v in big_traj), big_traj
    assert all(10 - 1e-6 <= v <= 40 + 1e-6 for v in small_traj), small_traj
    # Small must not exceed its own charge/discharge limits per step.
    for v in d.sto_charge_mw_th["small"]:
        assert -1e-6 <= v <= 4.0 + 1e-6, v
    for v in d.sto_discharge_mw_th["small"]:
        assert -1e-6 <= v <= 4.0 + 1e-6, v


def test_two_chps_with_different_capacities_each_respects_own_bounds(constant_inputs):
    """Two CHPs with very different P_el limits. If a refactor accidentally
    shared params across CHPs, one of these bounds would leak to the other
    and the test would catch it (small CHP exceeding 1.5 MW or big CHP
    capped at 1.5 MW).

    Scenario forces both CHPs to operate at their electrical maxima:
      - HPs/boilers/storage removed so CHPs are the only heat source.
      - Demand 9 MW_th equals the sum of both CHPs' thermal output at
        max (small: 1.5*1.2 = 1.8; big: 6.0*1.2 = 7.2). The only feasible
        plan is both at max every step.
    Under those conditions, a leak of ``chp_big``'s p_el_max (6.0) onto
    ``chp_small`` would push small to 6.0 MW and trip the upper-bound
    assertion. The lower assertions ensure the bounds are actually being
    exercised — without them the test could pass with both CHPs at zero.
    """
    demand, _ = constant_inputs
    idx = demand.index
    tight_demand = pd.Series(9.0, index=idx, name="demand_mw_th")
    flat_price = pd.Series(60.0, index=idx, name="price_eur_mwh")
    base = PlantConfig.legacy_default()
    cfg = replace(
        base,
        heat_pumps=(),
        boilers=(),
        chps=(
            CHPParams(
                id="chp_small",
                p_el_min_mw=0.5, p_el_max_mw=1.5,
                eff_el=0.40, eff_th=0.48,
                min_up_steps=1, min_down_steps=1, startup_cost_eur=0.0,
            ),
            CHPParams(
                id="chp_big",
                p_el_min_mw=2.0, p_el_max_mw=6.0,
                eff_el=0.40, eff_th=0.48,
                min_up_steps=1, min_down_steps=1, startup_cost_eur=0.0,
            ),
        ),
        storages=(),
    )
    state = DispatchState.cold_start(cfg, idx[0])
    m = build_model(tight_demand, flat_price, state, cfg, demand_safety_factor=1.0)
    r = solve(m, time_limit_s=30, mip_gap=0.005)
    assert r.feasible
    d = extract_dispatch(m, n_intervals=4, solve_time=idx[0])
    # Both CHPs must actually be running near their respective maxes —
    # otherwise the upper-bound assertions below pass trivially.
    assert max(d.chp_p_el_mw["chp_small"]) > 1.5 - 0.05, d.chp_p_el_mw["chp_small"]
    assert max(d.chp_p_el_mw["chp_big"]) > 6.0 - 0.05, d.chp_p_el_mw["chp_big"]
    for v in d.chp_p_el_mw["chp_small"]:
        assert -1e-6 <= v <= 1.5 + 1e-6, ("chp_small leaked bound", v)
    for v in d.chp_p_el_mw["chp_big"]:
        assert -1e-6 <= v <= 6.0 + 1e-6, ("chp_big leaked bound", v)


# ---------------------- Scaling envelope ----------------------

@pytest.mark.slow
def test_scaling_envelope_5_hp_solves_within_budget(constant_inputs):
    """5 heat pumps + 1 boiler + 1 CHP + 2 storages must solve under the
    production time limit (30s).

    Documents the envelope frontend operators can rely on. Above this size,
    solve time is not guaranteed under the prod MIP gap. If this drifts,
    the operator UI must surface a warning before letting users push past
    these counts.
    """
    import time

    demand, price = constant_inputs
    base = PlantConfig.legacy_default()
    cfg = replace(
        base,
        heat_pumps=tuple(
            HeatPumpParams(
                id=f"hp_{i}",
                p_el_min_mw=0.0,
                p_el_max_mw=2.0,
                cop=3.0 + 0.2 * i,  # heterogeneous COPs to keep solver honest
            )
            for i in range(5)
        ),
        storages=base.storages + (
            StorageParams(
                id="tank2",
                capacity_mwh_th=120.0, floor_mwh_th=20.0,
                charge_max_mw_th=8.0, discharge_max_mw_th=8.0,
                loss_mwh_per_step=0.0, soc_init_mwh_th=80.0,
            ),
        ),
    )
    state = DispatchState.cold_start(cfg, demand.index[0])
    m = build_model(demand, price, state, cfg, demand_safety_factor=1.0)
    t0 = time.monotonic()
    r = solve(m, time_limit_s=30, mip_gap=0.005)
    elapsed = time.monotonic() - t0
    assert r.feasible, (elapsed, r)
    # Documented envelope: well below time limit on hardware comparable to CI.
    assert elapsed < 30, f"solve took {elapsed:.1f}s, exceeded 30s budget"


# ---------------------- Config validation ----------------------

def test_duplicate_asset_id_raises():
    """Cross-family ID uniqueness must be enforced in PlantConfig."""
    base = PlantConfig.legacy_default()
    with pytest.raises(ValueError, match="duplicate asset id"):
        # Add a boiler with the same id as the HP.
        replace(
            base,
            boilers=base.boilers + (
                BoilerParams(
                    id="hp",  # collides with heat pump
                    q_min_mw_th=2.0, q_max_mw_th=12.0, eff=0.97,
                    min_up_steps=4, min_down_steps=4,
                ),
            ),
        )


@pytest.mark.parametrize("bad_id", ["", " ", "   ", " hp", "hp ", " hp ", "\thp", "hp\n"])
def test_whitespace_or_empty_id_rejected(bad_id):
    """Whitespace-padded or empty asset ids must fail at construction.

    Reason: ' hp ' would silently fail to match 'hp' in --drop-asset CLI and in
    long-format dispatch parquet groupbys. Caught at the dataclass level so it
    can never reach state files or the solver.
    """
    with pytest.raises(ValueError, match="asset id"):
        HeatPumpParams(id=bad_id, p_el_min_mw=1.0, p_el_max_mw=8.0, cop=3.5)


def test_zero_heat_producer_config_raises():
    """A plant with no HP/boiler/CHP cannot meet demand — must fail loud."""
    with pytest.raises(ValueError, match="zero heat producers"):
        PlantConfig(
            dt_h=0.25,
            gas_price_eur_mwh_hs=35.0,
            co2_factor_t_per_mwh_hs=0.201,
            co2_price_eur_per_t=60.0,
            heat_pumps=(), boilers=(), chps=(),
            storages=(
                StorageParams(
                    id="s", capacity_mwh_th=200.0, floor_mwh_th=0.0,
                    charge_max_mw_th=15.0, discharge_max_mw_th=15.0,
                    loss_mwh_per_step=0.0, soc_init_mwh_th=100.0,
                ),
            ),
        )


# ---------------------- State/config drift error UX ----------------------

def test_covers_missing_asset_message_lists_missing_ids():
    cfg = PlantConfig.legacy_default()
    state = DispatchState(
        timestamp=pd.Timestamp("2026-01-01 00:00:00", tz="Europe/Berlin"),
        units={"hp": UnitState(on=0, time_in_state_steps=TIS_LONG)},
        # Missing boiler, chp, storage entries entirely.
        storages={},
    )
    with pytest.raises(ValueError) as exc:
        state.covers(cfg)
    msg = str(exc.value)
    assert "missing unit state" in msg
    assert "missing storage state" in msg
    assert "optimization-init-state" in msg


def test_covers_extra_asset_message_lists_extra_ids():
    cfg = PlantConfig.legacy_default()
    state = DispatchState.cold_start(cfg, pd.Timestamp("2026-01-01 00:00:00", tz="Europe/Berlin"))
    # Inject a phantom unit entry that the config doesn't know about.
    state.units["ghost_chp"] = UnitState(on=1, time_in_state_steps=4)
    with pytest.raises(ValueError) as exc:
        state.covers(cfg)
    msg = str(exc.value)
    assert "ghost_chp" in msg
    assert "unknown asset" in msg


# ---------------------- PlantConfig JSON round-trip ----------------------

def test_plant_config_to_dict_includes_schema_version():
    payload = PlantConfig.legacy_default().to_dict()
    assert payload["schema_version"] >= 1
    assert payload["heat_pumps"][0]["id"] == "hp"


def test_plant_config_json_roundtrip_equals_legacy_default(tmp_path):
    cfg = PlantConfig.legacy_default()
    p = tmp_path / "plant_config.json"
    cfg.to_json(p)
    cfg2 = PlantConfig.from_json(p)
    assert cfg2 == cfg


def test_from_dict_rejects_missing_schema_version():
    payload = PlantConfig.legacy_default().to_dict()
    payload.pop("schema_version")
    with pytest.raises(ValueError, match="missing schema_version"):
        PlantConfig.from_dict(payload)


def test_from_dict_rejects_unknown_field():
    payload = PlantConfig.legacy_default().to_dict()
    payload["heat_pumps"][0]["typo_field"] = 42
    with pytest.raises(ValueError, match="extra="):
        PlantConfig.from_dict(payload)


def test_from_dict_rejects_unknown_top_level_key():
    """A misspelled top-level key (e.g. ``heat_pumpz``) must fail loud, not
    silently load as zero heat pumps."""
    payload = PlantConfig.legacy_default().to_dict()
    payload["heat_pumpz"] = payload.pop("heat_pumps")
    with pytest.raises(ValueError, match="unknown top-level keys"):
        PlantConfig.from_dict(payload)


def test_from_dict_rejects_future_config_schema_version():
    """A config from a newer (incompatible) schema must fail loud, not silently
    fall back. Mirrors test_load_rejects_unknown_schema_version for state."""
    payload = PlantConfig.legacy_default().to_dict()
    payload["schema_version"] = 99
    with pytest.raises(ValueError, match="schema_version=99"):
        PlantConfig.from_dict(payload)


# ---------------------- init_state seed with --config-file ----------------------

def test_init_state_seed_with_config_file_matches_custom_plant(tmp_path):
    """Seed mode with --config-file must produce a state whose ids match the
    custom plant exactly — i.e. covers(config) passes immediately, no manual
    add/drop dance.
    """
    cfg = replace(
        PlantConfig.legacy_default(),
        heat_pumps=(
            HeatPumpParams(id="hp_a", p_el_min_mw=0.5, p_el_max_mw=4.0, cop=3.5),
            HeatPumpParams(id="hp_b", p_el_min_mw=0.5, p_el_max_mw=4.0, cop=3.5),
        ),
        chps=(),  # plant without any CHP
    )
    cfg_path = tmp_path / "plant_config.json"
    cfg.to_json(cfg_path)

    state_path = tmp_path / "state.json"
    rc = init_state.main([
        "--state-out", str(state_path),
        "--config-file", str(cfg_path),
        "--solve-time", "2026-01-01T00:00:00+00:00",
        "--force",
    ])
    assert rc == 0
    s = DispatchState.load(state_path)
    s.covers(cfg)  # must not raise
    assert set(s.units) == {"hp_a", "hp_b", "boiler"}
    assert "chp" not in s.units
    assert set(s.storages) == {"storage"}


def test_init_state_seed_with_config_file_respects_per_storage_soc_init(tmp_path):
    """Seed mode without --soc-mwh-th must take each storage's
    ``soc_init_mwh_th`` from the config, not silently overwrite all of them
    with the argparse default.
    """
    cfg = replace(
        PlantConfig.legacy_default(),
        storages=(
            StorageParams(
                id="tank_a",
                capacity_mwh_th=80.0,
                floor_mwh_th=0.0,
                charge_max_mw_th=10.0,
                discharge_max_mw_th=10.0,
                loss_mwh_per_step=0.0,
                soc_init_mwh_th=40.0,
            ),
            StorageParams(
                id="tank_b",
                capacity_mwh_th=300.0,
                floor_mwh_th=0.0,
                charge_max_mw_th=20.0,
                discharge_max_mw_th=20.0,
                loss_mwh_per_step=0.0,
                soc_init_mwh_th=12.5,
            ),
        ),
    )
    cfg_path = tmp_path / "plant_config.json"
    cfg.to_json(cfg_path)

    state_path = tmp_path / "state.json"
    rc = init_state.main([
        "--state-out", str(state_path),
        "--config-file", str(cfg_path),
        "--solve-time", "2026-01-01T00:00:00+00:00",
        "--force",
    ])
    assert rc == 0
    s = DispatchState.load(state_path)
    assert s.storages["tank_a"].soc_mwh_th == 40.0
    assert s.storages["tank_b"].soc_mwh_th == 12.5


def test_init_state_seed_explicit_soc_overrides_config(tmp_path):
    """Explicit --soc-mwh-th still overrides per-asset SoC (operator
    escape hatch for all storages at once)."""
    cfg = replace(
        PlantConfig.legacy_default(),
        storages=(
            StorageParams(
                id="tank_a",
                capacity_mwh_th=80.0,
                floor_mwh_th=0.0,
                charge_max_mw_th=10.0,
                discharge_max_mw_th=10.0,
                loss_mwh_per_step=0.0,
                soc_init_mwh_th=40.0,
            ),
        ),
    )
    cfg_path = tmp_path / "plant_config.json"
    cfg.to_json(cfg_path)

    state_path = tmp_path / "state.json"
    rc = init_state.main([
        "--state-out", str(state_path),
        "--config-file", str(cfg_path),
        "--soc-mwh-th", "60.0",
        "--solve-time", "2026-01-01T00:00:00+00:00",
        "--force",
    ])
    assert rc == 0
    s = DispatchState.load(state_path)
    assert s.storages["tank_a"].soc_mwh_th == 60.0


def test_init_state_seed_without_config_file_uses_legacy_default(tmp_path):
    """Without --config-file, seed mode falls back to legacy_default()."""
    state_path = tmp_path / "state.json"
    rc = init_state.main([
        "--state-out", str(state_path),
        "--solve-time", "2026-01-01T00:00:00+00:00",
        "--force",
    ])
    assert rc == 0
    s = DispatchState.load(state_path)
    s.covers(PlantConfig.legacy_default())
    assert set(s.units) == {"hp", "boiler", "chp"}


def test_init_state_seed_invalid_config_file_errors(tmp_path):
    """A malformed --config-file path must fail with a non-zero exit, not
    silently fall back to legacy_default."""
    state_path = tmp_path / "state.json"
    rc = init_state.main([
        "--state-out", str(state_path),
        "--config-file", str(tmp_path / "does_not_exist.json"),
        "--solve-time", "2026-01-01T00:00:00+00:00",
        "--force",
    ])
    assert rc == 1
    assert not state_path.exists()


# ---------------------- Disable: operator-forced off-switch ----------------------
#
# A disabled asset stays registered (state retained, id stable) but the MILP
# constrains it to zero output. Tests cover every family and the storage's
# frozen-SoC semantic; the validator-level coverage lives in
# test_config_validation.py.


def test_disabled_hp_yields_zero_p_el_in_dispatch(constant_inputs):
    """HP marked disabled produces zero electrical setpoint across the commit
    window — the binary z_hp is fixed to 0 and the bound constraints follow."""
    demand, price = constant_inputs
    base = PlantConfig.legacy_default()
    # Two HPs so disabling one still leaves a producer mix capable of meeting
    # demand without leaning entirely on boiler+CHP.
    cfg = replace(
        base,
        heat_pumps=(
            HeatPumpParams(id="hp_a", p_el_min_mw=0.5, p_el_max_mw=4.0, cop=3.5),
            HeatPumpParams(id="hp_b", p_el_min_mw=0.5, p_el_max_mw=4.0, cop=3.5),
        ),
        disabled_asset_ids=("hp_a",),
    )
    state = DispatchState.cold_start(cfg, demand.index[0])
    m = build_model(demand, price, state, cfg)
    r = solve(m, time_limit_s=30, mip_gap=0.005)
    assert r.feasible
    d = extract_dispatch(m, n_intervals=4, solve_time=demand.index[0])
    assert d.hp_p_el_mw["hp_a"] == [pytest.approx(0.0)] * 4
    # The enabled HP is free to be used; not asserted on, the solver may also
    # leave it idle — the contract here is only "disabled means zero".


def test_disabled_boiler_yields_zero_q_th_in_dispatch(constant_inputs):
    """Disabled boiler emits zero thermal output and remains z=0 — even when
    initial state has it on at TIS=0, which would normally lock z=1 for the
    first ``min_up_steps`` steps. Disable wins."""
    demand, price = constant_inputs
    base = PlantConfig.legacy_default()
    cfg = replace(base, disabled_asset_ids=("boiler",))
    # Seed state with boiler on at TIS=0: in the absence of disable this would
    # force the first min_up_steps steps to z=1. Disable must overrule.
    state = DispatchState.cold_start(cfg, demand.index[0])
    state.units["boiler"] = UnitState(on=1, time_in_state_steps=0)
    m = build_model(demand, price, state, cfg)
    r = solve(m, time_limit_s=30, mip_gap=0.005)
    assert r.feasible
    d = extract_dispatch(m, n_intervals=4, solve_time=demand.index[0])
    assert d.boiler_q_th_mw["boiler"] == [pytest.approx(0.0)] * 4


def test_disabled_chp_yields_zero_p_el_in_dispatch(constant_inputs):
    """Disabled CHP: zero electrical output across the commit window."""
    demand, price = constant_inputs
    cfg = replace(PlantConfig.legacy_default(), disabled_asset_ids=("chp",))
    state = DispatchState.cold_start(cfg, demand.index[0])
    m = build_model(demand, price, state, cfg)
    r = solve(m, time_limit_s=30, mip_gap=0.005)
    assert r.feasible
    d = extract_dispatch(m, n_intervals=4, solve_time=demand.index[0])
    assert d.chp_p_el_mw["chp"] == [pytest.approx(0.0)] * 4


def test_disabled_storage_charge_discharge_zero_and_soc_frozen(constant_inputs):
    """Disabled storage: charge and discharge are zero AND SoC stays at its
    initial value across the full horizon (no loss drift while paused)."""
    demand, price = constant_inputs
    cfg = replace(PlantConfig.legacy_default(), disabled_asset_ids=("storage",))
    state = DispatchState.cold_start(cfg, demand.index[0])
    initial_soc = state.storages["storage"].soc_mwh_th
    m = build_model(demand, price, state, cfg)
    r = solve(m, time_limit_s=30, mip_gap=0.005)
    assert r.feasible
    d = extract_dispatch(m, n_intervals=4, solve_time=demand.index[0])
    assert d.sto_charge_mw_th["storage"] == [pytest.approx(0.0)] * 4
    assert d.sto_discharge_mw_th["storage"] == [pytest.approx(0.0)] * 4
    # SoC trajectory has length n_intervals + 1 (initial + each step end).
    soc = d.soc_trajectory_mwh_th["storage"]
    assert len(soc) == 5
    assert all(v == pytest.approx(initial_soc) for v in soc), soc


def test_re_enable_after_disable_uses_state_as_if_paused(constant_inputs):
    """The whole point of disable being non-destructive: re-enabling an asset
    after a period off resumes from the same SoC, with no special operator
    action. Builds two models back-to-back: one with disable, one without,
    same seed state. The enabled run must be free to dispatch the storage."""
    demand, price = constant_inputs
    base = PlantConfig.legacy_default()
    cfg_off = replace(base, disabled_asset_ids=("storage",))
    cfg_on = base
    state = DispatchState.cold_start(cfg_off, demand.index[0])
    initial_soc = state.storages["storage"].soc_mwh_th

    # While disabled — solve and confirm SoC frozen.
    m_off = build_model(demand, price, state, cfg_off)
    solve(m_off, time_limit_s=30, mip_gap=0.005)
    d_off = extract_dispatch(m_off, n_intervals=4, solve_time=demand.index[0])
    assert all(v == pytest.approx(initial_soc) for v in d_off.soc_trajectory_mwh_th["storage"])

    # Operator re-enables. Same state (SoC unchanged), no disable.
    m_on = build_model(demand, price, state, cfg_on)
    r = solve(m_on, time_limit_s=30, mip_gap=0.005)
    assert r.feasible
    # No assertion on the trajectory's exact shape (solver-dependent) — only
    # that the SoC variable is no longer forcibly pinned to initial_soc.
    d_on = extract_dispatch(m_on, n_intervals=4, solve_time=demand.index[0])
    # At least one of charge / discharge / drift differs from the frozen run.
    differs = (
        d_on.sto_charge_mw_th["storage"] != d_off.sto_charge_mw_th["storage"]
        or d_on.sto_discharge_mw_th["storage"] != d_off.sto_discharge_mw_th["storage"]
        or d_on.soc_trajectory_mwh_th["storage"] != d_off.soc_trajectory_mwh_th["storage"]
    )
    assert differs, "re-enable should unlock storage decisions"


def test_disabled_boiler_does_not_emit_shutdown_event_at_t1(constant_inputs):
    """If a boiler was on going into a disable, the shutdown marker
    ``d_boiler`` must NOT fire at t=1. The MILP would otherwise satisfy
    ``d_boiler[1] >= z_prev - z[1] = 1`` for free (no objective term), and
    a downstream dispatch consumer would read that as the optimizer's
    choice to shut down, not the operator's forced disable."""
    import pyomo.environ as pyo
    demand, price = constant_inputs
    cfg = replace(PlantConfig.legacy_default(), disabled_asset_ids=("boiler",))
    state = DispatchState.cold_start(cfg, demand.index[0])
    state.units["boiler"] = UnitState(on=1, time_in_state_steps=TIS_LONG)
    m = build_model(demand, price, state, cfg)
    r = solve(m, time_limit_s=30, mip_gap=0.005)
    assert r.feasible
    for t in m.T_set:
        assert pyo.value(m.d_boiler["boiler", t]) == pytest.approx(0.0)
        assert pyo.value(m.s_boiler["boiler", t]) == pytest.approx(0.0)


def test_disabled_chp_does_not_emit_startup_event_after_re_enable_window(constant_inputs):
    """Same contract for CHP. ``s_chp`` already had a positive startup cost
    that the solver minimizes, but ``d_chp`` had no objective coefficient
    and would otherwise float freely; pin it to 0 for honest dispatch
    reporting."""
    import pyomo.environ as pyo
    demand, price = constant_inputs
    cfg = replace(PlantConfig.legacy_default(), disabled_asset_ids=("chp",))
    state = DispatchState.cold_start(cfg, demand.index[0])
    state.units["chp"] = UnitState(on=1, time_in_state_steps=TIS_LONG)
    m = build_model(demand, price, state, cfg)
    r = solve(m, time_limit_s=30, mip_gap=0.005)
    assert r.feasible
    for t in m.T_set:
        assert pyo.value(m.d_chp["chp", t]) == pytest.approx(0.0)
        assert pyo.value(m.s_chp["chp", t]) == pytest.approx(0.0)


def test_disable_does_not_break_min_down_on_other_assets(constant_inputs):
    """Disabling boiler must not collaterally affect CHP's min-down init fix."""
    demand, price = constant_inputs
    cfg = replace(PlantConfig.legacy_default(), disabled_asset_ids=("boiler",))
    state = DispatchState.cold_start(cfg, demand.index[0])
    # CHP just shut down with TIS=1; min_down_steps=8 should still force z=0
    # for the next 7 steps on the CHP — independent of the boiler being off.
    state.units["chp"] = UnitState(on=0, time_in_state_steps=1)
    m = build_model(demand, price, state, cfg)
    r = solve(m, time_limit_s=30, mip_gap=0.005)
    assert r.feasible
    d = extract_dispatch(m, n_intervals=4, solve_time=demand.index[0])
    # First 4 steps (one commit hour) of CHP should be off — well within the
    # 7-step min-down remainder.
    assert d.chp_p_el_mw["chp"] == [pytest.approx(0.0)] * 4

