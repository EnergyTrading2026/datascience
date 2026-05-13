"""End-to-end smoke: build + solve a 35h MILP and pin the objective.

The reference value `obj=2571.7 EUR` comes from the notebook self-test in
`notebooks/optimization/mpc_prototype.ipynb` cell 3 — same inputs (constant
demand=10 MW_th, constant price=60 EUR/MWh_el, 35h horizon, default cold-start
state) and same MILP. Floor disabled to match the notebook (which uses
bounds (0, 200) for SoC).

If this test fails, the lift broke something — investigate before adding logic.
"""
from __future__ import annotations

import pandas as pd
import pytest

from optimization.model import build_model
from optimization.solve import SolverInfeasibleError, solve
from optimization.state import TIS_LONG, DispatchState, StorageState, UnitState


def test_solves_constant_demand_matches_notebook(constant_inputs_35h, cold_state, config_no_floor):
    demand, price = constant_inputs_35h
    model = build_model(demand, price, cold_state, config_no_floor, demand_safety_factor=1.0)
    result = solve(model, time_limit_s=30, mip_gap=0.005)
    assert result.feasible
    # Notebook reference: 2571.7 EUR. Allow ±0.5% tolerance for MIP-gap noise.
    assert result.objective_eur == pytest.approx(2571.7, rel=0.005)


def test_solves_with_hard_floor_feasible(constant_inputs_35h, cold_state, config):
    """Default config (floor=50) should also be feasible on the same inputs.
    Cost is expected to differ from no-floor (storage less freedom)."""
    demand, price = constant_inputs_35h
    model = build_model(demand, price, cold_state, config, demand_safety_factor=1.0)
    result = solve(model, time_limit_s=30, mip_gap=0.005)
    assert result.feasible


def test_solve_raises_on_infeasible_model(config):
    """Regression for the Pyomo-HiGHS no-feasible-solution path.

    When HiGHS terminates without an incumbent, Pyomo's appsi wrapper raises
    RuntimeError directly from solver.solve(...). The wrapper must catch that
    and reraise as SolverInfeasibleError so callers have a single contract.

    Construct a deliberately infeasible configuration: the boiler is forced ON
    by a min-up obligation carried over from the previous solve, but demand is
    below the boiler's Q_min and storage is at the upper bound, so the excess
    heat has no destination.
    """
    idx = pd.date_range("2026-01-01", periods=4, freq="1h", tz="Europe/Berlin")
    demand = pd.Series(1.5, index=idx, name="demand_mw_th")
    price = pd.Series(50.0, index=idx, name="price_eur_mwh")

    storage = config.storages[0]
    state = DispatchState(
        timestamp=idx[0],
        units={
            "hp": UnitState(on=0, time_in_state_steps=TIS_LONG),
            "boiler": UnitState(on=1, time_in_state_steps=1),
            "chp": UnitState(on=0, time_in_state_steps=TIS_LONG),
        },
        storages={storage.id: StorageState(soc_mwh_th=storage.capacity_mwh_th)},
    )
    model = build_model(demand, price, state, config)
    with pytest.raises(SolverInfeasibleError):
        solve(model)
