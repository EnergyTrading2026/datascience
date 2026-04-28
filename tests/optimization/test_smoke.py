"""End-to-end smoke: build + solve a 35h MILP and pin the objective.

The reference value `obj=2571.7 EUR` comes from the notebook self-test in
`notebooks/optimization/mpc_prototype.ipynb` cell 3 — same inputs (constant
demand=10 MW_th, constant price=60 EUR/MWh_el, 35h horizon, default cold-start
state) and same MILP. Floor disabled to match the notebook (which uses
bounds (0, 200) for SoC).

If this test fails, the lift broke something — investigate before adding logic.
"""
from __future__ import annotations

import pytest

from optimization.model import build_model
from optimization.solve import solve


def test_solves_constant_demand_matches_notebook(constant_inputs_35h, cold_state, params_no_floor):
    demand, price = constant_inputs_35h
    model = build_model(demand, price, cold_state, params_no_floor, demand_safety_factor=1.0)
    result = solve(model, time_limit_s=30, mip_gap=0.005)
    assert result.feasible
    # Notebook reference: 2571.7 EUR. Allow ±0.5% tolerance for MIP-gap noise.
    assert result.objective_eur == pytest.approx(2571.7, rel=0.005)


def test_solves_with_hard_floor_feasible(constant_inputs_35h, cold_state, params):
    """Default params (floor=50) should also be feasible on the same inputs.
    Cost is expected to differ from no-floor (storage less freedom)."""
    demand, price = constant_inputs_35h
    model = build_model(demand, price, cold_state, params, demand_safety_factor=1.0)
    result = solve(model, time_limit_s=30, mip_gap=0.005)
    assert result.feasible
