"""Numerical regression pin for the modular-assets refactor.

Locks today's behavior (PlantConfig.legacy_default(), 35h hourly horizon,
deterministic demand+price profile, mid-state init) to a hardcoded objective
and dispatch trajectory. Stays green only if the modular code, when configured
with the legacy 1-of-each-asset default, produces bit-identical output to the
pre-refactor hardcoded MILP.

The pin solves with ``mip_gap=1e-9`` (true optimum) instead of the prod
``0.005``. Reason: at the prod gap, multiple dispatches are within tolerance
of optimum and HiGHS picks one based on variable ordering, which the refactor
changes (named -> indexed vars). The true optimum is unique up to fp precision
and was verified bit-identical against main (commit 8751915, pre-refactor):
both old and new code converge to 732.6275984854354 EUR.
The committed 1h dispatch is identical at both gaps; only the 35h horizon
objective differs in the prod-gap window.

If this test breaks: STOP and investigate. Drift here means prod behavior
changed, regardless of whether the rest of the suite is green. The
re-harvesting recipe lives in ``scripts/_archive/harvest_regression_pin.py``
(checked out against the pre-refactor commit 8751915).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from optimization.config import PlantConfig, RuntimeConfig
from optimization.dispatch import extract_dispatch, extract_state
from optimization.model import build_model
from optimization.solve import solve
from optimization.state import DispatchState, StorageState, UnitState

# --- Pinned goldens (do not edit without re-harvesting on a known-good commit) ---

GOLDEN_OBJECTIVE_EUR = 732.6275984854354
PIN_MIP_GAP = 1e-9

GOLDEN_HP_P_EL_MW = [0.0, 0.0, 0.0, 0.0]
GOLDEN_BOILER_Q_TH_MW = [0.0, 0.0, 0.0, 0.0]
GOLDEN_CHP_P_EL_MW = [0.0, 0.0, 0.0, 0.0]
GOLDEN_STO_CHARGE_MW_TH = [0.0, 0.0, 0.0, 0.0]
GOLDEN_STO_DISCHARGE_MW_TH = [4.0, 4.0, 4.0, 4.0]
GOLDEN_SOC_TRAJECTORY_MWH_TH = [
    120.0,
    118.999875,
    117.99975,
    116.99962500000001,
    115.99950000000001,
]

GOLDEN_FINAL_STATE = {
    "sto_soc_mwh_th": 115.99950000000001,
    "hp_on": 0,
    "boiler_on": 0,
    "boiler_time_in_state_steps": 4,
    "chp_on": 0,
    "chp_time_in_state_steps": 24,
}

# Tight tolerance: HiGHS is deterministic on identical input, so the only
# legitimate drift is float reordering. 1e-6 catches anything bigger.
TOL = 1e-6


def _make_inputs() -> tuple[pd.Series, pd.Series, DispatchState]:
    idx = pd.date_range("2026-01-01 00:00:00", periods=35, freq="1h", tz="Europe/Berlin")
    hours = np.arange(35) % 24
    demand = 8.0 + 4.0 * np.sin((hours - 6) / 24 * 2 * np.pi)
    forecast = pd.Series(demand, index=idx, name="demand_mw_th")
    price = 40.0 + 30.0 * np.sin((hours - 9) / 24 * 2 * np.pi) + 5.0 * (hours == 18)
    prices = pd.Series(price, index=idx, name="price_eur_mwh")
    state = DispatchState(
        timestamp=idx[0],
        units={
            "hp": UnitState(on=0, time_in_state_steps=999),
            "boiler": UnitState(on=1, time_in_state_steps=8),
            "chp": UnitState(on=0, time_in_state_steps=20),
        },
        storages={"storage": StorageState(soc_mwh_th=120.0)},
    )
    return forecast, prices, state


def test_regression_pin_legacy_default():
    """End-to-end solve must match pinned objective + dispatch + final state."""
    forecast, prices, state = _make_inputs()
    config = PlantConfig.legacy_default()
    runtime = RuntimeConfig()

    model = build_model(forecast, prices, state, config, resolution="hour")
    result = solve(
        model,
        time_limit_s=120,  # tight gap takes longer
        mip_gap=PIN_MIP_GAP,
    )
    n_intervals = runtime.commit_hours * 4
    dispatch = extract_dispatch(model, n_intervals=n_intervals, solve_time=forecast.index[0])
    new_state = extract_state(
        model,
        t_end=n_intervals,
        commit_end_time=forecast.index[0] + pd.Timedelta(hours=runtime.commit_hours),
    )

    assert result.objective_eur == pytest.approx(GOLDEN_OBJECTIVE_EUR, abs=TOL)

    assert dispatch.hp_p_el_mw["hp"] == pytest.approx(GOLDEN_HP_P_EL_MW, abs=TOL)
    assert dispatch.boiler_q_th_mw["boiler"] == pytest.approx(GOLDEN_BOILER_Q_TH_MW, abs=TOL)
    assert dispatch.chp_p_el_mw["chp"] == pytest.approx(GOLDEN_CHP_P_EL_MW, abs=TOL)
    assert dispatch.sto_charge_mw_th["storage"] == pytest.approx(GOLDEN_STO_CHARGE_MW_TH, abs=TOL)
    assert dispatch.sto_discharge_mw_th["storage"] == pytest.approx(
        GOLDEN_STO_DISCHARGE_MW_TH, abs=TOL,
    )
    assert dispatch.soc_trajectory_mwh_th["storage"] == pytest.approx(
        GOLDEN_SOC_TRAJECTORY_MWH_TH, abs=TOL,
    )

    assert new_state.storages["storage"].soc_mwh_th == pytest.approx(
        GOLDEN_FINAL_STATE["sto_soc_mwh_th"], abs=TOL,
    )
    assert new_state.units["hp"].on == GOLDEN_FINAL_STATE["hp_on"]
    assert new_state.units["boiler"].on == GOLDEN_FINAL_STATE["boiler_on"]
    assert new_state.units["boiler"].time_in_state_steps == GOLDEN_FINAL_STATE["boiler_time_in_state_steps"]
    assert new_state.units["chp"].on == GOLDEN_FINAL_STATE["chp_on"]
    assert new_state.units["chp"].time_in_state_steps == GOLDEN_FINAL_STATE["chp_time_in_state_steps"]
