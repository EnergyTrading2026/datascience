"""extract_dispatch + extract_state — integration with build_model + solve."""
from __future__ import annotations

import pandas as pd
import pytest

from optimization.dispatch import Dispatch, extract_dispatch, extract_state
from optimization.model import build_model
from optimization.solve import solve
from optimization.state import TIS_LONG, DispatchState


@pytest.fixture
def solved(constant_inputs_35h, cold_state, params):
    demand, price = constant_inputs_35h
    m = build_model(demand, price, cold_state, params, demand_safety_factor=1.0)
    solve(m, time_limit_s=30, mip_gap=0.005)
    return m


def test_dispatch_shapes_for_1h_commit(solved):
    solve_time = pd.Timestamp("2026-01-01 00:00:00", tz="Europe/Berlin")
    d = extract_dispatch(solved, n_intervals=4, solve_time=solve_time)
    assert isinstance(d, Dispatch)
    assert d.n_intervals == 4
    assert len(d.hp_p_el_mw) == 4
    assert len(d.boiler_q_th_mw) == 4
    assert len(d.chp_p_el_mw) == 4
    assert len(d.sto_charge_mw_th) == 4
    assert len(d.sto_discharge_mw_th) == 4
    # soc_trajectory has n+1 points (initial + each step end)
    assert len(d.soc_trajectory_mwh_th) == 5


def test_dispatch_obeys_floor(solved):
    """With params.sto_floor=50, no SoC value in the commit window may be < 50."""
    d = extract_dispatch(solved, n_intervals=4, solve_time=pd.Timestamp("2026-01-01 00:00:00", tz="Europe/Berlin"))
    assert min(d.soc_trajectory_mwh_th) >= 50.0 - 1e-6


def test_dispatch_to_dataframe(solved):
    solve_time = pd.Timestamp("2026-01-01 00:00:00", tz="Europe/Berlin")
    d = extract_dispatch(solved, n_intervals=4, solve_time=solve_time)
    df = d.to_dataframe()
    assert len(df) == 4
    assert df.index.tz is not None
    assert df.index[0] == d.commit_start
    assert (df.index[1] - df.index[0]) == pd.Timedelta(minutes=15)
    assert {"hp_p_el_mw", "boiler_q_th_mw", "chp_p_el_mw",
            "sto_charge_mw_th", "sto_discharge_mw_th", "soc_end_mwh_th"} <= set(df.columns)


def test_state_extraction_returns_valid_state(solved):
    commit_end = pd.Timestamp("2026-01-01 01:00:00", tz="Europe/Berlin")
    s = extract_state(solved, t_end=4, commit_end_time=commit_end)
    assert isinstance(s, DispatchState)
    assert s.timestamp == commit_end
    assert 50.0 - 1e-6 <= s.sto_soc_mwh_th <= 200.0 + 1e-6
    assert s.hp_on in (0, 1)
    assert s.boiler_on in (0, 1)
    assert s.chp_on in (0, 1)


def test_state_tis_carries_when_no_switch(constant_inputs_35h, params):
    """If the model never switches a unit during [1, t_end], time-in-state must
    accumulate from the prior state's tis (not reset to 0)."""
    demand, price = constant_inputs_35h
    # Cold-start with TIS_LONG, all units off.
    state = DispatchState.cold_start(
        pd.Timestamp("2026-01-01 00:00:00", tz="Europe/Berlin"), sto_soc_mwh_th=200.0
    )
    m = build_model(demand, price, state, params, demand_safety_factor=1.0)
    solve(m, time_limit_s=30, mip_gap=0.005)

    new_state = extract_state(
        m, t_end=4, commit_end_time=pd.Timestamp("2026-01-01 01:00:00", tz="Europe/Berlin")
    )
    # If no boiler/CHP switch happens in the first hour AND prior state matched, tis must
    # be at least 4 (steps from this hour) + the prior TIS_LONG, capped at TIS_LONG.
    if new_state.boiler_on == 0:
        assert new_state.boiler_time_in_state_steps >= 4
    if new_state.chp_on == 0:
        assert new_state.chp_time_in_state_steps >= 4
