"""extract_dispatch + extract_state — integration with build_model + solve."""
from __future__ import annotations

import pandas as pd
import pytest

from optimization.dispatch import Dispatch, extract_dispatch, extract_state
from optimization.model import build_model
from optimization.solve import solve
from optimization.state import TIS_LONG, DispatchState


@pytest.fixture
def solved(constant_inputs_35h, cold_state, config):
    demand, price = constant_inputs_35h
    m = build_model(demand, price, cold_state, config, demand_safety_factor=1.0)
    solve(m, time_limit_s=30, mip_gap=0.005)
    return m


def test_dispatch_shapes_for_1h_commit(solved):
    solve_time = pd.Timestamp("2026-01-01 00:00:00", tz="Europe/Berlin")
    d = extract_dispatch(solved, n_intervals=4, solve_time=solve_time)
    assert isinstance(d, Dispatch)
    assert d.n_intervals == 4
    # Per-asset dicts: legacy default has 1 asset per family.
    assert len(d.hp_p_el_mw["hp"]) == 4
    assert len(d.boiler_q_th_mw["boiler"]) == 4
    assert len(d.chp_p_el_mw["chp"]) == 4
    assert len(d.sto_charge_mw_th["storage"]) == 4
    assert len(d.sto_discharge_mw_th["storage"]) == 4
    # soc_trajectory has n+1 points (initial + each step end)
    assert len(d.soc_trajectory_mwh_th["storage"]) == 5


def test_dispatch_obeys_floor(solved):
    """With storage floor=50, no SoC value in the commit window may be < 50."""
    d = extract_dispatch(
        solved, n_intervals=4,
        solve_time=pd.Timestamp("2026-01-01 00:00:00", tz="Europe/Berlin"),
    )
    assert min(d.soc_trajectory_mwh_th["storage"]) >= 50.0 - 1e-6


def test_dispatch_to_dataframe(solved):
    solve_time = pd.Timestamp("2026-01-01 00:00:00", tz="Europe/Berlin")
    d = extract_dispatch(solved, n_intervals=4, solve_time=solve_time)
    df = d.to_dataframe()
    # Long format: 4 timesteps * 6 quantities (hp p_el, boiler q_th, chp p_el,
    # storage charge/discharge/soc_end) for the 1-of-each legacy default.
    assert len(df) == 4 * 6
    assert set(df.columns) == {"asset_id", "family", "quantity", "value"}
    # tz-aware DatetimeIndex with non-unique 15-min steps.
    assert df.index.tz is not None
    unique_ts = df.index.unique().sort_values()
    assert len(unique_ts) == 4
    assert unique_ts[0] == d.commit_start
    assert (unique_ts[1] - unique_ts[0]) == pd.Timedelta(minutes=15)
    # Every (family, quantity) combination present exactly once per timestamp
    # (legacy default has 1 asset per family).
    expected_pairs = {
        ("hp", "p_el_mw"), ("boiler", "q_th_mw"), ("chp", "p_el_mw"),
        ("storage", "charge_mw_th"), ("storage", "discharge_mw_th"),
        ("storage", "soc_end_mwh_th"),
    }
    assert set(zip(df["family"], df["quantity"])) == expected_pairs


def test_state_extraction_returns_valid_state(solved, config):
    commit_end = pd.Timestamp("2026-01-01 01:00:00", tz="Europe/Berlin")
    s = extract_state(solved, t_end=4, commit_end_time=commit_end)
    assert isinstance(s, DispatchState)
    assert s.timestamp == commit_end
    soc = s.storages["storage"].soc_mwh_th
    assert 50.0 - 1e-6 <= soc <= 200.0 + 1e-6
    assert s.units["hp"].on in (0, 1)
    assert s.units["boiler"].on in (0, 1)
    assert s.units["chp"].on in (0, 1)
    s.covers(config)


def test_state_tis_carries_when_no_switch(constant_inputs_35h, config):
    """If the model never switches a unit during [1, t_end], time-in-state must
    accumulate from the prior state's tis (not reset to 0)."""
    demand, price = constant_inputs_35h
    state = DispatchState.cold_start(
        config, pd.Timestamp("2026-01-01 00:00:00", tz="Europe/Berlin"),
    )
    m = build_model(demand, price, state, config, demand_safety_factor=1.0)
    solve(m, time_limit_s=30, mip_gap=0.005)

    new_state = extract_state(
        m, t_end=4, commit_end_time=pd.Timestamp("2026-01-01 01:00:00", tz="Europe/Berlin")
    )
    if new_state.units["boiler"].on == 0:
        assert new_state.units["boiler"].time_in_state_steps >= 4
    if new_state.units["chp"].on == 0:
        assert new_state.units["chp"].time_in_state_steps >= 4
