"""End-to-end test: cold-start cycle using the real SMARD CSV + a synthetic forecast.

Verifies the full chain:
  CLI args -> state load -> SMARD slice -> forecast load -> build_model -> solve
            -> extract_dispatch -> extract_state -> write parquet+json -> exit 0.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from optimization.adapters.forecast import DEMAND_COLUMN
from optimization.run import run_one_cycle
from optimization.state import DispatchState

PRICES_CSV = Path("data/Gro_handelspreise_202403010000_202603020000_Stunde.csv")


@pytest.mark.skipif(not PRICES_CSV.exists(), reason="SMARD CSV not present in data/")
def test_cold_start_cycle_writes_dispatch_and_state(tmp_path):
    # Synthetic 35h forecast at a date covered by the SMARD CSV
    solve_time = pd.Timestamp("2025-12-01 13:00:00", tz="Europe/Berlin")
    idx = pd.date_range(solve_time, periods=35, freq="1h", tz="Europe/Berlin")
    forecast_path = tmp_path / "forecast.parquet"
    pd.DataFrame({DEMAND_COLUMN: [10.0] * 35}, index=idx).to_parquet(forecast_path)

    state_out = tmp_path / "state.json"
    dispatch_out = tmp_path / "dispatch.parquet"

    rc = run_one_cycle(
        solve_time=solve_time,
        forecast_path=forecast_path,
        prices_path=PRICES_CSV,
        state_in=None,
        state_out=state_out,
        dispatch_out=dispatch_out,
        cold_start=True,
    )
    assert rc == 0
    assert state_out.exists()
    assert dispatch_out.exists()

    # State is loadable + obeys floor
    s = DispatchState.load(state_out)
    assert 50.0 - 1e-6 <= s.sto_soc_mwh_th <= 200.0 + 1e-6
    assert s.timestamp == solve_time + pd.Timedelta(hours=1)  # 1h commit

    # Dispatch parquet has 4 rows (1h * 4 intervals/h) with expected columns
    df = pd.read_parquet(dispatch_out)
    assert len(df) == 4
    assert {"hp_p_el_mw", "boiler_q_th_mw", "chp_p_el_mw",
            "sto_charge_mw_th", "sto_discharge_mw_th", "soc_end_mwh_th"} <= set(df.columns)


@pytest.mark.skipif(not PRICES_CSV.exists(), reason="SMARD CSV not present in data/")
def test_state_chain_two_cycles(tmp_path):
    """First cycle cold-starts, second cycle reads its state. Verifies SoC + state continuity."""
    DEMAND = 10.0
    state_path = tmp_path / "state.json"
    dispatch_dir = tmp_path / "dispatch"

    # ---- Cycle 1 (cold start) at 13:00 ----
    t1 = pd.Timestamp("2025-12-01 13:00:00", tz="Europe/Berlin")
    idx1 = pd.date_range(t1, periods=35, freq="1h", tz="Europe/Berlin")
    f1 = tmp_path / "f1.parquet"
    pd.DataFrame({DEMAND_COLUMN: [DEMAND] * 35}, index=idx1).to_parquet(f1)
    rc = run_one_cycle(
        solve_time=t1, forecast_path=f1, prices_path=PRICES_CSV,
        state_in=None, state_out=state_path, dispatch_out=dispatch_dir / "1.parquet",
        cold_start=True,
    )
    assert rc == 0
    s1 = DispatchState.load(state_path)

    # ---- Cycle 2 at 14:00, reading state from cycle 1 ----
    t2 = t1 + pd.Timedelta(hours=1)
    idx2 = pd.date_range(t2, periods=35, freq="1h", tz="Europe/Berlin")
    f2 = tmp_path / "f2.parquet"
    pd.DataFrame({DEMAND_COLUMN: [DEMAND] * 35}, index=idx2).to_parquet(f2)
    rc = run_one_cycle(
        solve_time=t2, forecast_path=f2, prices_path=PRICES_CSV,
        state_in=state_path, state_out=state_path, dispatch_out=dispatch_dir / "2.parquet",
        cold_start=False,
    )
    assert rc == 0
    s2 = DispatchState.load(state_path)
    assert s2.timestamp == t2 + pd.Timedelta(hours=1)
    # State must have advanced (floor still respected)
    assert 50.0 - 1e-6 <= s2.sto_soc_mwh_th <= 200.0 + 1e-6
