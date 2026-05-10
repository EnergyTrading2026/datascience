"""End-to-end test: cold-start cycle using the real SMARD CSV + a synthetic forecast.

Verifies the full chain:
  CLI args -> state load -> SMARD slice -> forecast load -> build_model -> solve
            -> extract_dispatch -> extract_state -> write parquet+json -> exit 0.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from optimization.adapters import smard_live as smard_live_io
from optimization.adapters.forecast import DEMAND_COLUMN
from optimization.run import _upsample_hour_to_quarterhour, run_one_cycle
from optimization.state import DispatchState

PRICES_CSV = Path("data/optimization/raw/Gro_handelspreise_202403010000_202603020000_Stunde.csv")


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
        prices_source="csv",
        resolution="hour",
        forecast_resolution="hour",
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
        cold_start=True, prices_source="csv", resolution="hour", forecast_resolution="hour",
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
        cold_start=False, prices_source="csv", resolution="hour", forecast_resolution="hour",
    )
    assert rc == 0
    s2 = DispatchState.load(state_path)
    assert s2.timestamp == t2 + pd.Timedelta(hours=1)
    # State must have advanced (floor still respected)
    assert 50.0 - 1e-6 <= s2.sto_soc_mwh_th <= 200.0 + 1e-6


# ---------------- Hybrid mode: hourly forecast + QH prices ----------------
def test_upsample_hour_to_quarterhour_repeats_each_value_four_times():
    idx = pd.date_range("2026-01-01 13:00:00", periods=3, freq="1h", tz="Europe/Berlin")
    s = pd.Series([10.0, 20.0, 30.0], index=idx, name="demand_mw_th")
    out = _upsample_hour_to_quarterhour(s)
    assert len(out) == 12
    assert (out.index[1] - out.index[0]) == pd.Timedelta(minutes=15)
    assert out.index[0] == idx[0]
    assert list(out.iloc[:4]) == [10.0, 10.0, 10.0, 10.0]
    assert list(out.iloc[4:8]) == [20.0, 20.0, 20.0, 20.0]
    assert list(out.iloc[8:12]) == [30.0, 30.0, 30.0, 30.0]
    assert out.name == "demand_mw_th"


def test_upsample_handles_empty_series():
    idx = pd.DatetimeIndex([], tz="Europe/Berlin")
    s = pd.Series([], index=idx, name="demand_mw_th", dtype=float)
    out = _upsample_hour_to_quarterhour(s)
    assert len(out) == 0


def test_hybrid_mode_runs_with_mocked_qh_smard(tmp_path, monkeypatch):
    """End-to-end: hourly forecast file + mocked QH prices -> model solves at 15-min."""
    solve_time = pd.Timestamp("2026-01-01 13:00:00", tz="Europe/Berlin")
    # Hourly forecast (35h), unchanged contract.
    idx_h = pd.date_range(solve_time, periods=35, freq="1h", tz="Europe/Berlin")
    forecast_path = tmp_path / "forecast.parquet"
    pd.DataFrame({DEMAND_COLUMN: [10.0] * 35}, index=idx_h).to_parquet(forecast_path)

    # Mock SMARD live to return a deterministic QH curve covering the horizon.
    def fake_get_prices(at_time, resolution, horizon_h=48):
        assert resolution == "quarterhour"
        # Anchor at next 15-min boundary >= at_time
        floored = at_time.tz_convert("Europe/Berlin").floor("15min")
        anchor = floored if floored == at_time else at_time.tz_convert("Europe/Berlin").ceil("15min")
        idx = pd.date_range(anchor, periods=horizon_h * 4, freq="15min", tz="Europe/Berlin")
        # Vary price across the day (rough sine) to give the optimizer something to chew on.
        import numpy as np
        vals = 60.0 + 20.0 * np.sin(np.arange(len(idx)) * 2 * np.pi / 96)
        return pd.Series(vals, index=idx, name="price_eur_mwh")

    monkeypatch.setattr(smard_live_io, "get_published_prices", fake_get_prices)

    state_out = tmp_path / "state.json"
    dispatch_out = tmp_path / "dispatch.parquet"
    export_out = tmp_path / "export.json"
    rc = run_one_cycle(
        solve_time=solve_time,
        forecast_path=forecast_path,
        prices_path=None,
        state_in=None,
        state_out=state_out,
        dispatch_out=dispatch_out,
        export_out=export_out,
        cold_start=True,
        prices_source="live",
        resolution="quarterhour",
        forecast_resolution="hour",
    )
    assert rc == 0
    df = pd.read_parquet(dispatch_out)
    # 1h commit at QH model -> 4 dispatch rows
    assert len(df) == 4
    assert (df.index[1] - df.index[0]) == pd.Timedelta(minutes=15)
    payload = json.loads(export_out.read_text())
    assert payload["schema_version"] == "1.0"
    assert payload["status"] in {"optimal", "feasible"}
    assert len(payload["dispatch"]) == 4
    assert payload["dispatch"][0]["heat_pump"]["el_in_mw"] >= 0
    assert payload["next_initial_state"]["soc_mwh_th"] >= 50.0 - 1e-6


def test_hybrid_mode_rejects_inverse_combo(tmp_path):
    """forecast=quarterhour with model=hour would lose information -> exit 1."""
    solve_time = pd.Timestamp("2026-01-01 13:00:00", tz="Europe/Berlin")
    forecast_path = tmp_path / "f.parquet"
    pd.DataFrame(
        {DEMAND_COLUMN: [10.0] * 4},
        index=pd.date_range(solve_time, periods=4, freq="15min", tz="Europe/Berlin"),
    ).to_parquet(forecast_path)
    rc = run_one_cycle(
        solve_time=solve_time,
        forecast_path=forecast_path,
        prices_path=Path("ignored"),
        state_in=None,
        state_out=tmp_path / "state.json",
        dispatch_out=tmp_path / "dispatch.parquet",
        cold_start=True,
        prices_source="csv",
        resolution="hour",
        forecast_resolution="quarterhour",
    )
    assert rc == 1
