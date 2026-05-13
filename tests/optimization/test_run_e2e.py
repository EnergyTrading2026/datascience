"""End-to-end test: cold-start cycle using the real SMARD CSV + a synthetic forecast.

Verifies the full chain:
  CLI args -> state load -> SMARD slice -> forecast load -> build_model -> solve
            -> extract_dispatch -> extract_state -> write parquet+json -> exit 0.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from optimization import run as run_module
from optimization.adapters.forecast import DEMAND_COLUMN
from optimization.run import _upsample_hour_to_quarterhour, run_one_cycle
from optimization.state import DispatchState

PRICES_CSV = Path("data/optimization/raw/Gro_handelspreise_202403010000_202603020000_Stunde.csv")
EXPORT_TOP_LEVEL_KEYS = {
    "schema_version",
    "run_id",
    "status",
    "approach",
    "time_window",
    "summary",
    "dispatch",
    "initial_state",
    "next_initial_state",
    "diagnostics",
}
EXPORT_DISPATCH_KEYS = {
    "timestamp",
    "step",
    "demand_mw_th",
    "price_eur_per_mwh_el",
    "heat_pumps",
    "boilers",
    "chps",
    "storages",
    "heat_slack_mw_th",
}


def _assert_backend_export_shape(
    payload: dict[str, object],
    solve_time: pd.Timestamp,
    *,
    expected_n_dispatch: int = 4,
    expected_unit_ids: set[str] | None = None,
    expected_storage_ids: set[str] | None = None,
) -> None:
    """Assert v2.0 backend payload shape.

    expected_unit_ids / expected_storage_ids let multi-asset tests verify each
    family array has the right entries. Defaults to the legacy_default plant
    (1×HP, 1×boiler, 1×CHP, 1×storage)."""
    if expected_unit_ids is None:
        expected_unit_ids = {"hp", "boiler", "chp"}
    if expected_storage_ids is None:
        expected_storage_ids = {"storage"}

    assert set(payload) == EXPORT_TOP_LEVEL_KEYS
    assert payload["schema_version"] == "2.0"
    assert payload["run_id"] == f"mpc-{solve_time.isoformat()}"
    assert payload["status"] in {"optimal", "feasible"}
    assert payload["approach"] == {
        "name": "hourly_mpc_milp",
        "solve_horizon_hours": 35,
        "commit_horizon_hours": 1,
        "dt_hours": 0.25,
    }
    assert payload["time_window"] == {
        "start": solve_time.isoformat(),
        "end": (solve_time + pd.Timedelta(hours=1)).isoformat(),
        "timezone": "Europe/Berlin",
    }

    summary = payload["summary"]
    assert isinstance(summary, dict)
    assert set(summary) == {
        "objective_cost_eur",
        "real_cost_eur",
        "runtime_seconds",
        "solver",
        "termination_condition",
    }
    assert isinstance(summary["objective_cost_eur"], float)
    assert isinstance(summary["real_cost_eur"], float)
    assert isinstance(summary["runtime_seconds"], float)
    assert summary["solver"] == "appsi_highs"
    assert summary["termination_condition"] in {"optimal", "feasible"}

    dispatch = payload["dispatch"]
    assert isinstance(dispatch, list)
    assert len(dispatch) == expected_n_dispatch
    for step, row in enumerate(dispatch):
        assert set(row) == EXPORT_DISPATCH_KEYS
        assert row["timestamp"] == (solve_time + pd.Timedelta(minutes=15 * step)).isoformat()
        assert row["step"] == step
        assert isinstance(row["demand_mw_th"], float)
        assert isinstance(row["price_eur_per_mwh_el"], float)

        for family_key, unit_fields in (
            ("heat_pumps", {"id", "on", "el_in_mw", "th_out_mw"}),
            ("boilers", {"id", "on", "th_out_mw"}),
            ("chps", {"id", "on", "el_out_mw", "th_out_mw"}),
        ):
            assert isinstance(row[family_key], list)
            for entry in row[family_key]:
                assert set(entry) == unit_fields
                assert isinstance(entry["id"], str) and entry["id"]
                assert isinstance(entry["on"], bool)
        assert isinstance(row["storages"], list)
        for storage_entry in row["storages"]:
            assert set(storage_entry) == {
                "id",
                "charge_mw_th",
                "discharge_mw_th",
                "soc_mwh_th",
            }
            assert isinstance(storage_entry["id"], str) and storage_entry["id"]

    initial_state = payload["initial_state"]
    next_initial_state = payload["next_initial_state"]
    for state in (initial_state, next_initial_state):
        assert isinstance(state, dict)
        assert set(state) == {"units", "storages"}
        for unit_payload in state["units"].values():
            assert set(unit_payload) == {"on", "time_in_state_steps"}
            assert isinstance(unit_payload["on"], bool)
            assert isinstance(unit_payload["time_in_state_steps"], int)
        for storage_payload in state["storages"].values():
            assert set(storage_payload) == {"soc_mwh_th"}
            assert isinstance(storage_payload["soc_mwh_th"], float)

    assert set(next_initial_state["units"]) == expected_unit_ids
    assert set(next_initial_state["storages"]) == expected_storage_ids

    assert payload["diagnostics"] == {"notes": "", "warnings": []}


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
    soc = s.storages["storage"].soc_mwh_th
    assert 50.0 - 1e-6 <= soc <= 200.0 + 1e-6
    assert s.timestamp == solve_time + pd.Timedelta(hours=1)  # 1h commit

    # Dispatch parquet is long-format: 4 timesteps * 6 (family, quantity) pairs
    df = pd.read_parquet(dispatch_out)
    assert len(df) == 4 * 6
    assert set(df.columns) == {"asset_id", "family", "quantity", "value"}
    expected_pairs = {
        ("hp", "p_el_mw"), ("boiler", "q_th_mw"), ("chp", "p_el_mw"),
        ("storage", "charge_mw_th"), ("storage", "discharge_mw_th"),
        ("storage", "soc_end_mwh_th"),
    }
    assert set(zip(df["family"], df["quantity"])) == expected_pairs


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
    soc2 = s2.storages["storage"].soc_mwh_th
    assert 50.0 - 1e-6 <= soc2 <= 200.0 + 1e-6


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

    monkeypatch.setattr(run_module.smard_live_io, "get_published_prices", fake_get_prices)

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
    # 1h commit at QH model -> 4 unique 15-min timestamps; long format so
    # row count is 4 * 6 (family, quantity) pairs.
    assert len(df) == 4 * 6
    unique_ts = df.index.unique().sort_values()
    assert len(unique_ts) == 4
    assert (unique_ts[1] - unique_ts[0]) == pd.Timedelta(minutes=15)
    payload = json.loads(export_out.read_text())
    _assert_backend_export_shape(payload, solve_time)
    assert payload["dispatch"][0]["heat_pumps"][0]["el_in_mw"] >= 0


def test_run_one_cycle_exports_multi_asset_arrays(tmp_path, monkeypatch):
    """Multi-asset config (2 HPs) flows through the export pipeline: every
    dispatch row's ``heat_pumps`` array surfaces both HP ids; next_initial_state
    lists both units. v2.0 schema preserves per-asset granularity end to end."""
    from optimization.config import HeatPumpParams, PlantConfig

    solve_time = pd.Timestamp("2026-01-01 13:00:00", tz="Europe/Berlin")
    idx_h = pd.date_range(solve_time, periods=35, freq="1h", tz="Europe/Berlin")
    forecast_path = tmp_path / "forecast.parquet"
    pd.DataFrame({DEMAND_COLUMN: [10.0] * 35}, index=idx_h).to_parquet(forecast_path)

    def fake_get_prices(at_time, resolution, horizon_h=48):
        anchor = at_time.tz_convert("Europe/Berlin").floor("15min")
        idx = pd.date_range(anchor, periods=horizon_h * 4, freq="15min", tz="Europe/Berlin")
        return pd.Series([60.0] * len(idx), index=idx, name="price_eur_mwh")

    monkeypatch.setattr(run_module.smard_live_io, "get_published_prices", fake_get_prices)

    base = PlantConfig.legacy_default()
    multi_cfg = replace(
        base,
        heat_pumps=(
            HeatPumpParams(id="hp_1", p_el_min_mw=0.5, p_el_max_mw=4.0, cop=3.5),
            HeatPumpParams(id="hp_2", p_el_min_mw=0.5, p_el_max_mw=4.0, cop=3.0),
        ),
    )

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
        config=multi_cfg,
        prices_source="live",
        resolution="quarterhour",
        forecast_resolution="hour",
    )
    assert rc == 0

    payload = json.loads(export_out.read_text())
    _assert_backend_export_shape(
        payload,
        solve_time,
        expected_unit_ids={"hp_1", "hp_2", "boiler", "chp"},
        expected_storage_ids={"storage"},
    )
    for row in payload["dispatch"]:
        ids = {hp["id"] for hp in row["heat_pumps"]}
        assert ids == {"hp_1", "hp_2"}


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
