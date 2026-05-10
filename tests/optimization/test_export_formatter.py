"""Backend export formatter regressions."""
from __future__ import annotations

import json

import pandas as pd

from optimization.export_formatter import export_to_json, prepare_optimization_export


def test_prepare_optimization_export_formats_backend_payload():
    idx = pd.date_range("2026-01-01 13:00:00", periods=2, freq="15min", tz="Europe/Berlin")
    dispatch_df = pd.DataFrame(
        {
            "demand_th": [10.0, 11.0],
            "price_el": [60.0, 62.5],
            "hp_el_in": [1.0, 1.5],
            "hp_th_out": [3.5, 5.25],
            "boiler_th_out": [0.0, 2.0],
            "chp_el_out": [2.0, 0.0],
            "chp_th_out": [2.4, 0.0],
            "storage_charge": [0.0, 1.25],
            "storage_discharge": [4.1, 0.0],
            "storage_soc": [198.9, 199.2],
            "heat_slack": [0.0, 0.0],
            "hp_on": [True, True],
            "boiler_on": [False, True],
            "chp_on": [True, False],
        },
        index=idx,
    )

    payload = prepare_optimization_export(
        dispatch_df=dispatch_df,
        metadata={
            "run_id": "mpc-2026-01-01T13:00:00+01:00",
            "status": "optimal",
            "approach": {
                "name": "hourly_mpc_milp",
                "solve_horizon_hours": 35,
                "commit_horizon_hours": 1,
                "dt_hours": 0.25,
            },
            "time_window": {
                "start": idx[0],
                "end": idx[-1] + pd.Timedelta(minutes=15),
                "timezone": "Europe/Berlin",
            },
            "objective_cost_eur": 123.45,
            "real_cost_eur": 67.89,
        },
        solver_info={
            "solver": "appsi_highs",
            "runtime_seconds": 0.75,
            "termination_condition": "optimal",
            "status": "optimal",
        },
        initial_state={},
        next_state={
            "soc_mwh_th": 199.2,
            "heat_pump_on": True,
            "boiler_on": True,
            "chp_on": False,
            "boiler_time_in_state_steps": 1,
            "chp_time_in_state_steps": 0,
        },
    )

    assert set(payload) == {
        "schema_version",
        "run_id",
        "status",
        "approach",
        "time_window",
        "summary",
        "dispatch",
        "next_initial_state",
        "diagnostics",
    }
    assert payload["schema_version"] == "1.0"
    assert payload["run_id"] == "mpc-2026-01-01T13:00:00+01:00"
    assert payload["status"] == "optimal"
    assert payload["approach"] == {
        "name": "hourly_mpc_milp",
        "solve_horizon_hours": 35,
        "commit_horizon_hours": 1,
        "dt_hours": 0.25,
    }
    assert payload["time_window"] == {
        "start": "2026-01-01T13:00:00+01:00",
        "end": "2026-01-01T13:30:00+01:00",
        "timezone": "Europe/Berlin",
    }
    assert payload["summary"] == {
        "objective_cost_eur": 123.45,
        "real_cost_eur": 67.89,
        "runtime_seconds": 0.75,
        "solver": "appsi_highs",
        "termination_condition": "optimal",
    }
    assert payload["dispatch"] == [
        {
            "timestamp": "2026-01-01T13:00:00+01:00",
            "step": 0,
            "demand_mw_th": 10.0,
            "price_eur_per_mwh_el": 60.0,
            "heat_pump": {"on": True, "el_in_mw": 1.0, "th_out_mw": 3.5},
            "boiler": {"on": False, "th_out_mw": 0.0},
            "chp": {"on": True, "el_out_mw": 2.0, "th_out_mw": 2.4},
            "storage": {
                "charge_mw_th": 0.0,
                "discharge_mw_th": 4.1,
                "soc_mwh_th": 198.9,
            },
            "heat_slack_mw_th": 0.0,
        },
        {
            "timestamp": "2026-01-01T13:15:00+01:00",
            "step": 1,
            "demand_mw_th": 11.0,
            "price_eur_per_mwh_el": 62.5,
            "heat_pump": {"on": True, "el_in_mw": 1.5, "th_out_mw": 5.25},
            "boiler": {"on": True, "th_out_mw": 2.0},
            "chp": {"on": False, "el_out_mw": 0.0, "th_out_mw": 0.0},
            "storage": {
                "charge_mw_th": 1.25,
                "discharge_mw_th": 0.0,
                "soc_mwh_th": 199.2,
            },
            "heat_slack_mw_th": 0.0,
        },
    ]
    assert payload["next_initial_state"] == {
        "soc_mwh_th": 199.2,
        "heat_pump_on": True,
        "boiler_on": True,
        "chp_on": False,
        "boiler_time_in_state_steps": 1,
        "chp_time_in_state_steps": 0,
    }
    assert payload["diagnostics"] == {"notes": "", "warnings": []}
    assert json.loads(export_to_json(payload, indent=2)) == payload
