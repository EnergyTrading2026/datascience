"""Backend export formatter regressions (schema v2.0)."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from optimization.export_formatter import (
    SCHEMA_VERSION,
    export_to_json,
    prepare_optimization_export,
)


def _single_asset_records() -> list[dict[str, object]]:
    """Two timesteps, one asset per family. Mirrors the legacy_default plant."""
    idx = pd.date_range("2026-01-01 13:00:00", periods=2, freq="15min", tz="Europe/Berlin")
    return [
        {
            "timestamp": idx[0],
            "step": 0,
            "demand_mw_th": 10.0,
            "price_eur_per_mwh_el": 60.0,
            "heat_pumps": [{"id": "hp", "on": True, "el_in_mw": 1.0, "th_out_mw": 3.5}],
            "boilers": [{"id": "boiler", "on": False, "th_out_mw": 0.0}],
            "chps": [{"id": "chp", "on": True, "el_out_mw": 2.0, "th_out_mw": 2.4}],
            "storages": [
                {
                    "id": "storage",
                    "charge_mw_th": 0.0,
                    "discharge_mw_th": 4.1,
                    "soc_mwh_th": 198.9,
                }
            ],
            "heat_slack_mw_th": 0.0,
        },
        {
            "timestamp": idx[1],
            "step": 1,
            "demand_mw_th": 11.0,
            "price_eur_per_mwh_el": 62.5,
            "heat_pumps": [{"id": "hp", "on": True, "el_in_mw": 1.5, "th_out_mw": 5.25}],
            "boilers": [{"id": "boiler", "on": True, "th_out_mw": 2.0}],
            "chps": [{"id": "chp", "on": False, "el_out_mw": 0.0, "th_out_mw": 0.0}],
            "storages": [
                {
                    "id": "storage",
                    "charge_mw_th": 1.25,
                    "discharge_mw_th": 0.0,
                    "soc_mwh_th": 199.2,
                }
            ],
            "heat_slack_mw_th": 0.0,
        },
    ]


def _base_metadata() -> dict[str, object]:
    return {
        "run_id": "mpc-2026-01-01T13:00:00+01:00",
        "status": "optimal",
        "approach": {
            "name": "hourly_mpc_milp",
            "solve_horizon_hours": 35,
            "commit_horizon_hours": 1,
            "dt_hours": 0.25,
        },
        "time_window": {
            "start": pd.Timestamp("2026-01-01 13:00:00", tz="Europe/Berlin"),
            "end": pd.Timestamp("2026-01-01 13:30:00", tz="Europe/Berlin"),
            "timezone": "Europe/Berlin",
        },
        "objective_cost_eur": 123.45,
        "real_cost_eur": 67.89,
    }


def _base_solver_info() -> dict[str, object]:
    return {
        "solver": "appsi_highs",
        "runtime_seconds": 0.75,
        "termination_condition": "optimal",
        "status": "optimal",
    }


def test_schema_version_is_v2():
    assert SCHEMA_VERSION == "2.0"


def test_prepare_optimization_export_v2_single_asset():
    payload = prepare_optimization_export(
        dispatch_records=_single_asset_records(),
        metadata=_base_metadata(),
        solver_info=_base_solver_info(),
        initial_state={"units": {}, "storages": {}},
        next_state={
            "units": {
                "hp": {"on": True, "time_in_state_steps": 4},
                "boiler": {"on": True, "time_in_state_steps": 1},
                "chp": {"on": False, "time_in_state_steps": 0},
            },
            "storages": {"storage": {"soc_mwh_th": 199.2}},
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
        "initial_state",
        "next_initial_state",
        "diagnostics",
    }
    assert payload["schema_version"] == "2.0"
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

    assert len(payload["dispatch"]) == 2
    step0 = payload["dispatch"][0]
    assert step0["timestamp"] == "2026-01-01T13:00:00+01:00"
    assert step0["step"] == 0
    assert step0["demand_mw_th"] == 10.0
    assert step0["price_eur_per_mwh_el"] == 60.0
    assert step0["heat_pumps"] == [
        {"id": "hp", "on": True, "el_in_mw": 1.0, "th_out_mw": 3.5}
    ]
    assert step0["boilers"] == [{"id": "boiler", "on": False, "th_out_mw": 0.0}]
    assert step0["chps"] == [
        {"id": "chp", "on": True, "el_out_mw": 2.0, "th_out_mw": 2.4}
    ]
    assert step0["storages"] == [
        {
            "id": "storage",
            "charge_mw_th": 0.0,
            "discharge_mw_th": 4.1,
            "soc_mwh_th": 198.9,
        }
    ]
    assert step0["heat_slack_mw_th"] == 0.0

    assert payload["initial_state"] == {"units": {}, "storages": {}}
    assert payload["next_initial_state"] == {
        "units": {
            "hp": {"on": True, "time_in_state_steps": 4},
            "boiler": {"on": True, "time_in_state_steps": 1},
            "chp": {"on": False, "time_in_state_steps": 0},
        },
        "storages": {"storage": {"soc_mwh_th": 199.2}},
    }
    assert payload["diagnostics"] == {"notes": "", "warnings": []}
    assert json.loads(export_to_json(payload, indent=2)) == payload


def test_prepare_optimization_export_v2_multi_asset_arrays_per_family():
    """2 HPs + 2 storages: every family entry is preserved per asset id."""
    idx = pd.date_range("2026-01-01 13:00:00", periods=1, freq="15min", tz="Europe/Berlin")
    records = [
        {
            "timestamp": idx[0],
            "step": 0,
            "demand_mw_th": 12.0,
            "price_eur_per_mwh_el": 80.0,
            "heat_pumps": [
                {"id": "hp_1", "on": True, "el_in_mw": 1.0, "th_out_mw": 3.5},
                {"id": "hp_2", "on": False, "el_in_mw": 0.0, "th_out_mw": 0.0},
            ],
            "boilers": [{"id": "boiler", "on": True, "th_out_mw": 5.0}],
            "chps": [],
            "storages": [
                {
                    "id": "tank_main",
                    "charge_mw_th": 0.0,
                    "discharge_mw_th": 3.5,
                    "soc_mwh_th": 100.0,
                },
                {
                    "id": "tank_aux",
                    "charge_mw_th": 0.0,
                    "discharge_mw_th": 0.0,
                    "soc_mwh_th": 50.0,
                },
            ],
            "heat_slack_mw_th": 0.0,
        }
    ]
    payload = prepare_optimization_export(
        dispatch_records=records,
        metadata=_base_metadata(),
        solver_info=_base_solver_info(),
        initial_state={
            "units": {
                "hp_1": {"on": False, "time_in_state_steps": 999},
                "hp_2": {"on": False, "time_in_state_steps": 999},
                "boiler": {"on": False, "time_in_state_steps": 999},
            },
            "storages": {
                "tank_main": {"soc_mwh_th": 100.0},
                "tank_aux": {"soc_mwh_th": 50.0},
            },
        },
        next_state={
            "units": {
                "hp_1": {"on": True, "time_in_state_steps": 1},
                "hp_2": {"on": False, "time_in_state_steps": 1000},
                "boiler": {"on": True, "time_in_state_steps": 1},
            },
            "storages": {
                "tank_main": {"soc_mwh_th": 96.5},
                "tank_aux": {"soc_mwh_th": 50.0},
            },
        },
    )

    step = payload["dispatch"][0]
    assert len(step["heat_pumps"]) == 2
    assert {hp["id"] for hp in step["heat_pumps"]} == {"hp_1", "hp_2"}
    assert step["heat_pumps"][0] == {
        "id": "hp_1",
        "on": True,
        "el_in_mw": 1.0,
        "th_out_mw": 3.5,
    }
    assert step["chps"] == []  # zero-CHP family preserved as empty array
    assert len(step["storages"]) == 2
    assert {s["id"] for s in step["storages"]} == {"tank_main", "tank_aux"}

    assert set(payload["next_initial_state"]["units"]) == {"hp_1", "hp_2", "boiler"}
    assert set(payload["next_initial_state"]["storages"]) == {"tank_main", "tank_aux"}
    assert payload["next_initial_state"]["units"]["hp_1"] == {
        "on": True,
        "time_in_state_steps": 1,
    }


def test_empty_dispatch_records_emits_warning():
    payload = prepare_optimization_export(
        dispatch_records=[],
        metadata=_base_metadata(),
        solver_info=_base_solver_info(),
        initial_state={"units": {}, "storages": {}},
        next_state={"units": {}, "storages": {}},
    )
    assert payload["dispatch"] == []
    assert any(
        "dispatch_records is empty" in w for w in payload["diagnostics"]["warnings"]
    )


@pytest.mark.parametrize(
    "bad_input",
    [None, "not a list", 42, {"some": "dict"}],
)
def test_non_list_dispatch_records_warns_and_returns_empty(bad_input):
    payload = prepare_optimization_export(
        dispatch_records=bad_input,  # type: ignore[arg-type]
        metadata=_base_metadata(),
        solver_info=_base_solver_info(),
        initial_state={"units": {}, "storages": {}},
        next_state={"units": {}, "storages": {}},
    )
    assert payload["dispatch"] == []
    assert any(
        "dispatch_records was not a list" in w
        for w in payload["diagnostics"]["warnings"]
    )
