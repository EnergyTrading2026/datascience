"""Format MPC MILP dispatch output as a backend-ready JSON payload.

Schema v2.0 contract
====================

The backend export wraps each family in an array keyed by stable asset id, so
plants with N>1 heat pumps / boilers / CHPs / storages serialize without losing
per-unit granularity:

    "dispatch": [
        {
            "timestamp": "...",
            "step": 0,
            "demand_mw_th": ...,
            "price_eur_per_mwh_el": ...,
            "heat_pumps": [{"id": "hp_1", "on": ..., "el_in_mw": ..., "th_out_mw": ...}, ...],
            "boilers":    [{"id": "...",  "on": ..., "th_out_mw": ...}, ...],
            "chps":       [{"id": "...",  "on": ..., "el_out_mw": ..., "th_out_mw": ...}, ...],
            "storages":   [{"id": "...",  "charge_mw_th": ..., "discharge_mw_th": ..., "soc_mwh_th": ...}, ...],
            "heat_slack_mw_th": ...
        }
    ],
    "next_initial_state": {
        "units":    {"<asset_id>": {"on": bool, "time_in_state_steps": int}, ...},
        "storages": {"<storage_id>": {"soc_mwh_th": float}, ...}
    }

Application-Repo consumers must iterate the family arrays / dicts; the v1.0
singular shape (heat_pump/boiler/chp/storage as scalars) is gone.

prepare_optimization_export now expects ``dispatch_records`` as an already-
structured list of per-step dicts. Run.py's _dispatch_records() builds these
directly from the solved Pyomo model — no wide-DataFrame intermediate, because
the column space would have been dynamic in asset_id space.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from datetime import datetime, timezone
from numbers import Number
from typing import Any
from uuid import uuid4

import pandas as pd

SCHEMA_VERSION = "2.0"
DEFAULT_APPROACH_NAME = "mpc_milp"
DEFAULT_TIMEZONE = "UTC"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return float(default)
    except (TypeError, ValueError):
        pass
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _safe_bool(value: Any, default: bool = False) -> bool:
    try:
        if value is None or pd.isna(value):
            return bool(default)
    except (TypeError, ValueError):
        pass
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if value in {"0", "false", "f", "no", "n", "off", ""}:
            return False
        return bool(default)
    if isinstance(value, Number):
        return bool(_safe_float(value, 1.0 if default else 0.0))
    try:
        return bool(value)
    except (TypeError, ValueError):
        return bool(default)


def _safe_int(value: Any, default: int = 0) -> int:
    return int(_safe_float(value, float(default)))


def _safe_str(value: Any, default: str = "") -> str:
    converted = _to_jsonable(value)
    return default if converted is None else str(converted)


def _serialize_timestamp(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError):
        return str(value)
    return "" if pd.isna(parsed) else parsed.isoformat()


def _get(mapping: Mapping[str, Any], key: str, default: Any = None) -> Any:
    return mapping.get(key, default) if isinstance(mapping, Mapping) else default


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _pick(*values: Any, default: Any = None) -> Any:
    return next((value for value in values if value is not None), default)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return _serialize_timestamp(value)
    if isinstance(value, pd.Timedelta):
        return value.total_seconds()
    for method in ("item", "tolist"):
        if hasattr(value, method):
            try:
                return _to_jsonable(getattr(value, method)())
            except (TypeError, ValueError):
                pass
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return value if not isinstance(value, float) or math.isfinite(value) else None


def export_to_json(payload: Mapping[str, Any], *, indent: int | None = 2) -> str:
    return json.dumps(_to_jsonable(payload), indent=indent, default=lambda value: str(_to_jsonable(value)))


# v2.0 per-row family-array shapes. Used to coerce caller-supplied records
# defensively (caller may pass numpy scalars, native Python, NaN, ...).
_HP_FIELDS: tuple[str, ...] = ("on", "el_in_mw", "th_out_mw")
_BOILER_FIELDS: tuple[str, ...] = ("on", "th_out_mw")
_CHP_FIELDS: tuple[str, ...] = ("on", "el_out_mw", "th_out_mw")
_STORAGE_FIELDS: tuple[str, ...] = ("charge_mw_th", "discharge_mw_th", "soc_mwh_th")


def _coerce_unit_entry(entry: Any, fields: tuple[str, ...]) -> dict[str, Any]:
    """Coerce one element of a family array. Always emits id (str) plus each
    field — booleans for ``on``, floats for the rest."""
    src = _as_mapping(entry)
    out: dict[str, Any] = {"id": _safe_str(_get(src, "id"))}
    for field in fields:
        raw = _get(src, field)
        if field == "on":
            out[field] = _safe_bool(raw)
        else:
            out[field] = _safe_float(raw)
    return out


def _coerce_storage_entry(entry: Any) -> dict[str, Any]:
    src = _as_mapping(entry)
    out: dict[str, Any] = {"id": _safe_str(_get(src, "id"))}
    for field in _STORAGE_FIELDS:
        out[field] = _safe_float(_get(src, field))
    return out


def _build_dispatch_row(record: Mapping[str, Any], step: int, fallback_ts: str) -> dict[str, Any]:
    """Coerce one caller-supplied dispatch record into the v2.0 row shape."""
    timestamp = _serialize_timestamp(_get(record, "timestamp")) or fallback_ts
    return {
        "timestamp": timestamp,
        "step": _safe_int(_get(record, "step"), step),
        "demand_mw_th": _safe_float(_get(record, "demand_mw_th")),
        "price_eur_per_mwh_el": _safe_float(_get(record, "price_eur_per_mwh_el")),
        "heat_pumps": [
            _coerce_unit_entry(item, _HP_FIELDS)
            for item in _get(record, "heat_pumps", []) or []
        ],
        "boilers": [
            _coerce_unit_entry(item, _BOILER_FIELDS)
            for item in _get(record, "boilers", []) or []
        ],
        "chps": [
            _coerce_unit_entry(item, _CHP_FIELDS)
            for item in _get(record, "chps", []) or []
        ],
        "storages": [
            _coerce_storage_entry(item)
            for item in _get(record, "storages", []) or []
        ],
        "heat_slack_mw_th": _safe_float(_get(record, "heat_slack_mw_th")),
    }


def _coerce_state(state: Any) -> dict[str, Any]:
    """v2.0 next_initial_state shape: units{} for unit-commitment assets,
    storages{} for thermal storages. Mirrors DispatchState on the DS side."""
    src = _as_mapping(state)
    units_in = _as_mapping(_get(src, "units"))
    storages_in = _as_mapping(_get(src, "storages"))
    units_out: dict[str, Any] = {}
    for asset_id, payload in units_in.items():
        p = _as_mapping(payload)
        units_out[_safe_str(asset_id)] = {
            "on": _safe_bool(_get(p, "on")),
            "time_in_state_steps": _safe_int(_get(p, "time_in_state_steps")),
        }
    storages_out: dict[str, Any] = {}
    for storage_id, payload in storages_in.items():
        p = _as_mapping(payload)
        storages_out[_safe_str(storage_id)] = {
            "soc_mwh_th": _safe_float(_get(p, "soc_mwh_th")),
        }
    return {"units": units_out, "storages": storages_out}


def prepare_optimization_export(
    dispatch_records: list[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    solver_info: Mapping[str, Any],
    initial_state: Mapping[str, Any],
    next_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Prepare a backend API export payload (schema v2.0).

    Args:
        dispatch_records: per-step dicts already structured with family arrays.
            Build via run.py's ``_dispatch_records`` from a solved Pyomo model.
        metadata: run_id, status, approach{name, solve_horizon_hours,
            commit_horizon_hours, dt_hours}, time_window{start, end, timezone},
            objective_cost_eur, real_cost_eur, ...
        solver_info: solver, runtime_seconds, termination_condition, status.
        initial_state, next_state: DispatchState-shaped dicts with
            ``units{<id>: {on, time_in_state_steps}}`` and
            ``storages{<id>: {soc_mwh_th}}``.
    """
    warnings: list[str] = []
    metadata_map = _as_mapping(metadata)
    solver_map = _as_mapping(solver_info)

    for name, value in {
        "metadata": metadata,
        "solver_info": solver_info,
        "initial_state": initial_state,
        "next_state": next_state,
    }.items():
        if not isinstance(value, Mapping):
            warnings.append(f"{name} was not a mapping; defaults were used.")

    if not isinstance(dispatch_records, list):
        warnings.append("dispatch_records was not a list; dispatch is empty.")
        dispatch_records = []

    approach = _as_mapping(_get(metadata_map, "approach"))
    summary = _as_mapping(_get(metadata_map, "summary"))
    time_window = _as_mapping(_get(metadata_map, "time_window"))

    dt_hours = _safe_float(
        _pick(_get(metadata_map, "dt_hours"), _get(approach, "dt_hours")),
        0.0,
    )
    if dt_hours <= 0:
        # Try inferring from the first two record timestamps.
        if len(dispatch_records) >= 2:
            try:
                t0 = pd.Timestamp(_get(dispatch_records[0], "timestamp"))
                t1 = pd.Timestamp(_get(dispatch_records[1], "timestamp"))
                hours = (t1 - t0).total_seconds() / 3600
                if math.isfinite(hours) and hours > 0:
                    dt_hours = float(hours)
            except (TypeError, ValueError, OverflowError):
                pass
        if dt_hours <= 0:
            dt_hours = 1.0

    fallback_start = _serialize_timestamp(
        _pick(_get(time_window, "start"), _get(metadata_map, "start"))
    )

    dispatch: list[dict[str, Any]] = []
    timestamp_warning_added = False
    for step, record in enumerate(dispatch_records):
        if fallback_start:
            try:
                fallback_ts = (
                    pd.Timestamp(fallback_start) + pd.Timedelta(hours=step * dt_hours)
                ).isoformat()
            except (TypeError, ValueError, OverflowError):
                fallback_ts = ""
        else:
            fallback_ts = ""
        row = _build_dispatch_row(record, step, fallback_ts)
        if not row["timestamp"] and not timestamp_warning_added:
            warnings.append("No timestamp source found; timestamps defaulted to ''.")
            timestamp_warning_added = True
        dispatch.append(row)
    if not dispatch_records:
        warnings.append("dispatch_records is empty; exported an empty dispatch list.")

    horizon_default = round(len(dispatch_records) * dt_hours)
    run_id = _get(metadata_map, "run_id")
    if run_id is None:
        run_id = str(uuid4())
        warnings.append("metadata.run_id was missing; generated a UUID run_id.")

    start = _serialize_timestamp(
        _pick(_get(time_window, "start"), _get(metadata_map, "start"))
    )
    end = _serialize_timestamp(
        _pick(_get(time_window, "end"), _get(metadata_map, "end"))
    )
    if not start and dispatch:
        start = dispatch[0]["timestamp"]
    if not end and dispatch:
        try:
            end = (
                pd.Timestamp(dispatch[-1]["timestamp"]) + pd.Timedelta(hours=dt_hours)
            ).isoformat()
        except (TypeError, ValueError, OverflowError):
            end = dispatch[-1]["timestamp"]
    if not start:
        warnings.append("time_window.start could not be inferred; defaulted to ''.")
    if not end:
        warnings.append("time_window.end could not be inferred; defaulted to ''.")

    objective_cost = _pick(
        _get(summary, "objective_cost_eur"),
        _get(metadata_map, "objective_cost_eur"),
        _get(solver_map, "objective_cost_eur"),
        default=0.0,
    )
    real_cost = _pick(
        _get(summary, "real_cost_eur"),
        _get(metadata_map, "real_cost_eur"),
        _get(solver_map, "real_cost_eur"),
        default=objective_cost,
    )
    runtime = _pick(
        _get(summary, "runtime_seconds"),
        _get(metadata_map, "runtime_seconds"),
        _get(solver_map, "runtime_seconds"),
        default=0.0,
    )
    status = _pick(
        _get(solver_map, "status"), _get(metadata_map, "status"), default="unknown"
    )
    solver = _pick(
        _get(summary, "solver"),
        _get(metadata_map, "solver"),
        _get(solver_map, "solver"),
        default="unknown",
    )
    termination = _pick(
        _get(summary, "termination_condition"),
        _get(metadata_map, "termination_condition"),
        _get(solver_map, "termination_condition"),
        default="unknown",
    )

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": _safe_str(run_id),
        "status": _safe_str(status, "unknown"),
        "approach": {
            "name": _safe_str(
                _pick(_get(approach, "name"), _get(metadata_map, "approach_name")),
                DEFAULT_APPROACH_NAME,
            ),
            "solve_horizon_hours": _safe_int(
                _pick(
                    _get(approach, "solve_horizon_hours"),
                    _get(metadata_map, "solve_horizon_hours"),
                ),
                horizon_default,
            ),
            "commit_horizon_hours": _safe_int(
                _pick(
                    _get(approach, "commit_horizon_hours"),
                    _get(metadata_map, "commit_horizon_hours"),
                ),
                horizon_default,
            ),
            "dt_hours": dt_hours,
        },
        "time_window": {
            "start": start,
            "end": end,
            "timezone": _safe_str(
                _pick(_get(time_window, "timezone"), _get(metadata_map, "timezone")),
                DEFAULT_TIMEZONE,
            ),
        },
        "summary": {
            "objective_cost_eur": _safe_float(objective_cost),
            "real_cost_eur": _safe_float(real_cost),
            "runtime_seconds": _safe_float(runtime),
            "solver": _safe_str(solver, "unknown"),
            "termination_condition": _safe_str(termination, "unknown"),
        },
        "dispatch": dispatch,
        "initial_state": _coerce_state(initial_state),
        "next_initial_state": _coerce_state(next_state),
        "diagnostics": {
            "notes": _safe_str(
                _pick(_get(metadata_map, "notes"), _get(solver_map, "notes"))
            ),
            "warnings": list(dict.fromkeys(warnings)),
        },
    }
    return _to_jsonable(payload)


if __name__ == "__main__":
    # Smoke demo: 2 HPs, 1 boiler, 1 CHP, 1 storage, 2 hourly steps.
    index = pd.date_range(datetime(2026, 1, 1, tzinfo=timezone.utc), periods=2, freq="h")
    records = [
        {
            "timestamp": index[step],
            "step": step,
            "demand_mw_th": 8.0 + 0.4 * step,
            "price_eur_per_mwh_el": 92.5 - 4.5 * step,
            "heat_pumps": [
                {"id": "hp_1", "on": True, "el_in_mw": 1.2, "th_out_mw": 3.6},
                {"id": "hp_2", "on": False, "el_in_mw": 0.0, "th_out_mw": 0.0},
            ],
            "boilers": [{"id": "boiler", "on": False, "th_out_mw": 0.0}],
            "chps": [{"id": "chp", "on": True, "el_out_mw": 1.0, "th_out_mw": 4.0}],
            "storages": [
                {
                    "id": "storage",
                    "charge_mw_th": 0.0,
                    "discharge_mw_th": 0.4 + 0.7 * step,
                    "soc_mwh_th": 11.6 - 1.1 * step,
                }
            ],
            "heat_slack_mw_th": 0.0,
        }
        for step in range(2)
    ]
    payload = prepare_optimization_export(
        dispatch_records=records,
        metadata={
            "run_id": "run-20260101-0000",
            "status": "optimal",
            "approach": {
                "name": "mpc_milp",
                "solve_horizon_hours": 24,
                "commit_horizon_hours": 2,
                "dt_hours": 1,
            },
            "time_window": {
                "start": index[0],
                "end": index[-1] + pd.Timedelta(hours=1),
                "timezone": "UTC",
            },
            "objective_cost_eur": 1250.75,
            "real_cost_eur": 381.42,
        },
        solver_info={"solver": "highs", "runtime_seconds": 3.84, "termination_condition": "optimal"},
        initial_state={"units": {}, "storages": {}},
        next_state={
            "units": {
                "hp_1": {"on": True, "time_in_state_steps": 8},
                "hp_2": {"on": False, "time_in_state_steps": 16},
                "boiler": {"on": False, "time_in_state_steps": 32},
                "chp": {"on": True, "time_in_state_steps": 4},
            },
            "storages": {"storage": {"soc_mwh_th": 10.5}},
        },
    )
    print(export_to_json(payload))
