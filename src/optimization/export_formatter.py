"""Format MPC MILP dispatch output as a backend-ready JSON payload."""

from __future__ import annotations

import json, math
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from numbers import Number
from typing import Any
from uuid import uuid4

import pandas as pd

SCHEMA_VERSION = "1.0"
DEFAULT_APPROACH_NAME = "mpc_milp"
DEFAULT_TIMEZONE = "UTC"
FLOAT_COLUMN_NAMES = (
    "demand_th", "price_el", "hp_el_in", "hp_th_out", "boiler_th_out", "chp_el_out",
    "chp_th_out", "storage_charge", "storage_discharge", "storage_soc", "heat_slack",
)
FLOAT_COLUMNS: dict[str, float] = {name: 0.0 for name in FLOAT_COLUMN_NAMES}
BOOL_COLUMNS: dict[str, bool] = {"hp_on": False, "boiler_on": False, "chp_on": False}
OPTIONAL_COLUMNS: tuple[str, ...] = FLOAT_COLUMN_NAMES + tuple(BOOL_COLUMNS)

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

def _safe_int(value: Any, default: int = 0) -> int:
    return int(_safe_float(value, float(default)))

def _safe_str(value: Any, default: str = "") -> str:
    converted = _to_jsonable(value)
    return default if converted is None else str(converted)

def _get(mapping: Mapping[str, Any], key: str, default: Any = None) -> Any:
    return mapping.get(key, default) if isinstance(mapping, Mapping) else default

def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}

def _pick(*values: Any, default: Any = None) -> Any:
    return next((value for value in values if value is not None), default)

def _row_value(row: pd.Series, column: str) -> Any:
    return row[column] if column in row.index else FLOAT_COLUMNS.get(column, BOOL_COLUMNS.get(column))

def _infer_dt_hours(dispatch_df: pd.DataFrame, metadata: Mapping[str, Any]) -> float:
    approach = _as_mapping(_get(metadata, "approach"))
    dt_hours = _safe_float(_pick(_get(metadata, "dt_hours"), _get(approach, "dt_hours")), 0.0)
    if dt_hours > 0:
        return dt_hours
    if isinstance(dispatch_df.index, pd.DatetimeIndex) and len(dispatch_df.index) > 1:
        hours = (dispatch_df.index[1] - dispatch_df.index[0]).total_seconds() / 3600
        if math.isfinite(hours) and hours > 0:
            return float(hours)
    return 1.0

def _timestamp_for_row(row: pd.Series, index_value: Any, step: int, metadata: Mapping[str, Any], dt_hours: float) -> str:
    for column in ("timestamp", "time", "datetime"):
        if column in row.index and (timestamp := _serialize_timestamp(row[column])):
            return timestamp
    if not isinstance(index_value, Number) and (timestamp := _serialize_timestamp(index_value)):
        return timestamp
    time_window = _as_mapping(_get(metadata, "time_window"))
    start = _pick(_get(metadata, "start"), _get(time_window, "start"))
    try:
        parsed_start = pd.Timestamp(start)
    except (TypeError, ValueError, OverflowError):
        return ""
    return "" if pd.isna(parsed_start) else (parsed_start + pd.Timedelta(hours=step * dt_hours)).isoformat()

def _build_dispatch_row(row: pd.Series, step: int, timestamp: str) -> dict[str, Any]:
    return {
        "timestamp": timestamp, "step": step,
        "demand_mw_th": _safe_float(_row_value(row, "demand_th")),
        "price_eur_per_mwh_el": _safe_float(_row_value(row, "price_el")),
        "heat_pump": {"on": _safe_bool(_row_value(row, "hp_on")), "el_in_mw": _safe_float(_row_value(row, "hp_el_in")), "th_out_mw": _safe_float(_row_value(row, "hp_th_out"))},
        "boiler": {"on": _safe_bool(_row_value(row, "boiler_on")), "th_out_mw": _safe_float(_row_value(row, "boiler_th_out"))},
        "chp": {"on": _safe_bool(_row_value(row, "chp_on")), "el_out_mw": _safe_float(_row_value(row, "chp_el_out")), "th_out_mw": _safe_float(_row_value(row, "chp_th_out"))},
        "storage": {"charge_mw_th": _safe_float(_row_value(row, "storage_charge")), "discharge_mw_th": _safe_float(_row_value(row, "storage_discharge")), "soc_mwh_th": _safe_float(_row_value(row, "storage_soc"))},
        "heat_slack_mw_th": _safe_float(_row_value(row, "heat_slack")),
    }

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

def prepare_optimization_export(dispatch_df: pd.DataFrame, metadata: dict[str, Any], solver_info: dict[str, Any], initial_state: dict[str, Any], next_state: dict[str, Any]) -> dict[str, Any]:
    """Prepare a backend API export payload from optimization dispatch output."""
    warnings: list[str] = []
    metadata_map = _as_mapping(metadata)
    solver_map = _as_mapping(solver_info)
    next_state_map = _as_mapping(next_state)

    for name, value in {"metadata": metadata, "solver_info": solver_info, "initial_state": initial_state, "next_state": next_state}.items():
        if not isinstance(value, Mapping):
            warnings.append(f"{name} was not a mapping; defaults were used.")
    if isinstance(dispatch_df, pd.DataFrame):
        dispatch_df = dispatch_df.copy()
    else:
        warnings.append("dispatch_df was not a pandas DataFrame; dispatch is empty.")
        dispatch_df = pd.DataFrame()
    for column in OPTIONAL_COLUMNS:
        if column not in dispatch_df.columns:
            default = FLOAT_COLUMNS.get(column, BOOL_COLUMNS.get(column))
            warnings.append(f"Missing optional dispatch column '{column}'; defaulted to {default!r}.")

    dt_hours = _infer_dt_hours(dispatch_df, metadata_map)
    dispatch: list[dict[str, Any]] = []
    timestamp_warning_added = False
    for step, (index_value, row) in enumerate(dispatch_df.iterrows()):
        timestamp = _timestamp_for_row(row, index_value, step, metadata_map, dt_hours)
        if not timestamp and not timestamp_warning_added:
            warnings.append("No timestamp source found; timestamps defaulted to ''.")
            timestamp_warning_added = True
        dispatch.append(_build_dispatch_row(row, step, timestamp))
    if dispatch_df.empty:
        warnings.append("dispatch_df is empty; exported an empty dispatch list.")

    approach = _as_mapping(_get(metadata_map, "approach"))
    summary = _as_mapping(_get(metadata_map, "summary"))
    time_window = _as_mapping(_get(metadata_map, "time_window"))
    horizon_default = round(len(dispatch_df) * dt_hours)
    run_id = _get(metadata_map, "run_id")
    if run_id is None:
        run_id = str(uuid4())
        warnings.append("metadata.run_id was missing; generated a UUID run_id.")

    start = _serialize_timestamp(_pick(_get(time_window, "start"), _get(metadata_map, "start")))
    end = _serialize_timestamp(_pick(_get(time_window, "end"), _get(metadata_map, "end")))
    if not start and dispatch:
        start = dispatch[0]["timestamp"]
    if not end and dispatch:
        try:
            end = (pd.Timestamp(dispatch[-1]["timestamp"]) + pd.Timedelta(hours=dt_hours)).isoformat()
        except (TypeError, ValueError, OverflowError):
            end = dispatch[-1]["timestamp"]
    if not start:
        warnings.append("time_window.start could not be inferred; defaulted to ''.")
    if not end:
        warnings.append("time_window.end could not be inferred; defaulted to ''.")

    objective_cost = _pick(_get(summary, "objective_cost_eur"), _get(metadata_map, "objective_cost_eur"), _get(solver_map, "objective_cost_eur"), default=0.0)
    real_cost = _pick(_get(summary, "real_cost_eur"), _get(metadata_map, "real_cost_eur"), _get(solver_map, "real_cost_eur"), default=objective_cost)
    runtime = _pick(_get(summary, "runtime_seconds"), _get(metadata_map, "runtime_seconds"), _get(solver_map, "runtime_seconds"), default=0.0)
    status = _pick(_get(solver_map, "status"), _get(metadata_map, "status"), default="unknown")
    solver = _pick(_get(summary, "solver"), _get(metadata_map, "solver"), _get(solver_map, "solver"), default="unknown")
    termination = _pick(_get(summary, "termination_condition"), _get(metadata_map, "termination_condition"), _get(solver_map, "termination_condition"), default="unknown")

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": _safe_str(run_id),
        "status": _safe_str(status, "unknown"),
        "approach": {
            "name": _safe_str(_pick(_get(approach, "name"), _get(metadata_map, "approach_name")), DEFAULT_APPROACH_NAME),
            "solve_horizon_hours": _safe_int(_pick(_get(approach, "solve_horizon_hours"), _get(metadata_map, "solve_horizon_hours")), horizon_default),
            "commit_horizon_hours": _safe_int(_pick(_get(approach, "commit_horizon_hours"), _get(metadata_map, "commit_horizon_hours")), horizon_default),
            "dt_hours": dt_hours,
        },
        "time_window": {"start": start, "end": end, "timezone": _safe_str(_pick(_get(time_window, "timezone"), _get(metadata_map, "timezone")), DEFAULT_TIMEZONE)},
        "summary": {"objective_cost_eur": _safe_float(objective_cost), "real_cost_eur": _safe_float(real_cost), "runtime_seconds": _safe_float(runtime), "solver": _safe_str(solver, "unknown"), "termination_condition": _safe_str(termination, "unknown")},
        "dispatch": dispatch,
        "next_initial_state": {"soc_mwh_th": _safe_float(_get(next_state_map, "soc_mwh_th", 0.0)), "heat_pump_on": _safe_bool(_get(next_state_map, "heat_pump_on", False)), "boiler_on": _safe_bool(_get(next_state_map, "boiler_on", False)), "chp_on": _safe_bool(_get(next_state_map, "chp_on", False)), "boiler_time_in_state_steps": _safe_int(_get(next_state_map, "boiler_time_in_state_steps", 0)), "chp_time_in_state_steps": _safe_int(_get(next_state_map, "chp_time_in_state_steps", 0))},
        "diagnostics": {"notes": _safe_str(_pick(_get(metadata_map, "notes"), _get(solver_map, "notes"))), "warnings": list(dict.fromkeys(warnings))},
    }
    # FastAPI: return this dict directly or bind it to a response model.
    # Kafka: publish export_to_json(payload, indent=None) for stable encoding.
    # Schema versioning: bump SCHEMA_VERSION only for backend contract changes.
    # Protobuf migration: keep dict keys aligned with future message fields.
    return _to_jsonable(payload)

if __name__ == "__main__":
    index = pd.date_range(datetime(2026, 1, 1, tzinfo=timezone.utc), periods=2, freq="h")
    dispatch = pd.DataFrame({"demand_th": [8.0, 8.4], "price_el": [92.5, 88.0], "hp_el_in": [1.2, 1.1], "hp_th_out": [3.6, 3.3], "boiler_th_out": [0.0, 0.0], "chp_el_out": [1.0, 1.0], "chp_th_out": [4.0, 4.0], "storage_discharge": [0.4, 1.1], "storage_soc": [11.6, 10.5], "hp_on": [True, True], "boiler_on": [False, False], "chp_on": [True, True]}, index=index)
    payload = prepare_optimization_export(
        dispatch_df=dispatch,
        metadata={"run_id": "run-20260101-0000", "status": "optimal", "approach": {"solve_horizon_hours": 24, "commit_horizon_hours": 2, "dt_hours": 1}, "time_window": {"start": index[0], "end": index[-1] + timedelta(hours=1), "timezone": "UTC"}, "objective_cost_eur": 1250.75, "real_cost_eur": 381.42},
        solver_info={"solver": "highs", "runtime_seconds": 3.84, "termination_condition": "optimal"},
        initial_state={},
        next_state={"soc_mwh_th": 10.5, "heat_pump_on": True, "chp_on": True},
    )
    print(export_to_json(payload))
