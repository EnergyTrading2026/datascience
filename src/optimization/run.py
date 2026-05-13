"""CLI entrypoint for one hourly MPC dispatch cycle.

Usage (production: live SMARD pull, hourly resolution):
    python -m optimization.run \\
        --solve-time 2026-04-28T13:00:00+02:00 \\
        --forecast-path /shared/forecast/latest.parquet \\
        --state-in     /shared/state/current.json \\
        --state-out    /shared/state/current.json \\
        --dispatch-out /shared/dispatch/<solve_time>.parquet \\
        --export-out   /shared/export/<solve_time>.json

Backtest / replay against a historical SMARD CSV:
        --prices-source csv --prices-path /shared/smard/latest.csv

Quarter-hour resolution (requires QH demand forecast and QH-DA-aware backtest):
        --resolution quarterhour

Exit codes:
    0 = success, dispatch + state written
    1 = recoverable failure (forecast/price file missing, horizon too short,
        SMARD API unreachable) — caller may retry next cycle
    2 = solver infeasible — needs investigation
    3 = unexpected error (incl. SMARD schema break)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from optimization.adapters import forecast as forecast_io
from optimization.adapters import smard as smard_io
from optimization.adapters import smard_live as smard_live_io
from optimization.config import PlantConfig, RuntimeConfig
from optimization.dispatch import extract_dispatch, extract_state
from optimization.export_formatter import export_to_json, prepare_optimization_export
from optimization.model import build_model
from optimization.solve import SolverInfeasibleError, solve
from optimization.state import DispatchState

logger = logging.getLogger("optimization.run")

INT_PER_HOUR = 4
_SLOTS_PER_HOUR: dict[str, int] = {"hour": 1, "quarterhour": 4}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one hourly MPC dispatch cycle.")
    p.add_argument(
        "--solve-time",
        required=True,
        type=lambda s: pd.Timestamp(s),
        help="Solve timestamp, ISO8601 with timezone (e.g. 2026-04-28T13:00:00+02:00).",
    )
    p.add_argument("--forecast-path", required=True, type=Path)
    p.add_argument(
        "--prices-source",
        choices=["live", "csv"],
        default="live",
        help="Where to get DA prices. 'live' pulls from SMARD chart_data API "
             "(production default). 'csv' reads a SMARD-format CSV (backtest).",
    )
    p.add_argument(
        "--prices-path",
        type=Path,
        help="SMARD CSV path (semicolon-separated). Required when "
             "--prices-source=csv; ignored otherwise.",
    )
    p.add_argument(
        "--resolution",
        choices=["hour", "quarterhour"],
        default="quarterhour",
        help="Model resolution. Default 'quarterhour' = production setup: "
             "live 15-min DA prices from SMARD paired with hourly demand "
             "forecast (forward-filled to 15-min).",
    )
    p.add_argument(
        "--forecast-resolution",
        choices=["hour", "quarterhour"],
        default="hour",
        help="Resolution of the forecast file. Default 'hour' (= what the "
             "forecasting team delivers). With --resolution=quarterhour this "
             "triggers forward-fill of demand to 15-min.",
    )
    p.add_argument(
        "--state-in",
        type=Path,
        help="Path to read carry-over state. Required unless --cold-start.",
    )
    p.add_argument("--state-out", required=True, type=Path)
    p.add_argument("--dispatch-out", required=True, type=Path)
    p.add_argument(
        "--export-out",
        type=Path,
        help="Optional backend API JSON payload path.",
    )
    p.add_argument(
        "--cold-start",
        action="store_true",
        help="Ignore --state-in and start with default state (first deployment).",
    )
    p.add_argument(
        "--config-file",
        type=Path,
        default=None,
        help="Path to plant_config.json. Defaults to PlantConfig.legacy_default() "
             "baked in code if omitted.",
    )
    return p.parse_args(argv)


def _upsample_hour_to_quarterhour(s: pd.Series) -> pd.Series:
    """Forward-fill an hourly Series to 15-min resolution: each hour -> 4 slots.

    For thermally smooth district-heating demand, replicating the hourly value
    across the four 15-min sub-slots is a defensible approximation (network and
    building thermal inertia smooth out sub-hour variation). Lets the model see
    real intra-hour price spreads while keeping the existing forecast contract.
    """
    if len(s) == 0:
        return s
    new_idx = pd.date_range(
        start=s.index[0],
        periods=len(s) * INT_PER_HOUR,
        freq="15min",
        tz=s.index.tz,
    )
    return pd.Series(np.repeat(s.to_numpy(), INT_PER_HOUR), index=new_idx, name=s.name)


def _v(component: object, asset_id: str, t: int) -> float:
    """Return a solved Pyomo (asset_id, t)-indexed scalar value as float."""
    return float(pyo.value(component[asset_id, t]))  # type: ignore[index]


def _b(component: object, asset_id: str, t: int) -> bool:
    """Return a solved Pyomo (asset_id, t)-indexed binary variable as bool."""
    return bool(round(_v(component, asset_id, t)))


def _value_scalar(component: object, t: int) -> float:
    """Return a solved Pyomo scalar (t-indexed only) value as float."""
    return float(pyo.value(component[t]))  # type: ignore[index]


def _state_payload(state: DispatchState) -> dict[str, object]:
    """Serialize DispatchState into the v2.0 ``{units, storages}`` shape.

    Mirrors the on-disk DispatchState format so the application repo can
    deserialize directly. ``units`` covers HP/boiler/CHP keyed by asset id;
    ``storages`` keeps SoC per storage id.
    """
    return {
        "units": {
            asset_id: {
                "on": bool(unit.on),
                "time_in_state_steps": int(unit.time_in_state_steps),
            }
            for asset_id, unit in state.units.items()
        },
        "storages": {
            storage_id: {"soc_mwh_th": float(s.soc_mwh_th)}
            for storage_id, s in state.storages.items()
        },
    }


def _dispatch_records(
    model: object,
    config: PlantConfig,
    n_intervals: int,
    commit_start: pd.Timestamp,
) -> list[dict[str, object]]:
    """Build v2.0 per-step dispatch records from a solved Pyomo model.

    Each record carries one entry per asset in each family (arrays). With a
    legacy single-asset config the arrays each hold one element; with N>1
    the solver decisions are surfaced per asset id so downstream consumers
    keep per-unit visibility.
    """
    records: list[dict[str, object]] = []
    for t in range(1, n_intervals + 1):
        ts = (commit_start + (t - 1) * pd.Timedelta(minutes=15)).isoformat()
        record: dict[str, object] = {
            "timestamp": ts,
            "step": t - 1,
            "demand_mw_th": _value_scalar(model.demand, t),
            "price_eur_per_mwh_el": _value_scalar(model.da_price, t),
            "heat_pumps": [
                {
                    "id": hp.id,
                    "on": _b(model.z_hp, hp.id, t),
                    "el_in_mw": _v(model.P_hp_el, hp.id, t),
                    "th_out_mw": _v(model.Q_hp, hp.id, t),
                }
                for hp in config.heat_pumps
            ],
            "boilers": [
                {
                    "id": b.id,
                    "on": _b(model.z_boiler, b.id, t),
                    "th_out_mw": _v(model.Q_boiler, b.id, t),
                }
                for b in config.boilers
            ],
            "chps": [
                {
                    "id": c.id,
                    "on": _b(model.z_chp, c.id, t),
                    "el_out_mw": _v(model.P_chp_el, c.id, t),
                    "th_out_mw": _v(model.Q_chp, c.id, t),
                }
                for c in config.chps
            ],
            "storages": [
                {
                    "id": s.id,
                    "charge_mw_th": _v(model.Q_charge, s.id, t),
                    "discharge_mw_th": _v(model.Q_discharge, s.id, t),
                    "soc_mwh_th": _v(model.SoC, s.id, t),
                }
                for s in config.storages
            ],
            "heat_slack_mw_th": 0.0,
        }
        records.append(record)
    return records


def run_one_cycle(
    solve_time: pd.Timestamp,
    forecast_path: Path,
    prices_path: Path | None,
    state_in: Path | None,
    state_out: Path,
    dispatch_out: Path,
    cold_start: bool,
    config: PlantConfig | None = None,
    export_out: Path | None = None,
    runtime: RuntimeConfig | None = None,
    prices_source: str = "live",
    resolution: str = "quarterhour",
    forecast_resolution: str | None = "hour",
) -> int:
    """Run one hourly cycle. Returns exit code (0/1/2/3).

    Args:
        prices_source: 'live' (SMARD chart_data API, prod default) or 'csv'
            (SMARD-format CSV, backtest).
        resolution: model resolution, 'hour' or 'quarterhour'.
        forecast_resolution: resolution of the forecast file. Defaults to
            `resolution`. Allowed mismatch: 'hour' file + 'quarterhour' model
            (demand gets forward-filled). The inverse loses information and is
            rejected.
    """
    if solve_time.tzinfo is None:
        logger.error("solve-time must be tz-aware")
        return 1
    if prices_source not in ("live", "csv"):
        logger.error("prices_source must be 'live' or 'csv'; got %r", prices_source)
        return 1
    if resolution not in _SLOTS_PER_HOUR:
        logger.error("resolution must be 'hour' or 'quarterhour'; got %r", resolution)
        return 1
    forecast_resolution = forecast_resolution or resolution
    if forecast_resolution not in _SLOTS_PER_HOUR:
        logger.error(
            "forecast_resolution must be 'hour' or 'quarterhour'; got %r",
            forecast_resolution,
        )
        return 1
    if forecast_resolution != resolution and not (
        forecast_resolution == "hour" and resolution == "quarterhour"
    ):
        logger.error(
            "unsupported resolution combo: forecast=%s, model=%s. "
            "Allowed: identical, or forecast=hour + model=quarterhour (hybrid).",
            forecast_resolution, resolution,
        )
        return 1
    if prices_source == "csv" and prices_path is None:
        logger.error("--prices-path required when --prices-source=csv")
        return 1
    if prices_source == "csv" and resolution != "hour":
        logger.error(
            "--prices-source csv only supports --resolution hour "
            "(historical SMARD CSV is hourly); got resolution=%s",
            resolution,
        )
        return 1

    config = config or PlantConfig.legacy_default()
    rt = runtime or RuntimeConfig()
    slots_per_hour = _SLOTS_PER_HOUR[resolution]
    hybrid_mode = forecast_resolution != resolution

    # 1. Load state
    if cold_start:
        state = DispatchState.cold_start(config, solve_time)
        total_soc = sum(s.soc_mwh_th for s in state.storages.values())
        logger.info("cold start: total SoC=%.1f, all units off", total_soc)
    else:
        if state_in is None:
            logger.error("--state-in required when not --cold-start")
            return 1
        try:
            state = DispatchState.load(state_in)
        except FileNotFoundError as e:
            logger.error("state file missing: %s", e)
            return 1
        total_soc = sum(s.soc_mwh_th for s in state.storages.values())
        logger.info("loaded state from %s (total SoC=%.1f)", state_in, total_soc)

    # 2. Fetch inputs
    try:
        forecast = forecast_io.load_forecast(
            forecast_path, solve_time, resolution=forecast_resolution
        )
    except (FileNotFoundError, forecast_io.ForecastSchemaError, ValueError) as e:
        logger.error("forecast load failed: %s", e)
        return 1
    try:
        if prices_source == "live":
            prices = smard_live_io.get_published_prices(solve_time, resolution=resolution)
        else:
            prices = smard_io.get_published_prices(solve_time, prices_path)
    except smard_live_io.SmardApiUnavailableError as e:
        logger.error("SMARD live unavailable: %s", e)
        return 1
    except (FileNotFoundError, ValueError) as e:
        logger.error("price load failed: %s", e)
        return 1

    # 2b. Hybrid mode: upsample hourly demand onto the QH grid, then align starts.
    # Both adapters anchor independently (forecast at next hour, prices at next 15-min),
    # so when solve_time is off-hour the two starts disagree by < 1h. We trim to the
    # later of the two starts so model.py sees aligned indices.
    if hybrid_mode:
        forecast = _upsample_hour_to_quarterhour(forecast)
        common_start = max(forecast.index[0], prices.index[0])
        forecast = forecast.loc[common_start:]
        prices = prices.loc[common_start:]
        logger.info(
            "hybrid mode: hourly demand forward-filled to 15-min, common start=%s",
            common_start,
        )

    # 3. Reconcile horizons (in slots; thresholds are configured in hours)
    min_slots = rt.horizon_hours_min * slots_per_hour
    target_slots = rt.horizon_hours_target * slots_per_hour
    horizon_slots = min(len(forecast), len(prices))
    if horizon_slots < min_slots:
        logger.error(
            "horizon too short: forecast=%d slots, prices=%d slots, min=%d slots (%dh @ %s)",
            len(forecast), len(prices), min_slots, rt.horizon_hours_min, resolution,
        )
        return 1
    horizon_slots = min(horizon_slots, target_slots)
    forecast = forecast.iloc[:horizon_slots]
    prices = prices.iloc[:horizon_slots]
    logger.info(
        "horizon=%d slots (%.1fh @ %s); forecast=%d, prices=%d available",
        horizon_slots, horizon_slots / slots_per_hour, resolution,
        len(forecast), len(prices),
    )

    # 4. Build + solve
    try:
        model = build_model(
            forecast, prices, state, config,
            demand_safety_factor=rt.demand_safety_factor,
            resolution=resolution,
        )
        result = solve(model, time_limit_s=rt.solver_time_limit_s, mip_gap=rt.solver_mip_gap)
    except SolverInfeasibleError as e:
        logger.error("solver infeasible: %s", e)
        return 2
    logger.info(
        "solved in %.2fs, status=%s, objective=%.0f EUR",
        result.solve_time_s, result.status, result.objective_eur or 0,
    )

    # 5. Extract committed dispatch + new state
    commit_intervals = rt.commit_hours * INT_PER_HOUR
    dispatch = extract_dispatch(model, n_intervals=commit_intervals, solve_time=solve_time)
    commit_end = forecast.index[0] + pd.Timedelta(hours=rt.commit_hours)
    new_state = extract_state(model, t_end=commit_intervals, commit_end_time=commit_end)

    # 6. Persist (atomic for state; parquet for dispatch; optional backend JSON)
    dispatch_out.parent.mkdir(parents=True, exist_ok=True)
    dispatch.to_dataframe().to_parquet(dispatch_out)
    new_state.save(state_out)
    if export_out is not None:
        export_out.parent.mkdir(parents=True, exist_ok=True)
        records = _dispatch_records(
            model, config, commit_intervals, dispatch.commit_start
        )
        payload = prepare_optimization_export(
            dispatch_records=records,
            metadata={
                "run_id": f"mpc-{solve_time.isoformat()}",
                "status": result.status,
                "approach": {
                    "name": "hourly_mpc_milp",
                    "solve_horizon_hours": int(model._horizon_hours),
                    "commit_horizon_hours": rt.commit_hours,
                    "dt_hours": config.dt_h,
                },
                "time_window": {
                    "start": dispatch.commit_start,
                    "end": commit_end,
                    "timezone": str(dispatch.commit_start.tz),
                },
                "objective_cost_eur": result.objective_eur,
                "real_cost_eur": dispatch.expected_cost_eur,
            },
            solver_info={
                "solver": "appsi_highs",
                "runtime_seconds": result.solve_time_s,
                "termination_condition": result.status,
                "status": result.status,
            },
            initial_state=_state_payload(state),
            next_state=_state_payload(new_state),
        )
        export_out.write_text(export_to_json(payload, indent=2), encoding="utf-8")
        logger.info("wrote backend export -> %s", export_out)
    logger.info("wrote dispatch -> %s, state -> %s", dispatch_out, state_out)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.config_file is not None:
        try:
            config = PlantConfig.from_json(args.config_file)
        except (FileNotFoundError, ValueError) as e:
            logger.error("config load failed (%s): %s", args.config_file, e)
            return 1
        logger.info("loaded plant config from %s", args.config_file)
    else:
        config = None  # run_one_cycle defaults to legacy_default()
    return run_one_cycle(
        solve_time=args.solve_time,
        forecast_path=args.forecast_path,
        prices_path=args.prices_path,
        state_in=args.state_in,
        state_out=args.state_out,
        dispatch_out=args.dispatch_out,
        cold_start=args.cold_start,
        config=config,
        export_out=args.export_out,
        prices_source=args.prices_source,
        resolution=args.resolution,
        forecast_resolution=args.forecast_resolution,
    )


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        logger.exception("unexpected error")
        sys.exit(3)
