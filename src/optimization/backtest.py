"""Backtest harness for the hourly MPC.

Simulates the production hourly dispatch loop (`optimization.run.run_one_cycle`)
on historical data. Used to:
  - validate that the production code runs end-to-end over long ranges
  - profile solve-time distribution and infeasibility rate
  - compute KPIs (total cost, on/off counts, terminal SoC) for the dashboard
  - benchmark alternative dispatch strategies (e.g. merit-order baseline)
    against the MPC by passing a custom `strategy_fn`

By default, demand forecast == realized demand (perfect foresight). This is the
upper bound on achievable performance; real forecasts will do worse and the gap
quantifies the value of forecast quality.

Library usage:
    from optimization.backtest import run_backtest
    result = run_backtest(demand, prices, start, end)

CLI usage (after `pip install -e .`):
    optimization-backtest --start 2024-04-01 --end 2024-04-08 --output-dir out/backtest/

Multi-asset note: KPIs aggregate across all assets in a family (sum of SoC over
storages, count of on-units per family, sum of dispatch energy per family).
With ``PlantConfig.legacy_default()`` (1 of each), aggregates equal the single
asset's value, so legacy reports are unchanged.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

from optimization.config import PlantConfig, RuntimeConfig
from optimization.dispatch import Dispatch, extract_dispatch, extract_state
from optimization.model import build_model
from optimization.solve import SolverInfeasibleError, solve
from optimization.state import DispatchState

logger = logging.getLogger("optimization.backtest")

INT_PER_HOUR = 4
BERLIN = "Europe/Berlin"
SMARD_PRICE_COLUMN = "Deutschland/Luxemburg [€/MWh] Berechnete Auflösungen"


class StrategyInfeasibleError(RuntimeError):
    """Strategy could not produce a dispatch this cycle.

    The harness catches this, records the failure, and cold-starts the next
    cycle. MPC strategies typically wrap `SolverInfeasibleError` as this.
    """


@dataclass
class StrategyResult:
    """Per-cycle output every strategy must produce."""

    dispatch: Dispatch
    new_state: DispatchState
    objective_eur: float | None  # None if strategy has no notion of an objective
    expected_cost_eur: float
    solve_time_s: float


class StrategyFn(Protocol):
    """Signature any dispatch strategy must implement."""

    def __call__(
        self,
        forecast: pd.Series,
        prices: pd.Series,
        state: DispatchState,
        config: PlantConfig,
        runtime: RuntimeConfig,
        solve_time: pd.Timestamp,
    ) -> StrategyResult: ...


def mpc_strategy(
    forecast: pd.Series,
    prices: pd.Series,
    state: DispatchState,
    config: PlantConfig,
    runtime: RuntimeConfig,
    solve_time: pd.Timestamp,
) -> StrategyResult:
    """Default strategy: build + solve the production MILP."""
    try:
        model = build_model(
            forecast, prices, state, config,
            demand_safety_factor=runtime.demand_safety_factor,
        )
        result = solve(
            model,
            time_limit_s=runtime.solver_time_limit_s,
            mip_gap=runtime.solver_mip_gap,
        )
    except SolverInfeasibleError as e:
        raise StrategyInfeasibleError(str(e)) from e

    commit_intervals = runtime.commit_hours * INT_PER_HOUR
    dispatch = extract_dispatch(model, n_intervals=commit_intervals, solve_time=solve_time)
    commit_end = forecast.index[0] + pd.Timedelta(hours=runtime.commit_hours)
    new_state = extract_state(model, t_end=commit_intervals, commit_end_time=commit_end)
    return StrategyResult(
        dispatch=dispatch,
        new_state=new_state,
        objective_eur=result.objective_eur,
        expected_cost_eur=dispatch.expected_cost_eur,
        solve_time_s=result.solve_time_s,
    )


@dataclass
class BacktestResult:
    records: pd.DataFrame
    dispatch_log: pd.DataFrame
    summary: dict
    config: dict
    runtime: dict


def _total_soc(state: DispatchState) -> float:
    return float(sum(s.soc_mwh_th for s in state.storages.values()))


def _count_on(state: DispatchState, asset_ids: list[str]) -> int:
    return int(sum(state.units[i].on for i in asset_ids if i in state.units))


def run_backtest(
    demand: pd.Series,
    prices: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
    config: PlantConfig | None = None,
    runtime: RuntimeConfig | None = None,
    strategy_fn: StrategyFn | None = None,
    initial_state: DispatchState | None = None,
    log_every: int = 500,
) -> BacktestResult:
    """Run hourly dispatch from `start` (inclusive) to `end` (exclusive).

    Args:
        demand: realized hourly demand (MW_th), tz-aware index. Used as
            perfect-foresight forecast.
        prices: hourly DA prices (EUR/MWh), tz-aware index.
        start, end: tz-aware Berlin timestamps. Must be hour-aligned.
        config: plant configuration. Defaults to PlantConfig.legacy_default().
        runtime: solver/operational tuning. Defaults to RuntimeConfig().
        strategy_fn: dispatch strategy. Defaults to MPC.
        initial_state: carry-over state at `start`. Defaults to cold start.
        log_every: progress log frequency (in completed cycles). 0 = silent.
    """
    config = config or PlantConfig.legacy_default()
    runtime = runtime or RuntimeConfig()
    strategy_fn = strategy_fn or mpc_strategy
    state = initial_state or DispatchState.cold_start(config, start)

    hp_ids = [hp.id for hp in config.heat_pumps]
    boiler_ids = [b.id for b in config.boilers]
    chp_ids = [c.id for c in config.chps]

    records: list[dict] = []
    dispatch_frames: list[pd.DataFrame] = []
    n_infeasible = 0
    n_skipped_short = 0
    t_wallclock = time.time()
    solve_time = start

    while solve_time < end:
        forecast_slice = demand.loc[solve_time:]
        price_slice = prices.loc[solve_time:]
        horizon_h = min(
            len(forecast_slice), len(price_slice), runtime.horizon_hours_target,
        )
        if horizon_h < runtime.horizon_hours_min:
            n_skipped_short += 1
            logger.warning(
                "skip %s: horizon %dh < min %dh",
                solve_time, horizon_h, runtime.horizon_hours_min,
            )
            solve_time += pd.Timedelta(hours=runtime.commit_hours)
            continue

        forecast_slice = forecast_slice.iloc[:horizon_h]
        price_slice = price_slice.iloc[:horizon_h]

        try:
            res = strategy_fn(
                forecast_slice, price_slice, state, config, runtime, solve_time,
            )
        except StrategyInfeasibleError as e:
            n_infeasible += 1
            logger.warning("infeasible at %s: %s -- cold-starting next cycle", solve_time, e)
            state = DispatchState.cold_start(
                config, solve_time + pd.Timedelta(hours=runtime.commit_hours)
            )
            solve_time += pd.Timedelta(hours=runtime.commit_hours)
            continue

        records.append({
            "solve_time": solve_time,
            "horizon_h": horizon_h,
            "objective_eur": res.objective_eur,
            "expected_cost_eur": res.expected_cost_eur,
            "soc_start_mwh": _total_soc(state),
            "soc_end_mwh": _total_soc(res.new_state),
            # *_on_count_end = count of units of this family on at cycle end
            # (0..N). For legacy 1-of-each plants this is still 0/1.
            "boilers_on_count_end": _count_on(res.new_state, boiler_ids),
            "chps_on_count_end": _count_on(res.new_state, chp_ids),
            "hps_on_count_end": _count_on(res.new_state, hp_ids),
            "solve_time_s": res.solve_time_s,
        })
        dispatch_frames.append(res.dispatch.to_dataframe())

        state = res.new_state
        solve_time += pd.Timedelta(hours=runtime.commit_hours)

        if log_every and len(records) % log_every == 0:
            elapsed = time.time() - t_wallclock
            ms_per = elapsed / len(records) * 1000
            logger.info(
                "%5d cycles, last=%s, elapsed=%.0fs (%.0f ms/cycle)",
                len(records), records[-1]["solve_time"], elapsed, ms_per,
            )

    records_df = pd.DataFrame(records)
    dispatch_log = pd.concat(dispatch_frames) if dispatch_frames else pd.DataFrame()
    summary = _build_summary(
        records_df, dispatch_log, n_infeasible, n_skipped_short,
        wallclock_s=time.time() - t_wallclock,
    )
    return BacktestResult(
        records=records_df,
        dispatch_log=dispatch_log,
        summary=summary,
        config=config.to_dict(),
        runtime=asdict(runtime),
    )


def _sum_family_quantity(df: pd.DataFrame, family: str, quantity: str) -> float:
    """Sum 'value' over rows matching (family, quantity). 0.0 if none match.

    Long-format aggregator: replaces wide-column-prefix scanning. The dispatch
    log is one row per (timestamp, asset, quantity); summing across all assets
    in a family is a simple boolean mask.
    """
    if not len(df):
        return 0.0
    mask = (df["family"] == family) & (df["quantity"] == quantity)
    return float(df.loc[mask, "value"].sum())


def _build_summary(
    records: pd.DataFrame,
    dispatch_log: pd.DataFrame,
    n_infeasible: int,
    n_skipped_short: int,
    wallclock_s: float,
) -> dict:
    """KPI summary suitable for JSON serialization (no NaN, no timestamps)."""
    summary: dict = {
        "n_cycles": int(len(records)),
        "n_infeasible": int(n_infeasible),
        "n_skipped_short": int(n_skipped_short),
        "wallclock_s": round(wallclock_s, 2),
    }
    if len(records):
        solve_times = records["solve_time_s"].astype(float)
        summary["solve_time_s"] = {
            "p50": float(solve_times.quantile(0.5)),
            "p95": float(solve_times.quantile(0.95)),
            "max": float(solve_times.max()),
            "mean": float(solve_times.mean()),
        }
        summary["total_expected_cost_eur"] = float(records["expected_cost_eur"].sum())
        summary["soc_end"] = {
            "min_mwh": float(records["soc_end_mwh"].min()),
            "max_mwh": float(records["soc_end_mwh"].max()),
            "final_mwh": float(records["soc_end_mwh"].iloc[-1]),
        }
        # Sum of "units-on at cycle end" across cycles, per family. With 1h
        # commits this approximates unit-hours; with N units of a family it's
        # cycle-aggregated unit-hours (max = N * n_cycles).
        summary["unit_on_hours"] = {
            "boiler": int(records["boilers_on_count_end"].sum()),
            "chp": int(records["chps_on_count_end"].sum()),
            "hp": int(records["hps_on_count_end"].sum()),
        }
    if len(dispatch_log):
        # 15-min interval energy, MWh_th. Useful sanity for total throughput.
        # Long format: aggregate per (family, quantity) over all assets.
        dt_h = 0.25
        summary["total_energy_mwh_th"] = {
            "boiler": _sum_family_quantity(dispatch_log, "boiler", "q_th_mw") * dt_h,
            "chp_thermal_implied": _sum_family_quantity(dispatch_log, "chp", "p_el_mw") * dt_h,
            "hp_electrical": _sum_family_quantity(dispatch_log, "hp", "p_el_mw") * dt_h,
            "sto_charge": _sum_family_quantity(dispatch_log, "storage", "charge_mw_th") * dt_h,
            "sto_discharge": _sum_family_quantity(dispatch_log, "storage", "discharge_mw_th") * dt_h,
        }
    return summary


# --- Data loaders -----------------------------------------------------------
# Backtest reads bulk historical CSVs. The production adapters
# (optimization.adapters.{forecast,smard}) handle live data and are not designed
# for whole-year loads, so backtest has its own loaders.

def load_demand(path: Path) -> pd.Series:
    """Load measured heat demand CSV. Convert W -> MW_th, hourly Berlin tz."""
    df = pd.read_csv(path)
    ts = pd.to_datetime(df["Time Point"], format="ISO8601", utc=True)
    s = pd.Series(
        df["Measured Heat Demand[W]"].astype(float).to_numpy() / 1e6,
        index=ts,
        name="demand_mw_th",
    )
    s = s.groupby(s.index).first().sort_index().asfreq("1h")
    n_nan = int(s.isna().sum())
    if n_nan:
        s = s.interpolate(method="linear", limit_direction="both")
        logger.info("load_demand: filled %d NaN via linear interpolation", n_nan)
    s.index = s.index.tz_convert(BERLIN)
    return s


def load_smard_prices(path: Path) -> pd.Series:
    """Load full SMARD CSV as an hourly DA-price series (Berlin tz, NaN dropped)."""
    df = pd.read_csv(path, sep=";", decimal=",", low_memory=False)
    ts = pd.to_datetime(df["Datum von"], format="%d.%m.%Y %H:%M").dt.tz_localize(
        BERLIN, ambiguous="infer",
    )
    s = pd.Series(
        pd.to_numeric(df[SMARD_PRICE_COLUMN], errors="coerce").to_numpy(),
        index=ts,
        name="price_eur_mwh",
    )
    return s.groupby(s.index).first().sort_index().asfreq("1h").dropna()


# --- Output writers ---------------------------------------------------------

def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def write_outputs(result: BacktestResult, output_dir: Path) -> None:
    """Write summary.json + records.parquet + dispatch_log.parquet to `output_dir`.

    Overwrites any existing files in the directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "summary": result.summary,
        "config": result.config,
        "runtime": result.runtime,
    }
    (output_dir / "summary.json").write_text(json.dumps(_json_safe(payload), indent=2))

    if len(result.records):
        result.records.to_parquet(output_dir / "records.parquet")
    if len(result.dispatch_log):
        result.dispatch_log.to_parquet(output_dir / "dispatch_log.parquet")


# --- CLI --------------------------------------------------------------------

def _parse_ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz=BERLIN)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Year-long backtest of the hourly MPC.")
    p.add_argument("--start", type=_parse_ts, default=_parse_ts("2024-04-01"))
    p.add_argument("--end", type=_parse_ts, default=_parse_ts("2025-04-01"))
    # Defaults match the repo data layout established by the forecasting team
    # (data/forecasting/raw/ for demand) and mirror that convention for prices
    # (data/optimization/raw/). The forecasting team tracks the demand CSV in
    # git; the SMARD price CSV is uploaded by the optimization team.
    p.add_argument("--demand-path", type=Path,
                   default=Path("data/forecasting/raw/raw_data_measured_demand.csv"))
    p.add_argument("--prices-path", type=Path,
                   default=Path("data/optimization/raw/Gro_handelspreise_202403010000_202603020000_Stunde.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("out/backtest"))
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument(
        "--config-file",
        type=Path,
        default=None,
        help="Path to plant_config.json. Defaults to PlantConfig.legacy_default() "
             "if omitted.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("pyomo.contrib.appsi.solvers.highs").setLevel(logging.ERROR)

    demand = load_demand(args.demand_path)
    prices = load_smard_prices(args.prices_path)

    if args.config_file is not None:
        try:
            config = PlantConfig.from_json(args.config_file)
        except (FileNotFoundError, ValueError) as e:
            logger.error("config load failed (%s): %s", args.config_file, e)
            return 1
        logger.info("loaded plant config from %s", args.config_file)
    else:
        config = PlantConfig.legacy_default()

    n_hours = (args.end - args.start).total_seconds() / 3600
    logger.info("backtest range: %s -> %s (%.0fh)", args.start, args.end, n_hours)
    logger.info("demand: %s -> %s, %d hourly values",
                demand.index[0], demand.index[-1], len(demand))
    logger.info("prices: %s -> %s, %d hourly values",
                prices.index[0], prices.index[-1], len(prices))

    result = run_backtest(
        demand=demand, prices=prices,
        start=args.start, end=args.end,
        config=config, runtime=RuntimeConfig(),
        log_every=args.log_every,
    )

    write_outputs(result, args.output_dir)

    s = result.summary
    logger.info("--- SUMMARY ---")
    logger.info("cycles run:       %d", s["n_cycles"])
    logger.info("infeasible:       %d", s["n_infeasible"])
    logger.info("skipped (short):  %d", s["n_skipped_short"])
    if s["n_cycles"]:
        logger.info("total exp. cost:  %12.0f EUR", s["total_expected_cost_eur"])
        logger.info("solve time p50:   %.3fs  p95: %.3fs  max: %.3fs",
                    s["solve_time_s"]["p50"], s["solve_time_s"]["p95"],
                    s["solve_time_s"]["max"])
        logger.info("SoC end range:    [%.1f, %.1f] MWh_th, final=%.1f",
                    s["soc_end"]["min_mwh"], s["soc_end"]["max_mwh"],
                    s["soc_end"]["final_mwh"])
    logger.info("wallclock:        %.0fs", s["wallclock_s"])
    logger.info("results -> %s", args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
