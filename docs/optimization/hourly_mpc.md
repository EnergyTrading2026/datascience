# Hourly MPC — Production Architecture

Hourly model-predictive control for dispatch of Heat Pump, Condensing Boiler, CHP, and Thermal Storage. Each hour we solve a MILP over a forward horizon of up to 35 h, treat the first hour as committed, and re-plan on the next cycle with fresh forecast and price inputs.

This document describes the **architecture and orchestration** of the production code in `src/optimization/`. The MILP itself — decision variables, constraints, objective, parameters — is defined in [`optimization_problem.md`](optimization_problem.md). Where the two documents disagree on the time horizon or rolling-horizon mechanics, this one is current.

## Architecture

Three time concepts, distinct on purpose:

| Concept | Value | Meaning |
|---|---|---|
| **Step** | 15 min (`dt = 0.25 h`) | MILP discretization grid |
| **Plan horizon** | 11–35 h, target 35 h | How far the MILP looks ahead. `T = horizon_h × 4` steps. Determined per cycle by the joint coverage of demand forecast and DA prices. Below 11 h the cycle fails loudly. The 11 h floor is set so the daily 13:00 cycle (11 h pre-EPEX-clearing visibility) still runs. |
| **Commit horizon** | 1 h (4 steps) | Setpoints from the first hour of the solved plan are emitted as "the plan". Steps beyond are advisory and get overwritten by the next solve. |
| **Cadence** | hourly | One full solve every hour. |

### Why hourly

The forecasting pipeline publishes a fresh demand forecast each hour. Re-planning on every new forecast exposes the optimizer to information that wasn't available the previous hour. DA prices update only once per day (around 12:45 Berlin local, when the day-ahead auction publishes), so prices alone would not justify hourly cadence — the forecast freshness does.

### Why a 35 h plan with a 1 h commit

The plan has to look beyond the commit because storage arbitrage decisions made now depend on prices and demand many hours later (e.g. it is worth charging the storage now if electricity is cheap tonight). Committing only the first hour is consistent with the cadence: anything committed further would be overwritten by the next solve anyway, so calling it "committed" would be misleading.

### DA price visibility

SMARD publishes the day-ahead auction around 12:45 Berlin local, with prices for the next calendar day (00:00–24:00). Forward visibility from solve time `t`:

- `t` shortly after publish (e.g. 13:00) → prices reach to tomorrow 24:00 → ≈ 35 h
- `t` shortly before publish (e.g. 12:30) → prices reach only to today 24:00 → ≈ 12 h
- `t` overnight → coverage shrinks toward the today-24:00 boundary

The plan horizon adapts to this — `RuntimeConfig.horizon_hours_target = 35`, `horizon_hours_min = 11`. The 11 h floor exactly accommodates the 13:00 cycle (11 h until midnight, before EPEX clears at ~12:45 and SMARD mirrors by ~13:45). If neither forecast nor prices cover at least 11 h forward, the cycle returns exit-code 1 (recoverable).

## Cycle flow

`run.run_one_cycle` performs six steps every hour:

1. **Load carry-over state.** Either deserialize the previous hour's state from a JSON file, or initialize from `DispatchState.cold_start(...)` on first run (`--cold-start` flag).
2. **Fetch inputs.** Demand forecast via `adapters.forecast.load_forecast(...)`, DA prices via `adapters.smard.get_published_prices(...)`. Both anchor on the next full hour at or after `solve_time`.
3. **Reconcile horizons.** Take the joint length of forecast and prices, clamp to `[horizon_hours_min, horizon_hours_target]`, slice both inputs accordingly. Below the minimum, exit 1.
4. **Build and solve.** `model.build_model(...)` constructs the MILP from inputs, state, and `PlantParams`. `solve.solve(...)` calls HiGHS via Pyomo with the configured time limit and MIP gap, raising `SolverInfeasibleError` on non-OK termination.
5. **Extract.** `dispatch.extract_dispatch(...)` reads the first `commit_hours × 4` setpoints into a `Dispatch` dataclass (with per-component cost breakdown). `dispatch.extract_state(...)` reconstructs the carry-over `DispatchState` for the next cycle.
6. **Persist.** Dispatch is written to parquet at `--dispatch-out`. State is written atomically to `--state-out` (tmp-then-rename) so a partial-write cannot corrupt the next run's input. When `--export-out` is provided, a backend-ready JSON payload is also written for container export.

## Carry-over state

`state.DispatchState` is the safety-critical structure passed from one hourly solve to the next:

| Field | Source | Purpose |
|---|---|---|
| `timestamp` | end of last commit window | Sanity check (current run's `solve_time` should match) |
| `sto_soc_mwh_th` | planned SoC at `t = commit_end` | Initial SoC for next solve |
| `hp_on` | planned HP state at `t = commit_end` | Reported only — HP has no min-up/min-down, so it does not constrain the next horizon |
| `boiler_on`, `boiler_time_in_state_steps` | planned state at `t = commit_end` | Honor outstanding 1 h boiler min-up/min-down at start of next horizon |
| `chp_on`, `chp_time_in_state_steps` | planned state at `t = commit_end` | Honor outstanding 2 h CHP min-up/min-down at start of next horizon |

`time_in_state_steps` counts in 15-min steps, capped at `TIS_LONG = 999` (sentinel for "long enough that min-up/min-down is non-binding"). The capping handles the edge case where the carried-over state has not switched in many cycles — the integer would otherwise grow unboundedly without changing solver behavior.

`extract_state` accumulates time-in-state across the carry-over boundary: if the boiler did not switch at all in the committed window, the existing `time_in_state_steps` is added to the count rather than reset.

## Inputs

### Demand forecast — `adapters.forecast`

| Aspect | V1 contract |
|---|---|
| Format | Parquet |
| Index | hourly, tz-aware (any IANA tz; normalized to UTC on export) |
| Column | `demand_mw_th` (float, MW thermal) |
| Length | ≥ 11 h starting at the next full hour after `solve_time` |
| NaN | not allowed |

The schema is enforced strictly — violations raise `ForecastSchemaError`. The contract is V1 placeholder and will be locked down in a separate forecast-contract document.

### DA prices — `adapters.smard`

| Aspect | V1 contract |
|---|---|
| Source | SMARD CSV file, refreshed externally |
| Format | semicolon-separated, comma-decimal |
| Localization | Berlin local time, `ambiguous="infer"` for DST |
| Slice | hourly series anchored at the next full hour after `solve_time`; NaN dropped |

Live SMARD-API access is a deferred V2; until then the CSV path is the integration point.

## Code map

```
src/optimization/
├── __init__.py
├── config.py        PlantParams (physics/economics), RuntimeConfig (solver/operational tuning)
├── state.py         DispatchState dataclass, atomic save/load, cold_start factory
├── model.py         build_model — MILP construction
├── solve.py         solve, SolveResult, SolverInfeasibleError — HiGHS via Pyomo
├── dispatch.py      Dispatch dataclass, extract_dispatch, extract_state
├── run.py           run_one_cycle, CLI entrypoint, exit-code semantics
├── export_formatter.py backend JSON export schema formatter
└── adapters/
    ├── __init__.py
    ├── forecast.py  load_forecast, ForecastSchemaError
    └── smard.py     get_published_prices — CSV parser
```

Tests live in `tests/optimization/` and mirror this structure (smoke, state, adapters, dispatch, run end-to-end).

## Operational contract

CLI:

```
python -m optimization.run \
    --solve-time 2026-04-28T13:00:00+02:00 \
    --forecast-path /path/to/forecast.parquet \
    --prices-path  /path/to/smard.csv \
    --state-in     /path/to/state.json \
    --state-out    /path/to/state.json \
    --dispatch-out /path/to/dispatch.parquet \
    [--export-out  /path/to/backend_export.json] \
    [--cold-start]
```

`solve_time` must be tz-aware. `--state-in` is required unless `--cold-start` is set (used for the first deployment).

Exit codes:

| Code | Meaning |
|---|---|
| 0 | Success. Dispatch and state written. |
| 1 | Recoverable failure: missing input file, horizon below minimum, malformed schema. Cycle should be retried. |
| 2 | Solver reported infeasible. Manual intervention needed before the next cycle. |
| 3 | Unexpected error (caught at the top level). Investigation needed. |

Cadence is one solve per hour, anchored on a full hour. The exact firing offset within the hour (e.g. `xx:05`) is a deployment-side decision and not encoded in the CLI.
