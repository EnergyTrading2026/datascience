# Demand Forecast Contract

This document defines the forecast artifact that the optimization code
consumes today.

The goal of the contract is intentionally narrow: Forecasting publishes a
file, and Optimization can load it without needing to know anything about
the forecasting model that produced it.

## Purpose

Forecasting publishes a thermal demand forecast once per optimization cycle.
Optimization consumes that forecast as the expected heat demand over the
forward planning horizon.

The optimization model itself runs on an internal 15-minute grid. In the
current standard production setup, Forecasting provides hourly demand values
and Optimization expands each hourly value across the corresponding four
15-minute model steps.

The optimizer uses the forecast only as a time series of demand values. It
does not require model metadata, confidence intervals, quality metrics, or
any additional features.

## Artifact

- Format: Parquet
- Reader: must be readable by `pandas.read_parquet`
- Payload: one time series of thermal demand

## Required Schema

The file must contain:

- A `DatetimeIndex`
- Exactly one required demand column: `demand_mw_th`

### Index Requirements

- The index must be timezone-aware
- The index must be sorted in ascending time order
- The index must have uniform spacing
- The current standard delivery is hourly resolution

For the current production flow, hourly delivery is the expected default.
This does not mean the optimizer runs hourly internally; it means the forecast
artifact is hourly and is then mapped onto the optimizer's 15-minute internal
grid.

The optimizer can also validate quarter-hourly input when explicitly run in
that mode, but that is not the default contract.

### Column Requirements

`demand_mw_th`

- Type: numeric
- Unit: MW thermal
- Semantics: point forecast of thermal demand for each timestamp
- Missing values: not allowed

## Time Semantics

Each row represents the forecast demand at its timestamp.

For the standard hourly contract:

- The first timestamp must correspond to the cycle's `solve_time`
- Timestamps then continue in 1-hour steps

Optimization anchors consumption at the next valid slot boundary at or after
`solve_time`. In the standard hourly flow, that boundary is the next full
hour. After loading, each hourly forecast value is applied to the four
15-minute optimization intervals within that hour.

The index only needs to be timezone-aware. The optimization code currently
normalizes timestamps internally and does not require a specific source
timezone for the forecast file.

## Coverage

The forecast must extend far enough into the future for the optimizer to run.

- Operational target: 35 hours of forward coverage
- Current optimizer minimum: 11 hours of forward coverage

The 35-hour target is the intended planning horizon. The 11-hour minimum is
the runtime floor below which the optimizer skips the cycle.

## Cadence

- One forecast file per hour
- One file corresponds to one optimization cycle

The optimization system replans hourly, so forecasting should publish a fresh
forecast on the same cadence.

## File Handoff Convention

The optimizer core consumes any explicitly provided forecast path. Still, for
the handoff between Forecasting and Optimization, files should follow a stable
cycle-based naming convention.

- Forecasting must publish forecast files to `/shared/forecast/`
- One file corresponds to one optimization cycle
- Recommended filename format: `YYYY-MM-DDTHH:MM:SSZ.parquet`
- The timestamp in the filename is the cycle `solve_time`
- The first timestamp inside the file should match that `solve_time`

This naming convention is part of the artifact handoff contract between the
two components. It provides a simple and unambiguous way to identify which
forecast belongs to which optimization cycle and where optimization should
look for incoming forecast artifacts.

## Consumer Guarantees

If the file satisfies this contract, the optimization code will:

- load the forecast as the demand input for the MPC solve
- align it to the solve-time anchor
- map the forecast onto the optimizer's internal 15-minute model grid
- in the current standard setup, forward-fill each hourly value across four
  15-minute steps
- optionally accept quarter-hourly forecast input when explicitly configured

## Out Of Scope

The following are not part of this contract:

- confidence intervals
- probabilistic scenarios
- forecast model metadata
- training diagnostics
- quality scores
- weather features
- any signal other than thermal demand

## Example

Example hourly file:

| timestamp | demand_mw_th |
|---|---:|
| `2026-05-07 13:00:00+00:00` | `10.4` |
| `2026-05-07 14:00:00+00:00` | `10.1` |
| `2026-05-07 15:00:00+00:00` | `9.8` |

In practice, the file is stored as parquet with a timezone-aware
`DatetimeIndex`, not as a markdown table.
