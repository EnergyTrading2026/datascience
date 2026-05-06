# Demand Forecast Contract

Forecasting publishes an hourly thermal demand forecast that the
Optimization MPC cycle consumes once per hour.

## File schema

| | |
|---|---|
| Format | parquet, readable by `pandas.read_parquet` |
| Index | tz-aware `DatetimeIndex`, **UTC**, strictly hourly |
| Column | `demand_mw_th` (float, MW thermal) |
| Values | no NaN |
| Coverage | ≥ 35 hours, starting at the file's `solve_time` |

## File layout

- Base directory: `<shared>/forecast/` (location TBD; both teams have
  read+write access).
- Per-cycle file: `<base>/<YYYY-MM-DD>T<HH:MM:SS>Z.parquet`,
  where the timestamp is the **solve_time in UTC** the forecast targets.
  Example: forecast for `2026-05-04 13:00 UTC` →
  `2026-05-04T13:00:00Z.parquet`.
- The file's first index entry equals that solve_time.
- Symlink `<base>/latest.parquet` points to the newest published file
  (convenience for humans; the optimizer reads the timestamped path).

## Atomic write

Forecasting must write atomically:

```python
tmp = path.with_suffix(".parquet.tmp")
df.to_parquet(tmp)
os.rename(tmp, path)  # POSIX-atomic
```

so the optimizer never reads a half-written file.

## Cadence

- One file per hour, every hour.
- Deadline: file for solve_time `T` must exist on disk **by `T`** (i.e.
  before the wall-clock UTC hour `T` arrives).

## Failure semantics

- Missing file → optimizer exits 1, that cycle is skipped. No fallback
  to the previous forecast.
- Schema violation → optimizer raises `ForecastSchemaError`, exits 3.
- Strict delivery: Forecasting is responsible for publishing every hour.
  Recovery is "ship the next hour cleanly", not "patch the missed one".

## Out of scope (v1)

- Confidence intervals (point forecast only)
- Anything other than thermal demand
- Sub-hourly resolution
- Quality metrics (Forecasting's internal concern)
