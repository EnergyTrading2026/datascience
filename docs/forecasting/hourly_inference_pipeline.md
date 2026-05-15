# Hourly Inference Pipeline

This document describes the hourly forecasting pipeline, which serves as the integration layer between the forecasting models and the optimization group.

## Overview

The optimization team requires a regular (hourly) forecast of the thermal demand for the next 35 hours. The pipeline ingests the latest historical data, runs the configured forecasting model, calculates optional error metrics, and exports the prediction payload as a strict parquet file matching the `forecast_contract.md`.

## Key Components

1. **`BaseForecaster` Interface (`src/forecasting/base_model.py`)** 
   - All forecasting models implement the `predict(history, solve_time, horizon)` method. 
   - It expects training history, the anchor `solve_time` for the forecast, and defaults to a 35-hour `horizon`. 
   - Models also inherit a standard `calculate_error` feature.
2. **Exporter (`src/forecasting/export.py`)**
   - Implements the strict formatting restrictions defined in the [Forecast Contract](../../.vscode/forecast_contract.md).
   - Enforces a timezone-aware `DatetimeIndex`, single column (`demand_mw_th`), and Parquet serialization.
3. **Execution Script (`src/forecasting/run_hourly_forecast.py`)**
   - The primary entrypoint meant to be executed by a job scheduler (e.g., cron or Airflow).
   - Loads the latest data, sets the `solve_time`, executes the model, and calls the exporter.

## Usage

You can run the pipeline directly via terminal. 

```bash
# Run with default settings (Combined Seasonal Model, 35h horizon, current UTC time)
python3 src/forecasting/run_hourly_forecast.py

# Specify a specific model and enable error metric logging
python3 src/forecasting/run_hourly_forecast.py --model daily_naive --log_metrics

# Run an explicit solve_time (useful for backtesting/simulations)
python3 src/forecasting/run_hourly_forecast.py --solve_time "2026-03-01T12:00:00Z"
```

### Script Arguments
- `--model`: The model to use (choices: `daily_naive`, `weekly_naive`, `combined_seasonal`).
- `--input_file`: Path to the raw history `.csv`.
- `--output_dir`: Export directory (defaults to `/shared/forecast/`, locally points to `shared/forecast/`).
- `--solve_time`: The timestamp grounding the prediction. Defaults to the latest timestamp found in the data plus one hour.
- `--horizon`: How many hours into the future to predict (default: 35).
- `--log_metrics`: If set, evaluates the model's accuracy on the last `horizon` window and saves it to a `.json` file in the `output_dir`.

## Containerized deployment

The production path for the forecasting pipeline is a separate Docker stack
(`docker-compose.forecasting.yml`) that runs alongside — but completely
independent of — the optimization stack. The two containers communicate
only through the shared `data/forecast/` bind mount.

There is no live demand feed yet, so the container does not call
`run_hourly_forecast.py` per cycle. Instead it runs the **replay loop**
(`src/forecasting/replay_loop.py`), which walks the last
`REPLAY_LOOKBACK_MONTHS` (default 3) of the historical CSV: at each tick the
virtual `solve_time` advances by one hour, the configured model produces a
35h forecast from history-up-to-`solve_time`, and the parquet lands in
`/shared/forecast/`. When the virtual clock reaches `csv_end` the loop
**idles** — no more forecasts, heartbeat keeps ticking so the healthcheck
stays green.

The loop deliberately does **not** wrap. The optimization daemon enforces
`solve_time > last_processed`; a wrap-around would re-write older solve_times
and the daemon would silently skip them. To run the demo again, use
`scripts/reset_demo.sh` to wipe both forecast and daemon state.

### First-time setup

```bash
git clone <repo-url> /opt/forecasting
cd /opt/forecasting

# CSV input + forecast output dirs must exist with uids that both containers
# can write to. uid 1001 = forecasting, uid 1000 = optimization. Group sticky
# bit lets both write to the shared forecast dir.
mkdir -p data/forecasting/raw data/forecast data/state data/dispatch
sudo chown -R 1001:1000 data/forecast
sudo chmod 2775 data/forecast
sudo chown -R 1001:1001 data/forecasting/raw
sudo chown -R 1000:1000 data/state data/dispatch
# Put the CSV in place: data/forecasting/raw/raw_data_measured_demand.csv

docker compose -f docker-compose.forecasting.yml up -d --build
docker compose -f docker-compose.forecasting.yml logs -f forecasting
```

On startup the loop fires one tick immediately, so a fresh container
produces its first forecast within seconds (not after a full
`TICK_INTERVAL_S`). Subsequent ticks run on the configured interval.

### Price source coupling

The optimization daemon pulls DA prices for each cycle's solve_time. For the
replay (historical solve_times) the daemon's default `PRICES_SOURCE=live`
queries SMARD's chart_data API, which serves historical price buckets back
to 2018-09 — verified to return real DA prices for any solve_time in the
replay window. If the deployment cannot reach SMARD (offline demo,
restricted network), switch to file mode:

```bash
# In docker-compose.yml under the optimization service:
PRICES_SOURCE=csv
PRICES_PATH=/shared/smard/historical.csv   # bind-mount a SMARD-format CSV
```

The CSV must cover at least `[csv_end - REPLAY_LOOKBACK_MONTHS, csv_end + HORIZON_HOURS]`,
otherwise the daemon will fail to assemble a 35h horizon for the late
solve_times in the window.

### Configuration (environment variables)

| Var | Default | Meaning |
|---|---|---|
| `MODEL` | `combined_seasonal` | one of `daily_naive`, `weekly_naive`, `combined_seasonal` |
| `HORIZON_HOURS` | `35` | forward forecast length per cycle |
| `REPLAY_LOOKBACK_MONTHS` | `3` | size of the replay window before `csv_end` |
| `TICK_INTERVAL_S` | `3600` | seconds between ticks; drop low for demos |
| `CSV_PATH` | `/shared/forecasting/raw/raw_data_measured_demand.csv` | history CSV inside the container |
| `FORECAST_DIR` | `/shared/forecast` | parquet output dir inside the container |

### Coordination with the optimization stack

- Output filenames use hyphens in the time portion
  (`YYYY-MM-DDTHH-MM-SSZ.parquet`) to match the optimization daemon's
  scanner regex and to stay Windows-safe — see
  [`docs/optimization/forecast_contract.md`](../optimization/forecast_contract.md).
- Files are written atomically (tmp + rename) so the optimization daemon
  never sees a partial parquet when it scans concurrently.
- The daemon does not check `solve_time` against wall-clock — it consumes
  whatever the replay writes, in order. See the "Wall-clock independence"
  section in the forecast contract for the reasoning.
- The replay container clears stale `*.parquet` and the heartbeat file in
  `FORECAST_DIR` on startup, so each container start begins with a clean
  slate. To also clear daemon state (required when re-running the demo from
  the beginning), use `scripts/reset_demo.sh`.

### Restarting only one stack — footgun

The replay loop wipes the forecast directory on startup but does **not**
touch `data/state/` (owned by the optimization daemon, separate concern).
Consequence:

- If you `docker compose restart forecasting` **during** an active replay
  (the daemon still progressing through the window), the replay restarts at
  `csv_end - 3mo`. The daemon's `last_solve_time` is somewhere later in the
  window — newer than the replay's first new file — so monotonicity blocks
  every file until the replay catches up again.
- If you restart the forecasting container **after** the replay has been
  exhausted, the same thing happens at full magnitude: the daemon already
  processed `csv_end - 1h`, the replay now starts at `csv_end - 3mo`, every
  file is blocked, daemon sits idle forever.

The daemon will log a `WARNING` ("newest forecast ... is not newer than
processed state ...") once per blocked solve_time, so the condition is
visible in logs. The fix is always `scripts/reset_demo.sh` (clears state
and dispatch alongside the forecast dir). Do not restart the forecasting
container alone unless you accept this trade-off.

### Health and operations

```bash
# Pick up logs / status
docker compose -f docker-compose.forecasting.yml ps
docker compose -f docker-compose.forecasting.yml logs -f forecasting

# Restart only the forecasting container (independent of optimization)
docker compose -f docker-compose.forecasting.yml restart forecasting

# Tear down
docker compose -f docker-compose.forecasting.yml down

# Full demo reset (both stacks + shared state)
scripts/reset_demo.sh
```

The healthcheck watches `data/forecast/.forecasting-heartbeat`. The replay
loop touches it at startup, after every successful tick, and once per tick
interval after the replay window is exhausted (idle mode). The container
flips to `unhealthy` only if no heartbeat update has occurred in 90 minutes.