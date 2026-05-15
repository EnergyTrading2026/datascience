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
(`src/forecasting/replay_loop.py`), which simulates a live feed by walking
the last `REPLAY_LOOKBACK_MONTHS` (default 3) of the historical CSV: at
each tick the virtual `solve_time` advances by one hour, the configured
model produces a 35h forecast from history-up-to-`solve_time`, and the
parquet lands in `/shared/forecast/`. When the virtual clock reaches the
end of the CSV it wraps back to the start of the lookback window.

### First-time setup

```bash
git clone <repo-url> /opt/forecasting
cd /opt/forecasting

# CSV input + forecast output dirs must exist and be writable by uid 1001
# (the non-root user the container runs as). The chown matters when the
# directories are created by root; on a fresh personal dev box uid 1001 may
# not exist yet — the numeric ownership is what matters, not the name.
mkdir -p data/forecasting/raw data/forecast
# Put the CSV in place: data/forecasting/raw/raw_data_measured_demand.csv
sudo chown -R 1001:1001 data/forecasting/raw data/forecast

docker compose -f docker-compose.forecasting.yml up -d --build
docker compose -f docker-compose.forecasting.yml logs -f forecasting
```

On startup the loop fires one tick immediately, so a fresh container
produces its first forecast within seconds (not after a full
`TICK_INTERVAL_S`). Subsequent ticks run on the configured interval.

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
- The forecasting container runs as uid `1001`, the optimization container
  as uid `1000`. Both must be able to read/write `data/forecast/`; the
  simplest setup is `chown -R 1001:1000 data/forecast && chmod 2775
  data/forecast` (group sticky bit so files inherit the group from the
  directory), but any arrangement that gives both uids write access works.

### Health and operations

```bash
# Pick up logs / status
docker compose -f docker-compose.forecasting.yml ps
docker compose -f docker-compose.forecasting.yml logs -f forecasting

# Restart only the forecasting container (independent of optimization)
docker compose -f docker-compose.forecasting.yml restart forecasting

# Tear down
docker compose -f docker-compose.forecasting.yml down
```

The healthcheck watches `data/forecast/.forecasting-heartbeat`. The replay
loop touches it once at startup and after every successful tick; the
container flips to `unhealthy` if no tick has succeeded in 90 minutes.