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

The production path for the forecasting pipeline is a separate Docker
stack (`docker-compose.forecasting.yml`) running the replay loop
(`src/forecasting/replay_loop.py`). Operational details — first-time
host setup, configuration env vars, restart behavior, healthcheck
semantics, coordination with the optimization stack — live in the
combined deployment runbook:

→ [`docs/deploy.md`](../deploy.md)