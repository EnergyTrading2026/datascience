# Forecasting Replay — Deployment Runbook

Hourly demand-forecasting replay loop, packaged as a Docker image. The
container runs as a long-lived daemon that produces one forecast parquet
per virtual `solve_time` and drops it into the shared `data/forecast/`
directory the optimization daemon polls.

This runbook is the operational counterpart to the design notes in
[`hourly_inference_pipeline.md`](./hourly_inference_pipeline.md). For the
optimization side of the system see
[`docs/optimization/deploy.md`](../optimization/deploy.md).

## What runs

A separate Docker stack (`docker-compose.forecasting.yml`) that runs
alongside — but completely independent of — the optimization stack. The
two containers communicate only through the shared `data/forecast/`
bind mount.

There is no live demand feed yet, so the container does not call
`run_hourly_forecast.py` per cycle. Instead it runs the **replay loop**
(`src/forecasting/replay_loop.py`), which walks the last
`REPLAY_LOOKBACK_MONTHS` (default 3) of the historical CSV: at each tick
the virtual `solve_time` advances by one hour, the configured model
produces a 35h forecast from history-up-to-`solve_time`, and the parquet
lands in `/shared/forecast/`. When the virtual clock reaches `csv_end`
the loop **idles** — no more forecasts, heartbeat keeps ticking so the
healthcheck stays green.

The loop deliberately does **not** wrap. The optimization daemon
enforces `solve_time > last_processed`; a wrap-around would re-write
older solve_times and the daemon would silently skip them. To run the
demo again from the start, use `scripts/reset_demo.sh` to wipe both
forecast and daemon state.

`virt_solve_time` is persisted to `/shared/forecast/.replay-state.json`
after every successful tick (atomic write). A container restart — crash,
`docker compose restart`, image rebuild — therefore **resumes from where
it left off** rather than re-running the window from `csv_end - 3mo`.
The state file survives the startup cleanup of stale parquets; only
`reset_demo.sh` wipes it.

## First-time setup

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

## Price source coupling

The optimization daemon pulls DA prices for each cycle's solve_time. For
the replay (historical solve_times) the daemon's default
`PRICES_SOURCE=live` queries SMARD's chart_data API, which serves
historical price buckets back to 2018-09 — verified to return real DA
prices for any solve_time in the replay window. If the deployment cannot
reach SMARD (offline demo, restricted network), switch to file mode:

```bash
# In docker-compose.yml under the optimization service:
PRICES_SOURCE=csv
PRICES_PATH=/shared/smard/historical.csv   # bind-mount a SMARD-format CSV
```

The CSV must cover at least `[csv_end - REPLAY_LOOKBACK_MONTHS, csv_end + HORIZON_HOURS]`,
otherwise the daemon will fail to assemble a 35h horizon for the late
solve_times in the window.

## Configuration (environment variables)

| Var | Default | Meaning |
|---|---|---|
| `MODEL` | `combined_seasonal` | one of `daily_naive`, `weekly_naive`, `combined_seasonal` |
| `HORIZON_HOURS` | `35` | forward forecast length per cycle |
| `REPLAY_LOOKBACK_MONTHS` | `3` | size of the replay window before `csv_end` |
| `TICK_INTERVAL_S` | `3600` | seconds between ticks; drop low for demos |
| `CSV_PATH` | `/shared/forecasting/raw/raw_data_measured_demand.csv` | history CSV inside the container |
| `FORECAST_DIR` | `/shared/forecast` | parquet output dir inside the container |

## Coordination with the optimization stack

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

## Restarting only one stack

Restarting `forecasting` alone is safe in the typical case. The replay
state file (`/shared/forecast/.replay-state.json`) survives the startup
cleanup, so the loop resumes at the next unprocessed `solve_time`. The
daemon sees a contiguous stream and processes each new file in order.

There is still one situation that needs operator intervention: if the
state file is **missing or out of the current replay window** (e.g.
someone deleted it manually, or the input CSV has been replaced with one
whose range no longer overlaps), the loop falls back to `csv_end - 3mo`
and logs a `WARNING`. If the daemon's `last_solve_time` is later than
that fallback, the daemon's monotonicity check will block every new file
and emit its own warning ("newest forecast ... is not newer than
processed state ..."). The fix is `scripts/reset_demo.sh`, which wipes
both sides cleanly.

Rule of thumb: if you only restart the forecasting container, you're
fine. If you wipe `data/forecast/` (including the state file) without
also wiping `data/state/`, run `reset_demo.sh`.

## Health and operations

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
