# Deployment Runbook — Forecasting + Optimization

End-to-end runbook for the dispatch pipeline. Two Docker stacks share a
single `data/` tree on the host and communicate through the filesystem.
No HTTP between them, no orchestrator, no message bus — just files.

This is the single source of truth for deploying the pipeline. Component-
level design notes live next to the code (see
[`docs/optimization/`](./optimization/) and
[`docs/forecasting/`](./forecasting/)), but everything operational is
here.

## Overview

```
   ┌──────────────────────────┐               ┌──────────────────────────┐
   │   forecasting-replay     │               │   optimization-mpc       │
   │   (uid 1001)             │               │   (uid 1000)             │
   │                          │  parquet ↓    │                          │
   │   walks last 3 mo of CSV │──────────────▶│   one MILP cycle per     │
   │   1 tick = 1 h virtual   │  data/forecast│   new forecast file      │
   │                          │     (shared)  │   → dispatch + state     │
   └──────────────────────────┘               └──────────────────────────┘
            ▲                                            │
            │ raw CSV (read-only)                        │ live DA prices
   data/forecasting/raw/                                 ▼
                                                  https://www.smard.de
```

Three long-/short-lived containers:

| Service | Image | Lifecycle | Writes | Reads |
|---|---|---|---|---|
| `forecasting-replay` | `Dockerfile.forecasting` | long-running daemon | `data/forecast/` | `data/forecasting/raw/` |
| `optimization-init-state` | `Dockerfile` | one-shot on first deploy | `data/state/` | `data/config/` (optional) |
| `optimization-mpc` | `Dockerfile` | long-running daemon | `data/state/`, `data/dispatch/` | `data/forecast/`, `data/config/` |

The two stacks are deployed and restarted independently. Their only
shared surface is the `data/forecast/` bind mount; the contract there is
defined in [`forecast_contract.md`](./optimization/forecast_contract.md).

## Server requirements

- Linux (any Docker-supported distro). Images are platform-independent
  and validated on macOS Docker Desktop, but the production target is
  Linux.
- Docker Engine ≥ 24 with the Compose v2 plugin (`docker compose`, not
  `docker-compose`).
- Outbound network:
    - **Runtime:** HTTPS to `https://www.smard.de` for live DA prices —
      the only steady-state external dependency.
    - **Build time only** (`docker compose build` / `up --build`):
      additionally `ghcr.io` (uv image), `docker.io` /
      `registry-1.docker.io` (Python base image), `pypi.org` /
      `files.pythonhosted.org` (Python packages).
- Disk: ~2 GB for both images plus growth in `data/`:
    - `data/dispatch/` ~5–10 KB/cycle → 50–100 MB/year
    - `data/state/` ~200 B/cycle → 2 MB/year
    - `data/forecast/` ~5–10 KB/cycle, same order
    - All grow linearly and unbounded; no rotation is built in.
- Persistent storage under `data/` (must survive restarts).
- NTP-synced wall clock (chrony / systemd-timesyncd). The host timezone
  does not matter — both containers force UTC internally via
  `TZ=Etc/UTC`.

## First-time setup

The setup below covers both stacks from a fresh server in one pass.
**Do this in one shell session and from a single working directory.**
Both Compose files use *relative* bind mounts (`./data/...`), so they
must be invoked from the same CWD or they'll see different filesystem
trees and the containers will silently fail to talk to each other.

### 1. Clone and prepare the data tree

```bash
git clone <repo-url> /opt/dispatch
cd /opt/dispatch

mkdir -p data/forecast \
         data/state \
         data/dispatch \
         data/config \
         data/forecasting/raw
```

### 2. Set ownership

Both container images run as **non-root** with these uids:

| Container | uid:gid |
|---|---|
| `forecasting-replay` | 1001:1001 |
| `optimization-init-state` / `optimization-mpc` | 1000:1000 |

The bind mounts override the `chown` baked into the images, so host-side
permissions are what the containers actually see. The split below
matches each directory to the uid that needs **write** access; default
umask 022 leaves files world-readable so cross-stack *reads* work
without any extra group setup:

```bash
sudo chown -R 1001:1001 data/forecast data/forecasting
sudo chown -R 1000:1000 data/state data/dispatch data/config
```

If the host's first interactive user is uid 1000 (the Ubuntu/Debian
default), `data/state` etc. may already be correct without `sudo`. The
`chown` is safe to run regardless and avoids surprises on hardened
hosts where service accounts use different uids.

### 3. Put the input CSV in place

```bash
cp /path/to/raw_data_measured_demand.csv \
   data/forecasting/raw/raw_data_measured_demand.csv
```

The filename matters — the replay container reads exactly
`/shared/forecasting/raw/raw_data_measured_demand.csv` unless `CSV_PATH`
is overridden. If you move or rename the CSV, point `CSV_PATH` at the
new location.

### 4. Start both stacks

```bash
# Forecasting first so a forecast is ready by the time the daemon polls.
docker compose -f docker-compose.forecasting.yml up -d --build

# Optimization. init-state runs once and exits; the daemon then starts.
docker compose up -d --build
```

Order is *not* strictly required (the daemon waits idle until a
forecast appears) but bringing forecasting up first means the daemon
processes a cycle immediately on its first scan rather than logging
empty scans for one tick interval.

### 5. Verify

```bash
docker compose ps                                             # init-state Exited 0, optimization Up (healthy)
docker compose -f docker-compose.forecasting.yml ps           # forecasting Up (healthy)

# First forecast appears within seconds:
ls -lt data/forecast/

# First dispatch appears within ~30 s of the first forecast:
ls -lt data/dispatch/

# Both heartbeats fresh:
stat -c '%Y %n' data/forecast/.heartbeat data/state/.heartbeat
```

The healthchecks accept the heartbeats as fresh for up to 90 min. The
optimization container's `start_period` is 15 min (gives the first
solve room); the forecasting container's is 2 min.

## Configuration

### Environment variables

Defaults are baked into the Compose files; override via shell env or a
`.env` file at the repo root.

**Forecasting** (`docker-compose.forecasting.yml`):

| Var | Default | Meaning |
|---|---|---|
| `MODEL` | `combined_seasonal` | `daily_naive`, `weekly_naive`, or `combined_seasonal` |
| `HORIZON_HOURS` | `35` | forward forecast length per cycle |
| `REPLAY_LOOKBACK_MONTHS` | `3` | size of the replay window before `csv_end` |
| `TICK_INTERVAL_S` | `3600` | seconds between ticks; see "Demo mode" below |
| `CSV_PATH` | `/shared/forecasting/raw/raw_data_measured_demand.csv` | history CSV inside the container |
| `FORECAST_DIR` | `/shared/forecast` | parquet output dir inside the container |

**Optimization** (`docker-compose.yml`):

| Var | Default | Meaning |
|---|---|---|
| `CONFIG_FILE` | unset (= legacy default plant) | path to `plant_config.json` for multi-asset deployments |
| `PRICES_SOURCE` | `live` | `live` (SMARD HTTP) or `csv` (offline; see note) |
| `PRICES_PATH` | unset | in-container path to the SMARD-format CSV; required when `PRICES_SOURCE=csv` |
| `RESOLUTION` | `quarterhour` | optimizer time resolution |
| `FORECAST_RESOLUTION` | `hour` | matches forecasting output |
| `SCAN_INTERVAL_S` | `2` | how often the daemon polls `data/forecast/` |

### Demo mode — speeding up the replay

The default `TICK_INTERVAL_S=3600` plays back the 3-month window in real
time (3 months wall-clock). To compress the demo:

```bash
TICK_INTERVAL_S=60 docker compose -f docker-compose.forecasting.yml up -d
```

**Lower bound:** keep `TICK_INTERVAL_S` ≥ the daemon's per-cycle wall
time (roughly 5–30 s including SMARD fetch + solve). If the replay
writes forecasts faster than the daemon can solve them, the scanner's
*newest-wins* logic silently skips intermediate solve_times and the
dispatch history grows holes. ~60 s is safe; 5 s is not.

### Offline prices (`PRICES_SOURCE=csv`)

`docker-compose.yml` does not bind-mount a prices CSV by default — the
`live` path (SMARD HTTP) is the production setup. To run without
network access, add a mount to the `optimization` service's `volumes`
and set the two envs:

```yaml
# additions to docker-compose.yml under services.optimization
environment:
  PRICES_SOURCE: csv
  PRICES_PATH: /shared/smard/historical.csv
volumes:
  - ./data/smard:/shared/smard:ro   # add alongside the existing mounts
```

The CSV must use SMARD's semicolon-separated format and cover at least
`[earliest solve_time, latest solve_time + HORIZON_HOURS]`.

## Services

### `forecasting-replay`

Long-running. Loads the CSV at `CSV_PATH` (default
`data/forecasting/raw/raw_data_measured_demand.csv`) once at startup, then
every `TICK_INTERVAL_S` seconds advances a virtual `solve_time` by one
hour through `[csv_end − REPLAY_LOOKBACK_MONTHS, csv_end]` and writes the
resulting parquet into `data/forecast/`. When the virtual clock reaches
`csv_end` the loop **idles** — no more forecasts, heartbeat keeps
ticking. The loop deliberately does not wrap; the daemon's monotonicity
check would skip wrapped solve_times anyway. Use `scripts/reset_demo.sh`
to replay from the start.

`virt_solve_time` is persisted to `data/forecast/.replay-state.json`
after every successful tick (atomic write). A container restart resumes
from where it left off; only `reset_demo.sh` wipes that file.

### `optimization-init-state`

One-shot. On first deploy, seeds `data/state/current.json` (SoC full,
all units off, min-up/down relaxed) and `data/state/.heartbeat`.
Idempotent — exits 0 without rewriting if the state file already
exists. Compose recreates this container whenever the image is rebuilt
(`up -d --build`), so the noop path runs on every update.

When `CONFIG_FILE` is set, the seeded state's asset IDs match the
plant config; otherwise it falls back to `PlantConfig.legacy_default()`
(1 HP + 1 boiler + 1 CHP + 1 storage).

### `optimization-mpc`

Long-running daemon. Polls `data/forecast/` every `SCAN_INTERVAL_S`
seconds for parquet files with name pattern
`YYYY-MM-DDTHH-MM-SSZ.parquet`. When the newest unseen file is newer
than the last processed `solve_time`, one cycle is enqueued. Polling
(not inotify) is intentional: identical behavior on Linux and macOS
bind mounts, no platform quirks.

Per-cycle behavior:

1. Read forecast + load `data/state/current.json` + fetch DA prices.
2. Solve the MILP (HiGHS via pyomo).
3. Stage `data/dispatch/<solve_time>.parquet.tmp`, then atomically
   rename to `.parquet`.
4. Write `data/state/<solve_time>.json`, atomically retarget
   `current.json` at it.
5. Bump `data/state/.heartbeat`.

Cycles never overlap (single worker thread). The daemon catches
per-cycle failures and keeps running.

### Forecast filename format

The contract — see [`forecast_contract.md`](./optimization/forecast_contract.md)
for full detail:

- File: `data/forecast/<solve_time>.parquet`
- Filename: UTC, `Z` suffix, **hyphens** in the time portion
  (`2026-02-15T13-00-00Z.parquet`). Colons are invalid on Windows
  filesystems; the daemon's scanner regex requires hyphens.
- Atomic write (`.tmp` → rename), so the daemon never sees partials.

## Data layout

Everything under `data/` is bind-mounted into the containers under
`/shared/`:

```
data/
├── forecasting/
│   └── raw/
│       └── raw_data_measured_demand.csv     # input (read-only)
├── forecast/
│   ├── 2026-02-15T13-00-00Z.parquet         # forecasting → optimization
│   ├── .heartbeat                            # forecasting healthcheck
│   └── .replay-state.json                    # replay resume marker
├── state/                                    # persistent across restarts
│   ├── current.json -> 2026-02-15T13-00-00Z.json   (symlink)
│   ├── 2026-02-15T12-00-00Z.json
│   ├── 2026-02-15T13-00-00Z.json
│   └── .heartbeat                            # optimization healthcheck
├── dispatch/                                 # output
│   └── 2026-02-15T13-00-00Z.parquet
└── config/                                   # optional plant config
    └── plant_config.json
```

`state/` is append-only with a moving symlink from the very first
deploy. Past states stay on disk for audit and rollback.

## Plant config (modular assets)

The optimizer ships with `PlantConfig.legacy_default()` baked in (1 HP,
1 boiler, 1 CHP, 1 storage). For the default plant **no
`plant_config.json` is needed**.

To deploy a different plant:

```bash
# 1. Generate a starter config
optimization-write-default-config /opt/dispatch/data/config/plant_config.json
# edit: add/remove HPs/boilers/CHPs/storages, adjust limits

# 2. Point compose at it (e.g. via /opt/dispatch/.env):
echo 'CONFIG_FILE=/shared/config/plant_config.json' >> /opt/dispatch/.env

# 3. Stop the daemon and clear the old state. init-state is idempotent
#    on existing files, so a re-seed only happens once the state file
#    is gone. State is part of an MPC trajectory — discarding it for a
#    plant-topology change is intended; the next cycle cold-starts.
docker compose stop optimization
rm -f data/state/current.json data/state/*.json

# 4. Re-seed state with asset IDs matching the new config, then bring
#    the stack back up.
docker compose up -d --build
```

Both `init-state` and the daemon read the same `CONFIG_FILE` env, so
asset IDs stay in sync. The daemon loads the config once at startup;
editing the file mid-run has no effect until restart (by design — a
half-edited config can never reach the solver).

`state.covers(config)` is called per cycle inside `build_model`; on
config/state drift the cycle raises `ValueError` and the daemon's
worker logs the traceback. The daemon stays up but makes no progress
until the state is re-seeded — fails loud instead of silently
mis-dispatching.

`plant_config.json` is JSON with a `schema_version` field; each asset
has a globally unique string `id` that also appears in the state file.

## Smoke test (optional)

To validate the optimization path without the replay loop (e.g. before
loading a new CSV, or to test a plant config change), there's a helper
that drops a single contract-shaped forecast for a `now`-anchored
solve_time:

```bash
# Requires uv on the host: curl -LsSf https://astral.sh/uv/install.sh | sh
uv run python scripts/sim_forecaster.py once \
    --forecast-dir data/forecast \
    --offset-h 1 \
    --horizon-hours 35 \
    --demand-mw-th 10.0

docker compose logs -f optimization
```

Expect to see `scan: enqueueing forecast ...`, `solved in ... status=optimal`,
then `cycle ok: current.json -> <solve_time>.json` within ~30 s.

This works whether or not `forecasting-replay` is running. If both run,
note that `sim_forecaster` writes a `now`-anchored solve_time while
replay writes historical ones — the daemon's newest-wins logic will
pick whichever timestamp is larger.

## Operations

### Watch the daemons

```bash
docker compose logs -f optimization
docker compose -f docker-compose.forecasting.yml logs -f forecasting
```

A successful optimization cycle:

```
INFO optimization.daemon: scan: enqueueing forecast 2026-02-15T13:00:00+00:00
INFO optimization.run:    loaded state from /shared/state/current.json (total SoC=180.5)
INFO optimization.run:    hybrid mode: hourly demand forward-filled to 15-min, common start=2026-02-15 13:00:00+00:00
INFO optimization.run:    horizon=140 slots (35.0h @ quarterhour); forecast=140, prices=140 available
INFO optimization.run:    solved in 0.78s, status=optimal, objective=12345 EUR
INFO optimization.run:    wrote dispatch -> /shared/dispatch/...parquet.tmp, state -> /shared/state/...json
INFO optimization.daemon: cycle ok: current.json -> 2026-02-15T13-00-00Z.json
```

A successful replay tick:

```
INFO forecasting.replay_loop: tick: solve_time=2026-02-15T13:00:00+00:00 history_rows=17424 model=combined_seasonal horizon=35h
INFO forecasting.replay_loop: tick done: wrote /shared/forecast/2026-02-15T13-00-00Z.parquet
```

Useful filters:

```bash
docker compose logs optimization | grep "cycle ok"
docker compose logs optimization | grep -iE "error|exception|infeasible"
```

### Restart commands

```bash
# Restart one stack independently:
docker compose -f docker-compose.forecasting.yml restart forecasting
docker compose restart optimization

# Stop without removing containers:
docker compose stop
docker compose -f docker-compose.forecasting.yml stop

# Full takedown (containers gone, data/ preserved):
docker compose down
docker compose -f docker-compose.forecasting.yml down
```

Restarting `forecasting` alone resumes at the next unprocessed
`solve_time` thanks to `.replay-state.json`. Restarting `optimization`
alone picks up whatever forecast is currently newest on disk — **any
forecasts written while the daemon was down are silently skipped**
because the scanner is newest-wins. For replay this means dispatch gaps;
for live operation it means at most one missed hour.

### Inspect a cycle's output

```bash
ls -lt data/dispatch/ | head
docker compose run --rm --no-deps optimization \
    -c "import pandas as pd; print(pd.read_parquet('/shared/dispatch/<file>.parquet'))"
readlink data/state/current.json
```

The `docker compose run` form uses the optimizer image (pandas + pyarrow
already installed), so no host-side Python required.

### Trigger a cycle manually (debugging)

```bash
docker compose run --rm --no-deps optimization \
    -m optimization.run \
    --solve-time 2026-02-15T13:00:00+00:00 \
    --forecast-path /shared/forecast/2026-02-15T13-00-00Z.parquet \
    --state-in     /shared/state/current.json \
    --state-out    /shared/state/manual-debug.json \
    --dispatch-out /shared/dispatch/manual-debug.parquet
```

`--no-deps` skips the `init-state` dependency. The manual run doesn't
advance `current.json` — use it to reproduce failures, not to seed state.

## Updates

```bash
cd /opt/dispatch
git pull

docker compose -f docker-compose.forecasting.yml up -d --build
docker compose up -d --build
```

State persists through the rebuild. Each stack is down for ~10–30 s
during its rebuild. As noted above, forecasts written while the
optimization daemon is down may be skipped — do updates well inside
the hour, not in the last few minutes before the top of the next hour.

## Demo reset

To restart the replay from the beginning of the window:

```bash
scripts/reset_demo.sh           # interactive
scripts/reset_demo.sh --yes     # non-interactive
```

This tears down both stacks, wipes `data/forecast/`, `data/state/`,
and `data/dispatch/`, then rebuilds and restarts. Required because the
daemon's monotonicity check would otherwise refuse to re-process the
same solve_times it has already seen; without a state wipe the daemon
would sit idle after restart.

`data/forecasting/raw/` is preserved — the CSV stays in place.

## Failure semantics

### Optimization exit codes

Per-cycle exit codes from `optimization.run` (visible in daemon logs):

| Exit | Meaning | Action |
|---|---|---|
| 0 | Success, dispatch + state written | none |
| 1 | Recoverable: forecast schema bad, SMARD API down, horizon too short | next cycle is a fresh attempt |
| 2 | Solver infeasible | investigate; state was not advanced; last good plan is in `data/dispatch/` |
| 3 | Unexpected error (e.g. SMARD schema break) | open an issue with the stack trace |

There is intentionally **no per-hour automatic retry**. If a cycle
fails, the next forecast file is a fresh attempt with fresh inputs.

### Replay failure modes

| Symptom | Cause | Fix |
|---|---|---|
| Container exits with code 2 at startup | `CSV_PATH` not found or `MODEL` unknown | check mount + env |
| Container restart-loops forever | Same as above, or CSV unreadable | inspect `docker compose logs forecasting`; check `data/forecasting/raw/` ownership |
| `replay window exhausted; idling` in logs | virtual clock reached `csv_end` | expected at the end of the window; use `reset_demo.sh` to replay |
| Daemon idle, `newest forecast ... not newer than processed state` warning | `data/forecast/` was wiped (incl. `.replay-state.json`) without wiping `data/state/` | run `reset_demo.sh` |

### Healthchecks

Both containers expose a healthcheck on a heartbeat file mtime:

- Forecasting: `data/forecast/.heartbeat`, bumped at startup and after
  every tick (including idle ticks).
- Optimization: `data/state/.heartbeat`, seeded by `init-state` and
  bumped after every successful cycle.

Either container flips to `unhealthy` if its heartbeat is older than
90 min. Nothing pages anyone — wire up monitoring against
`docker inspect <container> --format '{{.State.Health.Status}}'` or
the absence of fresh files in `data/dispatch/`.

## State recovery

Basic operator playbook for manual intervention. Not a complete
disaster-recovery strategy.

If `data/state/current.json` is corrupt or its target is missing:

```bash
docker compose stop optimization

# Move the broken symlink aside (don't delete — diagnostic value)
mv data/state/current.json data/state/current.broken.$(date +%s)

# Re-seed
docker compose run --rm init-state

docker compose up -d optimization
```

A fresh init writes SoC = full (200 MWh_th), all units off, with
min-up/down constraints relaxed. The first cycle's plan is a defensible
baseline.

To roll back to a known-good past state instead:

```bash
ls data/state/   # pick a dated *.json
cd data/state
ln -sfn 2026-02-15T13-00-00Z.json current.json.tmp && mv current.json.tmp current.json
cd -
docker compose up -d optimization
```

## Dev-env note: pyomo / highspy pins

`pyproject.toml` pins `pyomo~=6.10.0` and `highspy==1.14.0`. The pins
exist so the regression pin (`tests/optimization/test_regression_pin.py`)
stays bit-identical against HiGHS branching changes. Bump deliberately
and re-run the pin if you do.

## What this deployment does NOT do

- **No SCADA / plant control feedback.** Output is advisory dispatch
  parquet files. Whatever consumes them is out of scope.
- **No live demand feed yet.** The forecasting container runs a CSV
  replay; the cutover to a live feed is a container swap (same image
  interface, different data source). See
  [`docs/forecasting/hourly_inference_pipeline.md`](./forecasting/hourly_inference_pipeline.md).
- **No alerting.** Healthchecks flag stale heartbeats; nothing pages.
- **No full outage recovery.** Long downtime, large forecast backlogs,
  and automatic catch-up across multiple missed hours are not handled.
- **No multi-host / HA.** Single VM, single daemon. State is local to
  the host.
- **No automatic data rotation.** `data/dispatch/`, `data/state/`,
  `data/forecast/` grow linearly. Prune manually if disk becomes a
  concern.
