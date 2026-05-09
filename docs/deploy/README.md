# Optimization MPC — Deployment Runbook

Hourly MPC dispatch optimizer, packaged as a Docker image. The container
runs as a long-lived daemon that triggers one MPC cycle per new forecast
parquet file dropped into the shared forecast directory.

This runbook describes the first deployable Docker setup for the optimization
service. Scope is intentionally narrow: the normal operating path is covered,
while more advanced recovery and hardening for failure scenarios will be added
later.

## Server requirements

- Linux (any distro Docker supports). Image is platform-independent (built
  and validated on macOS via Docker Desktop); first production deploy is the
  first real Linux test.
- Docker Engine ≥ 24 with the Compose v2 plugin (`docker compose`, not
  `docker-compose`).
- Outbound network access:
    - **Runtime:** HTTPS to `https://www.smard.de` (live DA prices). This is
      the only outbound endpoint the daemon hits in steady-state operation.
    - **Build time only** (during `docker compose build` / `up --build`):
      additionally HTTPS to `ghcr.io` (uv image), `docker.io` /
      `registry-1.docker.io` (python base image), and `pypi.org` /
      `files.pythonhosted.org` (Python packages). On firewalled hosts that
      whitelist outbound traffic, all three must be reachable for the build.
- Disk: ~1 GB for the image, plus growth in `data/dispatch/` (one parquet
  per hour, ~5–10 KB each — roughly 50–100 MB/year) and `data/state/` (one
  JSON per hour, ~200 B each — roughly 2 MB/year). Both grow linearly and
  unbounded; no rotation is built in.
- Persistent storage for `data/state/` — must survive restarts.
- Time-sync (chrony / systemd-timesyncd). The wall clock must be accurate
  (NTP-synced). The host timezone does not matter — the container forces
  UTC internally via `TZ=Etc/UTC`.

## First-time setup

### Host directory ownership (read this first)

The container runs as a non-root user with **uid 1000 / gid 1000**. The
`data/forecast`, `data/state`, and `data/dispatch` directories are bind-mounted
into the container, so the host-side permissions are what the daemon actually
sees — the `chown` baked into the image is overridden by the bind mount and
does not help here.

The daemon must be able to:

- read `data/forecast/` (forecast files arrive here from the forecasting
  pipeline)
- read and write `data/state/` (state JSON, dated history, heartbeat, atomic
  symlink)
- write `data/dispatch/` (one parquet per cycle)

If these directories are owned by `root` (which happens when Docker
auto-creates them or when an operator runs `sudo mkdir`) or by a uid other
than 1000, the daemon will exit with permission errors on the first cycle.

The simplest, distro-independent fix is to create the directories explicitly
and chown them to uid 1000 before the first `docker compose up`:

```bash
git clone <repo-url> /opt/optimization
cd /opt/optimization
mkdir -p data/forecast data/state data/dispatch
sudo chown -R 1000:1000 data/
docker compose up -d --build
```

Notes:

- On Ubuntu / Debian, the first interactive user typically already is uid
  1000. In that case `mkdir` without `sudo` is enough and the `chown` step is
  a no-op — but running it anyway is safe and avoids surprises on hardened
  hosts where service accounts use different uids.
- The forecasting pipeline that drops files into `data/forecast/` runs
  outside this container. It must also be able to write to that directory.
  If it runs under a different uid, give the directory group-write
  permissions (e.g. `chmod 2775 data/forecast` plus matching group
  membership) rather than loosening the daemon's mount.
- The forecast file format and naming convention are defined in
  [`docs/optimization/forecast_contract.md`](../optimization/forecast_contract.md).

### Verify the install came up cleanly

```bash
docker compose ps                                                   # both services listed
docker compose logs init-state                                      # ends with exit code 0, "wrote initial state ..."
docker inspect optimization-mpc --format '{{.State.Health.Status}}' # typically 'healthy' within seconds — see note below
```

The healthcheck reports `healthy` as soon as the first probe finds a fresh
`state/.heartbeat` file. Because `init-state` seeds the heartbeat before the
daemon starts, the container usually flips to `healthy` within seconds, not
after the full 15-minute `start_period`. The 15-minute window only matters as
a grace period: failing probes during that window do not count against the
container, so a slow first cycle won't flap the status.

In normal operation, each newly dropped parquet file in
`/opt/optimization/data/forecast/` triggers one MPC cycle. The container
restarts itself on crashes (`restart: unless-stopped`).

### Smoke test with a synthetic forecast

Use this when the optimization container is up but the real forecasting
pipeline is not wired in yet, or when you want to validate a fresh server
deploy end to end.

This is optional test tooling, not part of production runtime. The optimizer
container does not need `scripts/sim_forecaster.py`, and the Docker image does
not include it. In real production, the forecasting service writes the parquet
files into `data/forecast/`; the synthetic helper is only for manual smoke
tests from a checked-out repo.

The helper script writes the same parquet shape that the optimizer expects
from forecasting:

- file: `data/forecast/<solve_time>.parquet`
- filename timestamp: UTC with `Z` suffix, e.g. `2026-05-09T10:00:00Z.parquet`
- index: tz-aware `Europe/Berlin` hourly `DatetimeIndex`
- column: `demand_mw_th`
- default horizon: 35 hours

From the repo root on the host:

```bash
# Make sure the daemon is running.
docker compose up -d --build

# Drop one forecast for the next top-of-hour solve_time.
uv run python scripts/sim_forecaster.py once \
    --forecast-dir data/forecast \
    --offset-h 1 \
    --horizon-hours 35 \
    --demand-mw-th 10.0

# Watch the daemon pick it up, fetch live SMARD prices, solve, and persist.
docker compose logs -f optimization
```

Expected result:

- logs contain `scan: enqueueing forecast ...`
- logs contain `hybrid mode: hourly demand forward-filled to 15-min`
- logs contain `solved in ... status=optimal`
- logs contain `cycle ok: current.json -> <solve_time>.json`
- `data/dispatch/<solve_time>.parquet` exists
- `data/state/current.json` points at `data/state/<solve_time>.json`

Inspect the output:

```bash
ls -lt data/dispatch/ | head
docker compose run --rm --no-deps optimization \
    -c "import pandas as pd; print(pd.read_parquet('/shared/dispatch/<solve_time>.parquet'))"
readlink data/state/current.json
```

(The `docker compose run` form uses the optimizer image, which already has
pandas + pyarrow, so it works on minimal server images that don't ship
Python.)

For a longer soak test, the same helper can write one forecast per real hour:

```bash
uv run python scripts/sim_forecaster.py loop \
    --forecast-dir data/forecast \
    --period-s 3600 \
    --offset-h 1
```

If you deploy from a minimal artifact that contains only the Dockerfile,
Compose file, and source package, you can skip this section entirely. Production
forecasting remains a separate service that writes contract-compliant parquet
files into `data/forecast/`.

## What's inside

Two services in `docker-compose.yml`:

- **`init-state`** — one-shot. On first deploy, writes a synthetic initial
  state (SoC = 200 MWh_th, all units off, min-up/down constraints relaxed)
  to a dated JSON file and points `data/state/current.json` at it. It also
  seeds `data/state/.heartbeat` so a fresh-but-idle deployment starts healthy.
  On subsequent `docker compose up` runs the state file already exists, so
  this service exits 0 immediately.
- **`optimization`** — long-running daemon. Polls `data/forecast/` every
  `SCAN_INTERVAL_S` seconds (default 2) for new `<solve_time>.parquet`
  files (filename format `YYYY-MM-DDTHH:MM:SSZ.parquet`, UTC, Z suffix).
  When the newest file on disk is newer than the last enqueued/processed
  solve_time, one cycle is enqueued. Polling (rather than inotify) is
  intentional: identical behavior on Linux and macOS bind mounts, one
  mechanism, no platform-specific event semantics. At a 2 s interval the
  cost is negligible and pickup latency is well below the hourly cadence.

After every successful cycle the daemon:
1. Writes the dispatch parquet to `data/dispatch/<solve_time>.parquet`.
2. Writes a dated state file to `data/state/<solve_time>.json` and
   atomically retargets the `current.json` symlink at it.
3. Bumps `data/state/.heartbeat` (used by the container's healthcheck).

## Data layout

Everything lives under `/opt/optimization/data/` (bind-mounted into the
container at `/shared/`):

```
data/
├── forecast/                              # input  (read-only mount)
│   └── 2026-05-07T13:00:00Z.parquet
├── state/                                 # persistent across restarts
│   ├── current.json -> 2026-05-07T13:00:00Z.json   (symlink)
│   ├── 2026-05-07T12:00:00Z.json
│   ├── 2026-05-07T13:00:00Z.json
│   └── .heartbeat
└── dispatch/                              # output
    └── 2026-05-07T13:00:00Z.parquet
```

State is append-only with a moving symlink from the very first deploy. Past
states stay on disk for audit and rollback.

## Operations

### Watch the daemon

```bash
docker compose logs -f optimization
```

Each cycle logs: detected file, started cycle, solver result, written
paths. Crashes log a stack trace; the daemon restarts automatically.

### Generate a test forecast

For manual testing without the real forecasting service:

```bash
uv run python scripts/sim_forecaster.py once --forecast-dir data/forecast --offset-h 1
```

This writes a single hourly, contract-shaped forecast for the next top of the
hour. The daemon should pick it up automatically. Use `--help` to see knobs
for demand level, horizon length, and loop mode. This command requires the
repo's `scripts/` directory on the host; it is not needed by production.

### Inspect a cycle's output

```bash
ls -lt /opt/optimization/data/dispatch/ | head
docker compose run --rm --no-deps optimization \
    -c "import pandas as pd; print(pd.read_parquet('/shared/dispatch/<file>.parquet'))"
```

The `docker compose run` form runs inside the optimizer image (pandas +
pyarrow already installed) and uses the in-container path `/shared/dispatch/`.
No host-side Python required.

### Trigger a cycle manually (debugging)

The daemon picks up files automatically. To force a cycle for a specific
solve_time without dropping a forecast file, run a one-off container:

```bash
docker compose run --rm --no-deps optimization \
    -m optimization.run \
    --solve-time 2026-05-07T13:00:00+00:00 \
    --forecast-path /shared/forecast/2026-05-07T13:00:00Z.parquet \
    --state-in     /shared/state/current.json \
    --state-out    /shared/state/manual-debug.json \
    --dispatch-out /shared/dispatch/manual-debug.parquet
```

`--no-deps` is important: without it, `docker compose run optimization`
also triggers `init-state` (because of `depends_on`). That's harmless —
`init-state` is idempotent and will exit early if state already exists —
but it's noise during debugging. With `--no-deps`, only the optimization
container runs.

This bypasses the symlink update — the manual run won't advance
`current.json`. Use it to reproduce a failure, not to seed state.

## Updating

```bash
cd /opt/optimization
git pull
docker compose up -d --build
```

The daemon stops, the image rebuilds, the daemon restarts. State persists
through the restart (it lives in the bind-mounted `data/` dir, not the
container).

**Downtime / cycle loss:** the rebuild + restart takes roughly 10–30 seconds
during which the daemon is not running. If a forecast file lands in
`data/forecast/` exactly during that window, it will still be picked up
when the daemon comes back (the scanner picks the newest file on disk).
But if the forecasting pipeline writes a *newer* file before the daemon
returns, the older file in the gap is skipped — the daemon only enqueues
the newest unseen solve_time. To stay on the safe side, do updates well
inside an hour, not in the last few minutes before the top of the next
hour.

## Failure semantics

The daemon catches per-cycle failures and keeps running. Per-cycle exit
codes from `optimization.run` (visible in the daemon logs):

| Exit | Meaning | Action |
|---|---|---|
| 0 | Success, dispatch + state written | none |
| 1 | Recoverable: forecast schema bad, SMARD API down, horizon too short | next cycle is a fresh attempt |
| 2 | Solver infeasible | investigate. State was not advanced. Last good plan in `dispatch/`. |
| 3 | Unexpected error (e.g. SMARD schema break) | open an issue with the stack trace |

There is intentionally **no per-hour automatic retry**. If a cycle fails,
the next forecast file is a fresh attempt with fresh inputs. The scanner
will not re-enqueue the same solve_time after a failure (dedupe is by
filename / solve_time). Sub-hour retries would risk solving twice with
overlapping commit windows.

This `v1` deployment does not yet try to provide a full recovery strategy for
longer outages, large forecast backlogs, or operator-free catch-up across
multiple missed hours. Those scenarios are out of scope for this first
Dockerized deployment and should be treated as manual intervention cases.

The container's healthcheck starts out healthy because `init-state` seeds the
heartbeat once. After that it flips to `unhealthy` if no successful cycle has
occurred in 90 minutes (no fresh `state/.heartbeat`). It does not auto-page
anyone — wire up monitoring of choice against
`docker inspect optimization-mpc --format '{{.State.Health.Status}}'`
or the absence of fresh files in `data/dispatch/`.

## State recovery

This section is a basic operator playbook for manual intervention. It is not
meant to be a complete disaster-recovery strategy yet.

If `data/state/current.json` is corrupt or its target is missing:

```bash
# 1. Stop the daemon
docker compose stop optimization

# 2. Move the broken symlink aside (do not delete — diagnostic value).
#    `mv` on the symlink itself; the target file (if it still exists) stays
#    in place under its dated name so it can also be inspected.
mv /opt/optimization/data/state/current.json \
   /opt/optimization/data/state/current.broken.$(date +%s)

# 3. Re-seed the synthetic initial state
docker compose run --rm init-state

# 4. Restart
docker compose up -d optimization
```

A fresh init writes SoC = full (200 MWh_th), all units off, with min-up/down
constraints relaxed (TIS = 999 sentinel). The first cycle's plan is a
defensible baseline, not a warm-up phase that needs to be discounted.

If you want to roll back to a known-good past state instead:

```bash
ls /opt/optimization/data/state/   # pick a dated *.json
cd /opt/optimization/data/state
ln -sfn 2026-05-07T13:00:00Z.json current.json.tmp && mv current.json.tmp current.json
docker compose up -d optimization
```

## What this deployment does NOT do

- **No SCADA / plant control feedback.** Output is advisory dispatch
  parquet files. Whatever consumes them is out of scope.
- **No forecast pipeline.** The forecaster is a separate component that
  drops parquet files into `data/forecast/`. If it stops, the daemon
  simply waits — no false runs.
- **No alerting.** The healthcheck flags stale heartbeats but nothing
  pages anyone. Hook your monitoring of choice against the heartbeat
  file or the dispatch directory.
- **No full outage recovery yet.** Long downtime, large forecast backlogs,
  and automatic catch-up across multiple missed hours are not part of this
  first Dockerized version.
- **No multi-host / HA.** Single VM, single daemon. State is local to the
  host.
