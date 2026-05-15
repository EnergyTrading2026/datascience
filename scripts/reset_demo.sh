#!/usr/bin/env bash
# Reset the MVP demo: tear down both compose stacks, clear the shared state,
# and bring them back up.
#
# Why this exists: the replay loop walks linearly from csv_end - 3 months to
# csv_end. The optimization daemon enforces solve_time monotonicity. Restarting
# the replay container alone would re-write the same solve_times the daemon
# has already processed -> daemon skips them all. To run the demo again we
# need to wipe both sides: forecast files AND daemon state.
#
# Usage:
#     scripts/reset_demo.sh            # prompts for confirmation
#     scripts/reset_demo.sh --yes      # non-interactive

set -euo pipefail

CONFIRM=0
for arg in "$@"; do
    case "$arg" in
        -y|--yes) CONFIRM=1 ;;
        -h|--help)
            sed -n '2,13p' "$0"
            exit 0
            ;;
        *)
            echo "unknown argument: $arg" >&2
            exit 2
            ;;
    esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

FORECAST_DIR="$ROOT/data/forecast"
STATE_DIR="$ROOT/data/state"
DISPATCH_DIR="$ROOT/data/dispatch"

COMPOSE_OPT="docker-compose.yml"
COMPOSE_FCST="docker-compose.forecasting.yml"

echo "About to:"
echo "  - docker compose -f $COMPOSE_FCST down"
echo "  - docker compose -f $COMPOSE_OPT down"
echo "  - wipe $FORECAST_DIR/*"
echo "  - wipe $STATE_DIR/*"
echo "  - wipe $DISPATCH_DIR/*"
echo "  - rebuild and bring both stacks back up"
echo

if [[ $CONFIRM -eq 0 ]]; then
    read -r -p "Proceed? [y/N] " reply
    case "$reply" in
        y|Y|yes|YES) ;;
        *) echo "aborted"; exit 1 ;;
    esac
fi

echo ">> stopping stacks"
docker compose -f "$COMPOSE_FCST" down --remove-orphans || true
docker compose -f "$COMPOSE_OPT" down --remove-orphans || true

echo ">> wiping shared dirs"
# Container uids (1000 for optimization, 1001 for forecasting) own these files.
# Use sudo to remove cleanly; on systems where the operator already owns the
# dirs this falls back without sudo.
wipe() {
    local d="$1"
    [[ -d "$d" ]] || return 0
    if [[ -w "$d" ]] && find "$d" -mindepth 1 -maxdepth 1 -writable -quit 2>/dev/null; then
        find "$d" -mindepth 1 -delete
    else
        sudo find "$d" -mindepth 1 -delete
    fi
}
wipe "$FORECAST_DIR"
wipe "$STATE_DIR"
wipe "$DISPATCH_DIR"

echo ">> bringing stacks back up (build)"
docker compose -f "$COMPOSE_FCST" up -d --build
docker compose -f "$COMPOSE_OPT" up -d --build

echo
echo "done. follow logs:"
echo "  docker compose -f $COMPOSE_FCST logs -f forecasting"
echo "  docker compose -f $COMPOSE_OPT logs -f optimization"
