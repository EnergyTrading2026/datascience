"""Write an initial DispatchState file if none exists.

Idempotent: if the target path already exists, exit 0 without rewriting the
state. If an older deployment left state behind without a heartbeat file, this
command backfills the missing heartbeat so the healthcheck starts from a sane
baseline.

This is intended to run once on a fresh deployment to seed
``state/current.json`` so the first real cycle has a state to load. It
deliberately performs no forecast/price IO and no solve, so it can run
before any forecast file is on disk.

Pass ``--config-file`` to seed against a custom PlantConfig (matching what
the daemon will run with); without it, seeds for
``PlantConfig.legacy_default()``.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

from optimization.config import PlantConfig
from optimization.state import DispatchState

logger = logging.getLogger("optimization.init_state")
CURRENT_NAME = "current.json"
HEARTBEAT_NAME = ".heartbeat"


def _atomic_symlink(target_name: str, link_path: Path) -> None:
    tmp = link_path.with_suffix(link_path.suffix + ".tmp")
    try:
        if tmp.is_symlink() or tmp.exists():
            tmp.unlink()
    except FileNotFoundError:
        pass
    tmp.symlink_to(target_name)
    os.replace(tmp, link_path)


def _bump_heartbeat(state_dir: Path) -> None:
    hb = state_dir / HEARTBEAT_NAME
    hb.touch(exist_ok=True)
    now = pd.Timestamp.now(tz="UTC").timestamp()
    os.utime(hb, (now, now))


def _state_filename(ts: pd.Timestamp) -> str:
    return f"{ts.tz_convert('UTC').strftime('%Y-%m-%dT%H:%M:%SZ')}.json"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Seed an initial DispatchState file.")
    p.add_argument(
        "--state-out",
        required=True,
        type=Path,
        help="Where to write the initial state JSON.",
    )
    p.add_argument(
        "--solve-time",
        type=lambda s: pd.Timestamp(s),
        default=None,
        help="Timestamp embedded in the state. Defaults to current top-of-hour UTC.",
    )
    p.add_argument(
        "--soc-mwh-th",
        type=float,
        default=None,
        help="Per-storage override of the config's soc_init_mwh_th. "
             "If omitted, every storage gets its own soc_init_mwh_th from "
             "the config (or from legacy_default if no --config-file).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing state file (default: leave existing file untouched).",
    )
    p.add_argument(
        "--config-file",
        type=Path,
        default=None,
        help="Cold-start against this PlantConfig JSON. If omitted, seeds for "
             "PlantConfig.legacy_default(). Required when the daemon will run "
             "with --config-file pointing at a custom plant; otherwise "
             "state.covers(config) fails on the first cycle.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.state_out.exists() and not args.force:
        if not (args.state_out.parent / HEARTBEAT_NAME).exists():
            _bump_heartbeat(args.state_out.parent)
            logger.info("state already exists at %s; seeded missing heartbeat", args.state_out)
            return 0
        logger.info("state already exists at %s; nothing to do", args.state_out)
        return 0

    ts = args.solve_time or pd.Timestamp.now(tz="UTC").floor("h")
    if ts.tzinfo is None:
        logger.error("--solve-time must be tz-aware")
        return 1

    if args.config_file is not None:
        try:
            config = PlantConfig.from_json(args.config_file)
        except (FileNotFoundError, ValueError) as e:
            logger.error("config load failed (%s): %s", args.config_file, e)
            return 1
        logger.info("seeding state for plant config %s", args.config_file)
    else:
        config = PlantConfig.legacy_default()
    # Only build SoC overrides if the operator explicitly passed --soc-mwh-th;
    # otherwise cold_start picks each storage's own soc_init_mwh_th from the
    # config. Building an unconditional override from the argparse default
    # used to silently overwrite per-asset config SoCs.
    overrides: dict[str, float] | None = (
        {s.id: args.soc_mwh_th for s in config.storages}
        if args.soc_mwh_th is not None
        else None
    )
    state = DispatchState.cold_start(config, ts, soc_init_overrides=overrides)
    if args.state_out.name == CURRENT_NAME:
        state_dated = args.state_out.with_name(_state_filename(ts))
        state.save(state_dated)
        _atomic_symlink(state_dated.name, args.state_out)
    else:
        state.save(args.state_out)
        state_dated = args.state_out
    _bump_heartbeat(args.state_out.parent)
    n_units = len(state.units)
    n_storages = len(state.storages)
    total_soc = sum(s.soc_mwh_th for s in state.storages.values())
    logger.info(
        "wrote initial state to %s via %s "
        "(timestamp=%s, %d units off, %d storages with total SoC=%.1f MWh_th)",
        args.state_out, state_dated, ts.isoformat(), n_units, n_storages, total_soc,
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        logger.exception("unexpected error")
        sys.exit(3)
