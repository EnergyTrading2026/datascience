"""Write an initial DispatchState file if none exists.

Idempotent: if the target path already exists, exit 0 without rewriting the
state. If an older deployment left state behind without a heartbeat file, this
command backfills the missing heartbeat so the healthcheck starts from a sane
baseline.

This is intended to run once on a fresh deployment to seed
``state/current.json`` so the first real cycle has a state to load. It
deliberately performs no forecast/price IO and no solve, so it can run
before any forecast file is on disk.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

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
        default=200.0,
        help="Initial storage SoC in MWh_th (default: 200, = full).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing state file (default: leave existing file untouched).",
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

    state = DispatchState.cold_start(ts, sto_soc_mwh_th=args.soc_mwh_th)
    if args.state_out.name == CURRENT_NAME:
        state_dated = args.state_out.with_name(_state_filename(ts))
        state.save(state_dated)
        _atomic_symlink(state_dated.name, args.state_out)
    else:
        state.save(args.state_out)
        state_dated = args.state_out
    _bump_heartbeat(args.state_out.parent)
    logger.info(
        "wrote initial state to %s via %s (timestamp=%s, SoC=%.1f, all units off)",
        args.state_out, state_dated, ts.isoformat(), args.soc_mwh_th,
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        logger.exception("unexpected error")
        sys.exit(3)
