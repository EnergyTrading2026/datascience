"""Dump PlantConfig.legacy_default() to a JSON file the operator can edit.

Usage:
    python -m optimization.write_default_config /shared/config/plant_config.json

After editing, point run.py / backtest.py at the file with --config-file.
Refuses to overwrite an existing file unless --force is passed.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from optimization.config import PlantConfig

logger = logging.getLogger("optimization.write_default_config")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Write PlantConfig.legacy_default() to JSON as a starter file.",
    )
    p.add_argument("path", type=Path, help="Output path for the config JSON.")
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the file if it already exists (default: refuse).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.path.exists() and not args.force:
        logger.error("%s already exists; pass --force to overwrite", args.path)
        return 1
    PlantConfig.legacy_default().to_json(args.path)
    logger.info("wrote default plant config to %s", args.path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
