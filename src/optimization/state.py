"""DispatchState — carry-over between hourly solves.

Critical: this is the SAFETY-CRITICAL data structure for hourly MPC. Every hour's
solve depends on the realized state at the end of the previous hour's commit.
Bugs here surface as silently wrong dispatch (units restarting unnecessarily,
storage drifting from reality).

State is keyed by asset id, matching the IDs in ``PlantConfig``. ``units``
holds 0/1 + time-in-state for HP/boiler/CHP. ``storages`` holds SoC per
storage.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from optimization.config import PlantConfig

# Time-in-state sentinel for "long enough that min-up/down is non-binding".
TIS_LONG = 999

# Bump on every breaking change to the on-disk schema. Old files are rejected
# by load() with an actionable error pointing at the migration path.
# v1: pre-modular flat fields (sto_soc_mwh_th, hp_on, boiler_on, ...).
# v2: per-asset units{} + storages{} dicts.
SCHEMA_VERSION = 2


@dataclass
class UnitState:
    """State of a unit-commitment asset (HP / boiler / CHP) at an interval boundary.

    Time-in-state is in 15-min steps (not hours), matching the MILP grid.
    """

    on: int                      # 0 or 1
    time_in_state_steps: int     # how long the unit has been in current state


@dataclass
class StorageState:
    """State of a thermal storage at an interval boundary."""

    soc_mwh_th: float


@dataclass
class DispatchState:
    """State at a 15-min interval boundary, carried into the next solve.

    ``units`` is keyed by asset id and contains every unit-commitment asset
    (HPs, boilers, CHPs). ``storages`` is keyed by storage id.
    """

    timestamp: pd.Timestamp                  # tz-aware Berlin, end of last commit window
    units: dict[str, UnitState] = field(default_factory=dict)
    storages: dict[str, StorageState] = field(default_factory=dict)

    @classmethod
    def cold_start(
        cls,
        config: "PlantConfig",
        timestamp: pd.Timestamp,
        soc_init_overrides: dict[str, float] | None = None,
    ) -> "DispatchState":
        """Default state when no prior commit exists (e.g., first deployment).

        All units off with TIS_LONG (no min-up/down constraints binding).
        Storages start at their per-asset ``soc_init_mwh_th``, unless an
        override is supplied.
        """
        overrides = soc_init_overrides or {}
        units: dict[str, UnitState] = {}
        for hp in config.heat_pumps:
            units[hp.id] = UnitState(on=0, time_in_state_steps=TIS_LONG)
        for b in config.boilers:
            units[b.id] = UnitState(on=0, time_in_state_steps=TIS_LONG)
        for c in config.chps:
            units[c.id] = UnitState(on=0, time_in_state_steps=TIS_LONG)
        storages = {
            s.id: StorageState(soc_mwh_th=overrides.get(s.id, s.soc_init_mwh_th))
            for s in config.storages
        }
        return cls(timestamp=timestamp, units=units, storages=storages)

    def save(self, path: Path) -> None:
        """Serialize to JSON via atomic tmp-then-replace.

        Atomic = either the file is fully written or the previous version stays
        untouched. Important because a partial state file would crash the next
        hourly run.
        """
        payload = {
            "schema_version": SCHEMA_VERSION,
            "timestamp": self.timestamp.isoformat(),
            "units": {k: asdict(v) for k, v in self.units.items()},
            "storages": {k: asdict(v) for k, v in self.storages.items()},
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> "DispatchState":
        """Deserialize from JSON. Raises FileNotFoundError if missing —
        caller decides whether to fall back to cold_start().

        Validates ``schema_version`` first. Old (v1, pre-modular flat) state
        files are rejected with a clear migration message so partial reads can
        never silently lose asset state.
        """
        path = Path(path)
        payload = json.loads(path.read_text())
        version = payload.get("schema_version")
        if version is None:
            # Pre-versioning files (v1 flat schema or any other layout).
            raise ValueError(
                f"state file {path} is missing schema_version. "
                f"This is a pre-v2 state file (likely pre-modular-assets). "
                f"Re-seed via `python -m optimization.init_state --state-out {path} "
                f"--solve-time <ISO-ts> --force [--config-file <plant_config.json>]` "
                f"to cold-start. Pass --config-file iff the daemon will run with "
                f"a custom plant config; otherwise the legacy 1-of-each default is seeded."
            )
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"state file {path} has schema_version={version}, expected "
                f"{SCHEMA_VERSION}. No migration available — re-seed via "
                f"`python -m optimization.init_state --state-out {path} "
                f"--solve-time <ISO-ts> --force [--config-file <plant_config.json>]` "
                f"to cold-start. Pass --config-file iff the daemon will run with "
                f"a custom plant config; otherwise the legacy 1-of-each default is seeded."
            )
        if "units" not in payload or "storages" not in payload:
            raise ValueError(
                f"state file {path} has unexpected schema (missing units/storages); "
                f"keys present: {sorted(payload)}"
            )
        ts = pd.Timestamp(payload["timestamp"])
        if ts.tzinfo is None:
            raise ValueError(f"State timestamp must be tz-aware, got {ts!r}")
        units = {
            asset_id: UnitState(
                on=int(v["on"]),
                time_in_state_steps=int(v["time_in_state_steps"]),
            )
            for asset_id, v in payload["units"].items()
        }
        storages = {
            asset_id: StorageState(soc_mwh_th=float(v["soc_mwh_th"]))
            for asset_id, v in payload["storages"].items()
        }
        return cls(timestamp=ts, units=units, storages=storages)

    def feasible_against(self, config: "PlantConfig") -> list[str]:
        """Return a list of infeasibility messages, empty if state is feasible.

        Catches parameter changes that the asset-id-set gate (used by live
        reload) can't see: most importantly, raising ``floor_mwh_th`` above the
        currently-stored SoC, or shrinking ``capacity_mwh_th`` below it. Either
        would make the next MILP solve start from an infeasible initial SoC.

        Unit ``time_in_state_steps`` is intentionally not checked: the MILP
        enforces min-up/min-down against the new bounds, and a longer-than-
        necessary on/off run is legal under any (positive) min_up/min_down.

        Assumes the asset id sets already match (caller's gate). Storages
        present in state but missing from config are skipped here; ``covers``
        reports them.
        """
        problems: list[str] = []
        storages_by_id = {s.id: s for s in config.storages}
        # SoC is a float carried through the solver; tolerate sub-Wh rounding
        # against an integer-MWh edit. Anything beyond that is a real bound bust.
        tol = 1e-9
        for sid, sstate in self.storages.items():
            s = storages_by_id.get(sid)
            if s is None:
                continue
            if sstate.soc_mwh_th < s.floor_mwh_th - tol:
                problems.append(
                    f"storage {sid!r}: current SoC={sstate.soc_mwh_th} is below "
                    f"new floor_mwh_th={s.floor_mwh_th}"
                )
            elif sstate.soc_mwh_th > s.capacity_mwh_th + tol:
                problems.append(
                    f"storage {sid!r}: current SoC={sstate.soc_mwh_th} exceeds "
                    f"new capacity_mwh_th={s.capacity_mwh_th}"
                )
        return problems

    def covers(self, config: "PlantConfig") -> None:
        """Validate that this state has an entry for every asset in `config`.

        Raises ValueError listing missing or extra assets. Used in build_model
        to fail loud on config/state drift instead of silently mis-initializing.
        Reconcile by re-seeding via ``optimization-init-state --force
        --config-file <plant_config.json>``.
        """
        cfg_unit_ids = set(config.all_unit_ids())
        cfg_storage_ids = {s.id for s in config.storages}
        st_unit_ids = set(self.units)
        st_storage_ids = set(self.storages)
        missing_units = sorted(cfg_unit_ids - st_unit_ids)
        extra_units = sorted(st_unit_ids - cfg_unit_ids)
        missing_storages = sorted(cfg_storage_ids - st_storage_ids)
        extra_storages = sorted(st_storage_ids - cfg_storage_ids)
        if not (missing_units or extra_units or missing_storages or extra_storages):
            return

        problems: list[str] = []
        if missing_units:
            problems.append(f"missing unit state for: {missing_units}")
        if extra_units:
            problems.append(f"unit state for unknown asset: {extra_units}")
        if missing_storages:
            problems.append(f"missing storage state for: {missing_storages}")
        if extra_storages:
            problems.append(f"storage state for unknown asset: {extra_storages}")
        msg = (
            "state/config mismatch: " + "; ".join(problems)
            + ". Re-seed with: optimization-init-state --state-out <path> "
            "--force --config-file <plant_config.json>"
        )
        raise ValueError(msg)
