"""Read-only dry-run for ``plant_config.json`` edits.

Console entry: ``optimization-config-diff``. Reads a *proposed* config file
(typically the operator's local edit), the *current* config the daemon is
running on, and the live ``DispatchState`` — and prints what would happen
if the operator copied the proposed file over the live one.

The CLI never touches the live config file, the daemon, or the state file.
It exits 0 when the proposed config is apply-able (no validation errors,
no family moves, post-migration state still feasible) and 1 otherwise, so
operators can wire it into a deploy pipeline as a gate.

The same checks the daemon's reload path runs at every cycle boundary are
re-used here:

  * ``validate_plant_payload`` for hard errors and warnings;
  * ``ConfigDiff.between`` for the structural classification;
  * ``DispatchState.migrate_to`` + ``feasible_against`` for the
    post-migration feasibility gate;
  * family-move detection (id present in both removed and added sets).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from optimization.config import CONFIG_SCHEMA_VERSION, PlantConfig
from optimization.config_diff import ConfigDiff, asset_param_changes
from optimization.config_validation import (
    ConfigValidationError,
    ValidationResult,
    validate_plant_payload,
)
from optimization.state import DispatchState


def _read_payload(path: Path) -> dict:
    """Parse the JSON file; let JSONDecodeError surface so main() can format it."""
    return json.loads(path.read_text())


def _format_validation(result: ValidationResult) -> str:
    lines = [
        "VALIDATION (proposed)",
        f"  errors:   {len(result.errors)}",
        f"  warnings: {len(result.warnings)}",
    ]
    for issue in result.errors:
        lines.append(f"    [ERROR {issue.code}] {issue.path}: {issue.message}")
    for issue in result.warnings:
        lines.append(f"    [WARNING {issue.code}] {issue.path}: {issue.message}")
    return "\n".join(lines)


def _family_of(cfg: PlantConfig, asset_id: str) -> str:
    """Best-effort family lookup for printing. Returns 'asset' if unknown."""
    if any(hp.id == asset_id for hp in cfg.heat_pumps):
        return "heat_pump"
    if any(b.id == asset_id for b in cfg.boilers):
        return "boiler"
    if any(c.id == asset_id for c in cfg.chps):
        return "chp"
    if any(s.id == asset_id for s in cfg.storages):
        return "storage"
    return "asset"


def _format_added(cfg: PlantConfig, ids: frozenset[str], kind: str) -> list[str]:
    """``+ added <family> 'id' (field=value, ...)`` for each added asset."""
    lines: list[str] = []
    for aid in sorted(ids):
        family = _family_of(cfg, aid)
        asset = _find_asset(cfg, aid)
        details = _short_asdict(asset) if asset is not None else ""
        lines.append(f"  + added {family} {aid!r}{details}")
    return lines


def _format_removed(cfg: PlantConfig, ids: frozenset[str], kind: str) -> list[str]:
    lines: list[str] = []
    for aid in sorted(ids):
        family = _family_of(cfg, aid)
        asset = _find_asset(cfg, aid)
        details = _short_asdict(asset) if asset is not None else ""
        lines.append(f"  - removed {family} {aid!r}{details}")
    return lines


def _format_param_changes(
    current: PlantConfig, candidate: PlantConfig, ids: frozenset[str],
) -> list[str]:
    """``~ changed <family> 'id': field: old → new`` per asset.

    Sorted by id (deterministic for tests). Multi-field changes are emitted
    as a header + one indented line per field, keeping the operator's eye
    on which field belongs to which asset.
    """
    lines: list[str] = []
    for aid in sorted(ids):
        family = _family_of(cfg=current, asset_id=aid)
        lines.append(f"  ~ changed {family} {aid!r}:")
        for field, (old, new) in sorted(asset_param_changes(current, candidate, aid).items()):
            lines.append(f"      {field}: {old} → {new}")
    return lines


def _format_globals_changes(
    current: PlantConfig, candidate: PlantConfig,
) -> list[str]:
    lines: list[str] = []
    fields = (
        "dt_h", "gas_price_eur_mwh_hs",
        "co2_factor_t_per_mwh_hs", "co2_price_eur_per_t",
    )
    for f in fields:
        old, new = getattr(current, f), getattr(candidate, f)
        if old != new:
            lines.append(f"  globals: {f} {old} → {new}")
    return lines


def _format_disable(
    title: str, ids: frozenset[str], current: PlantConfig,
) -> list[str]:
    return [
        f"  disabled: {title} {aid!r} ({_family_of(current, aid)})"
        for aid in sorted(ids)
    ]


def _find_asset(cfg: PlantConfig, asset_id: str):
    for family in (cfg.heat_pumps, cfg.boilers, cfg.chps, cfg.storages):
        for a in family:
            if a.id == asset_id:
                return a
    return None


def _short_asdict(asset) -> str:
    """Inline-rendering of a dataclass asset for added/removed lines."""
    d = asdict(asset)
    parts = [f"{k}={v}" for k, v in d.items() if k != "id"]
    return f" ({', '.join(parts)})" if parts else ""


def _format_diff(current: PlantConfig, candidate: PlantConfig, diff: ConfigDiff) -> str:
    lines: list[str] = ["CONFIG DIFF (proposed vs. current)"]
    lines.extend(_format_added(candidate, diff.added_unit_ids, "unit"))
    lines.extend(_format_added(candidate, diff.added_storage_ids, "storage"))
    lines.extend(_format_removed(current, diff.removed_unit_ids, "unit"))
    lines.extend(_format_removed(current, diff.removed_storage_ids, "storage"))
    lines.extend(_format_param_changes(current, candidate, diff.param_changed_unit_ids))
    lines.extend(_format_param_changes(current, candidate, diff.param_changed_storage_ids))
    lines.extend(_format_globals_changes(current, candidate))
    lines.extend(_format_disable("+", diff.disabled_added, candidate))
    lines.extend(_format_disable("-", diff.disabled_removed, current))
    if len(lines) == 1:
        lines.append("  (no observable change)")
    return "\n".join(lines)


def _format_migration(
    state: DispatchState | None,
    diff: ConfigDiff,
    candidate: PlantConfig,
    state_source: str,
    infeasibility: list[str] | None,
    family_moves: list[str],
) -> str:
    lines = [f"STATE MIGRATION (against {state_source})"]
    if state is None:
        lines.append("  (no state file available — feasibility check skipped)")
        return "\n".join(lines)

    migrated = state.migrate_to(candidate)
    for uid in sorted(diff.removed_unit_ids):
        u = state.units.get(uid)
        detail = (
            f"on={u.on}, tis={u.time_in_state_steps}"
            if u is not None else "missing in state"
        )
        lines.append(f"  drop:    unit {uid!r}  ({detail})")
    for sid in sorted(diff.removed_storage_ids):
        s = state.storages.get(sid)
        detail = f"soc={s.soc_mwh_th}" if s is not None else "missing in state"
        lines.append(f"  drop:    storage {sid!r}  ({detail})")
    for uid in sorted(diff.added_unit_ids):
        u = migrated.units[uid]
        lines.append(
            f"  init:    unit {uid!r}  (on={u.on}, tis={u.time_in_state_steps})"
        )
    for sid in sorted(diff.added_storage_ids):
        s = migrated.storages[sid]
        lines.append(f"  init:    storage {sid!r}  (soc={s.soc_mwh_th})")

    kept_unit_ids = sorted(set(state.units) & set(candidate.all_unit_ids()))
    kept_storage_ids = sorted(set(state.storages) & {s.id for s in candidate.storages})
    if kept_unit_ids or kept_storage_ids:
        kept = kept_unit_ids + kept_storage_ids
        lines.append(f"  keep:    {', '.join(kept)}")

    if family_moves:
        lines.append(
            f"  family-move check: REJECT — id(s) {family_moves} changed family. "
            "Resubmit as two edits (remove, then add)."
        )
    else:
        lines.append("  family-move check: OK")
    if infeasibility:
        lines.append("  feasibility under proposed: REJECT")
        for p in infeasibility:
            lines.append(f"    - {p}")
    else:
        lines.append("  feasibility under proposed: OK")
    return "\n".join(lines)


def _resolve_current_config(
    explicit: Path | None,
) -> PlantConfig:
    """Load the daemon's current config or fall back to the legacy default.

    The legacy default fallback exists so the CLI works in repos where the
    daemon is running on the hardcoded plant (no CONFIG_FILE set); the
    operator can still diff a proposed config against a known baseline.
    """
    if explicit is None:
        return PlantConfig.legacy_default()
    return PlantConfig.from_json(explicit)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``optimization-config-diff``. Returns the exit code.

    Exit 0  = proposed config is apply-able (no errors, no family moves,
              post-migration state feasible).
    Exit 1  = at least one of: validation error, family-move, infeasibility,
              I/O / parse error on any input file.
    Exit 2  = argparse usage error (handled by argparse itself).
    """
    parser = argparse.ArgumentParser(
        prog="optimization-config-diff",
        description=(
            "Show what would change if the proposed plant_config.json were "
            "applied to the daemon. Never modifies any file."
        ),
    )
    parser.add_argument(
        "--proposed", type=Path, required=True,
        help="Path to the candidate plant_config.json (the file the operator "
             "is considering as the new live config).",
    )
    parser.add_argument(
        "--current", type=Path, default=None,
        help="Path to the daemon's live plant_config.json. Defaults to "
             "PlantConfig.legacy_default() when omitted, so the CLI also "
             "works in legacy_default deployments where the daemon has no "
             "CONFIG_FILE.",
    )
    parser.add_argument(
        "--state", type=Path, default=None,
        help="Path to current.json (DispatchState). Optional; without it "
             "the feasibility check is skipped.",
    )
    args = parser.parse_args(argv)

    # Load proposed payload first — validate even before constructing the
    # current config, so a broken proposed always surfaces its full error
    # list to the operator regardless of unrelated current-side issues.
    try:
        proposed_payload = _read_payload(args.proposed)
    except (OSError, json.JSONDecodeError) as e:
        print(f"error: could not read --proposed {args.proposed}: {e}", file=sys.stderr)
        return 1

    val_result = validate_plant_payload(proposed_payload, CONFIG_SCHEMA_VERSION)

    # If the payload doesn't even validate, we can't construct a PlantConfig
    # for the diff. Print validation only and exit 1.
    if not val_result.ok:
        print(_format_validation(val_result))
        print()
        print("RESULT: cannot apply. Fix validation errors first.")
        return 1

    try:
        candidate = PlantConfig.from_dict(proposed_payload)
    except ConfigValidationError as e:
        # Defensive: validate_plant_payload was ok but post-init failed.
        # Should not happen given consistent rule set, but treat as error.
        print(_format_validation(e.result))
        print("\nRESULT: cannot apply. PlantConfig construction failed.")
        return 1

    try:
        current = _resolve_current_config(args.current)
    except (OSError, ValueError) as e:
        print(f"error: could not read --current {args.current}: {e}", file=sys.stderr)
        return 1

    diff = ConfigDiff.between(current, candidate)
    family_moves = sorted(diff.added_unit_ids & diff.removed_unit_ids)

    state: DispatchState | None = None
    state_source = "(no --state given)"
    infeasibility: list[str] | None = None
    if args.state is not None:
        state_source = str(args.state)
        try:
            state = DispatchState.load(args.state)
        except (OSError, ValueError, KeyError, TypeError) as e:
            print(
                f"warning: could not read --state {args.state}: {e}; "
                "skipping feasibility check.",
                file=sys.stderr,
            )
            state = None
        else:
            migrated = state.migrate_to(candidate)
            problems = migrated.feasible_against(candidate)
            infeasibility = problems if problems else None

    print(_format_diff(current, candidate, diff))
    print()
    print(_format_validation(val_result))
    print()
    print(_format_migration(
        state, diff, candidate, state_source, infeasibility, family_moves,
    ))
    print()

    blocked = bool(family_moves or infeasibility)
    if blocked:
        print(
            "RESULT: cannot apply. Resolve the family-move / feasibility "
            "issue(s) above before committing."
        )
        return 1

    if diff.is_noop:
        print("RESULT: no observable change; applying would be a no-op.")
        return 0

    print(
        f"RESULT: apply-able. Copy {args.proposed} to the daemon's "
        "CONFIG_FILE to commit; the next cycle boundary will pick it up."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via entry point
    sys.exit(main())
