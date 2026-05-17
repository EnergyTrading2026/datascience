"""Structural diff between two PlantConfigs.

Used by the live-reload path (to decide whether a reload is parameter-only,
needs state migration, or is a no-op) and by the dry-run CLI (to surface
exactly what a proposed config would change before the operator commits it).

The diff is keyed by asset id, not by list index — moving an HP from
``heat_pumps[0]`` to ``heat_pumps[1]`` is not a change. Family membership
is part of identity: an id that exists in ``boilers`` of the current config
and in ``chps`` of the candidate is reported as a remove + add, since
per-family state schemas differ.

This module is intentionally read-only: it never constructs a new
``PlantConfig`` and never mutates state. State migration lives on
``DispatchState.migrate_to``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optimization.config import PlantConfig


def _by_id(items) -> dict:
    """Build {id: asset} lookups for both PlantConfig families."""
    return {a.id: a for a in items}


def _unit_family_map(cfg: "PlantConfig") -> dict[str, tuple[str, object]]:
    """{unit_id: (family_label, asset)} across HP+Boiler+CHP.

    Family label distinguishes "the same id" appearing in two different
    families across the diff — that's reported as remove+add, not a param
    change, because the state schemas differ per family.
    """
    out: dict[str, tuple[str, object]] = {}
    for hp in cfg.heat_pumps:
        out[hp.id] = ("heat_pumps", hp)
    for b in cfg.boilers:
        out[b.id] = ("boilers", b)
    for c in cfg.chps:
        out[c.id] = ("chps", c)
    return out


@dataclass(frozen=True)
class ConfigDiff:
    """Set-typed summary of what differs between two PlantConfigs.

    Each ``*_ids`` field is a frozenset of asset ids. ``param_changed_*`` ids
    exist in both configs (same family) but differ in at least one field.
    ``globals_changed`` covers plant-level scalars (dt_h, prices, CO2).

    ``disabled_added`` / ``disabled_removed`` are intentionally independent
    of the asset add/remove sets: a freshly-added asset listed in the
    candidate's ``disabled_asset_ids`` shows up in *both* ``added_*`` and
    ``disabled_added``. The daemon needs both signals (init state entry,
    and pin the MILP to zero).
    """

    added_unit_ids: frozenset[str] = frozenset()
    added_storage_ids: frozenset[str] = frozenset()
    removed_unit_ids: frozenset[str] = frozenset()
    removed_storage_ids: frozenset[str] = frozenset()
    param_changed_unit_ids: frozenset[str] = frozenset()
    param_changed_storage_ids: frozenset[str] = frozenset()
    disabled_added: frozenset[str] = frozenset()
    disabled_removed: frozenset[str] = frozenset()
    globals_changed: bool = False

    @classmethod
    def between(cls, current: "PlantConfig", candidate: "PlantConfig") -> "ConfigDiff":
        """Compute the diff that turns ``current`` into ``candidate``."""
        cur_units = _unit_family_map(current)
        cand_units = _unit_family_map(candidate)
        cur_storages = _by_id(current.storages)
        cand_storages = _by_id(candidate.storages)

        cur_unit_ids = set(cur_units)
        cand_unit_ids = set(cand_units)
        cur_storage_ids = set(cur_storages)
        cand_storage_ids = set(cand_storages)

        added_units = cand_unit_ids - cur_unit_ids
        removed_units = cur_unit_ids - cand_unit_ids
        # An id in both but in a different family is reported as remove+add
        # (state schema differs across families).
        for uid in cur_unit_ids & cand_unit_ids:
            if cur_units[uid][0] != cand_units[uid][0]:
                added_units.add(uid)
                removed_units.add(uid)

        param_changed_units: set[str] = set()
        for uid in (cur_unit_ids & cand_unit_ids) - added_units:
            if cur_units[uid][1] != cand_units[uid][1]:
                param_changed_units.add(uid)

        added_storages = cand_storage_ids - cur_storage_ids
        removed_storages = cur_storage_ids - cand_storage_ids
        param_changed_storages: set[str] = set()
        for sid in cur_storage_ids & cand_storage_ids:
            if cur_storages[sid] != cand_storages[sid]:
                param_changed_storages.add(sid)

        cur_disabled = set(current.disabled_asset_ids)
        cand_disabled = set(candidate.disabled_asset_ids)
        disabled_added = cand_disabled - cur_disabled
        disabled_removed = cur_disabled - cand_disabled

        globals_changed = (
            current.dt_h != candidate.dt_h
            or current.gas_price_eur_mwh_hs != candidate.gas_price_eur_mwh_hs
            or current.co2_factor_t_per_mwh_hs != candidate.co2_factor_t_per_mwh_hs
            or current.co2_price_eur_per_t != candidate.co2_price_eur_per_t
        )

        return cls(
            added_unit_ids=frozenset(added_units),
            added_storage_ids=frozenset(added_storages),
            removed_unit_ids=frozenset(removed_units),
            removed_storage_ids=frozenset(removed_storages),
            param_changed_unit_ids=frozenset(param_changed_units),
            param_changed_storage_ids=frozenset(param_changed_storages),
            disabled_added=frozenset(disabled_added),
            disabled_removed=frozenset(disabled_removed),
            globals_changed=globals_changed,
        )

    @property
    def is_noop(self) -> bool:
        """True iff applying the candidate would change nothing observable."""
        return not (
            self.added_unit_ids
            or self.added_storage_ids
            or self.removed_unit_ids
            or self.removed_storage_ids
            or self.param_changed_unit_ids
            or self.param_changed_storage_ids
            or self.disabled_added
            or self.disabled_removed
            or self.globals_changed
        )

    @property
    def changes_asset_set(self) -> bool:
        """True iff any asset id was added or removed (in either family).

        This is the trigger for state migration and for backup-before-swap.
        """
        return bool(
            self.added_unit_ids
            or self.added_storage_ids
            or self.removed_unit_ids
            or self.removed_storage_ids
        )

    @property
    def is_destructive(self) -> bool:
        """True iff at least one asset is removed.

        Adds are non-destructive (state gains an entry); only removes drop
        state. The reload path backs up ``current.json`` exactly when this
        is true.
        """
        return bool(self.removed_unit_ids or self.removed_storage_ids)


def asset_param_changes(
    current: "PlantConfig", candidate: "PlantConfig", asset_id: str,
) -> dict[str, tuple[object, object]]:
    """Return per-field ``{field: (old, new)}`` for one asset id present in
    both configs. Used by the dry-run CLI for printing.

    Returns an empty dict if the asset is in different families across the
    two configs, or absent from either; callers should already have
    classified it via ``ConfigDiff``.
    """
    cur_units = _unit_family_map(current)
    cand_units = _unit_family_map(candidate)
    if asset_id in cur_units and asset_id in cand_units:
        cur_fam, cur_asset = cur_units[asset_id]
        cand_fam, cand_asset = cand_units[asset_id]
        if cur_fam != cand_fam:
            return {}
        return _field_diff(asdict(cur_asset), asdict(cand_asset))

    cur_storages = _by_id(current.storages)
    cand_storages = _by_id(candidate.storages)
    if asset_id in cur_storages and asset_id in cand_storages:
        return _field_diff(asdict(cur_storages[asset_id]), asdict(cand_storages[asset_id]))
    return {}


def _field_diff(old: dict, new: dict) -> dict[str, tuple[object, object]]:
    out: dict[str, tuple[object, object]] = {}
    for k in old.keys() | new.keys():
        if old.get(k) != new.get(k):
            out[k] = (old.get(k), new.get(k))
    return out
