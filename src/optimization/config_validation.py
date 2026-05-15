"""Centralized validation for plant configuration.

Single source of truth for every sanity check on a PlantConfig — the hard rules
that historically lived in each dataclass's ``__post_init__`` (now delegated
here) plus self-contained warnings (legal but suspect configurations).

Two public entrypoints:

- ``validate_plant_payload(payload)`` takes a raw dict (e.g. parsed from the
  on-disk JSON or submitted via the operator API) and returns a
  ``ValidationResult`` listing *all* errors and warnings without raising.
  This is the contract Tasks 3-5 will call when the operator submits a
  candidate config.

- ``validate_plant_config(cfg)`` takes an already-constructed ``PlantConfig``.
  Every construction path runs the same checks via ``__post_init__``, so on
  a config that successfully loaded, ``errors`` will be empty. This exists
  to surface warnings and to re-validate after structural surgery
  (e.g. ``dataclasses.replace``).

Each ``Issue`` carries a ``severity`` (``"error"`` / ``"warning"``), a stable
``code`` (UI/API key), a ``path`` pointing at the offending field (e.g.
``"boilers[0].q_max_mw_th"``), and a human-readable ``message``.

When the payload-level entrypoint rejects a config, it raises
``ConfigValidationError`` which carries the full ``ValidationResult`` so
callers (Task 5's operator API) can surface every error and warning at once.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from optimization.config import PlantConfig


ERROR = "error"
WARNING = "warning"

# The MILP grid is hardcoded to 15-min steps in model.py (INT_PER_HOUR=4); any
# other dt_h silently breaks cost/SoC scaling. Hard error until model.py is
# generalized.
DT_H_REQUIRED = 0.25

# A storage whose usable range (capacity - floor) is less than this fraction
# of its capacity is almost certainly a typo in floor. 10% is a working
# heuristic — anything tighter and the floor leaves no buffer for the MILP
# to schedule against; revisit once we have real operator feedback.
STORAGE_USABLE_RANGE_WARN_RATIO = 0.10

# Min-up / min-down at or beyond this many steps (24h at 15-min grid) spans
# most of the default 35h MPC horizon (140 steps), so the optimizer can't
# usefully toggle the unit within a single commit cycle.
MIN_UP_DOWN_WARN_STEPS = 96


class Codes:
    """Stable issue codes. UI/operator API will key off these, not message text."""

    # Errors
    BAD_ASSET_ID = "BAD_ASSET_ID"
    NEGATIVE_OR_ZERO = "NEGATIVE_OR_ZERO"
    NEGATIVE = "NEGATIVE"
    MIN_EXCEEDS_MAX = "MIN_EXCEEDS_MAX"
    EFF_OUT_OF_RANGE = "EFF_OUT_OF_RANGE"
    MIN_UP_DOWN_TOO_SMALL = "MIN_UP_DOWN_TOO_SMALL"
    FLOOR_OUT_OF_RANGE = "FLOOR_OUT_OF_RANGE"
    SOC_INIT_OUT_OF_RANGE = "SOC_INIT_OUT_OF_RANGE"
    DT_H_NOT_QUARTERHOUR = "DT_H_NOT_QUARTERHOUR"
    DUPLICATE_ASSET_ID = "DUPLICATE_ASSET_ID"
    ZERO_HEAT_PRODUCERS = "ZERO_HEAT_PRODUCERS"
    PAYLOAD_NOT_DICT = "PAYLOAD_NOT_DICT"
    SCHEMA_VERSION_MISSING = "SCHEMA_VERSION_MISSING"
    SCHEMA_VERSION_MISMATCH = "SCHEMA_VERSION_MISMATCH"
    UNKNOWN_TOP_LEVEL_KEYS = "UNKNOWN_TOP_LEVEL_KEYS"
    ASSET_NOT_A_DICT = "ASSET_NOT_A_DICT"
    ASSET_FIELDS_WRONG = "ASSET_FIELDS_WRONG"
    MISSING_REQUIRED_FIELD = "MISSING_REQUIRED_FIELD"
    NON_NUMERIC = "NON_NUMERIC"

    # Warnings
    CHP_EFF_SUM_OVER_ONE = "CHP_EFF_SUM_OVER_ONE"
    STORAGE_USABLE_RANGE_TINY = "STORAGE_USABLE_RANGE_TINY"
    STORAGE_LOSS_EXCEEDS_CHARGE = "STORAGE_LOSS_EXCEEDS_CHARGE"
    STORAGE_DISCHARGE_DISABLED = "STORAGE_DISCHARGE_DISABLED"
    STORAGE_CHARGE_DISABLED = "STORAGE_CHARGE_DISABLED"
    MIN_UP_OR_DOWN_VERY_LONG = "MIN_UP_OR_DOWN_VERY_LONG"
    NO_STORAGES = "NO_STORAGES"


@dataclass(frozen=True)
class Issue:
    """A single validation finding.

    ``severity``: ``"error"`` (hard contradiction; reject) or ``"warning"``
    (legal but suspect; allow). ``code`` is a short stable symbol.
    ``path`` points at the offending field (e.g. ``"boilers[0].q_max_mw_th"``;
    ``""`` for plant-wide). ``message`` is the human sentence.
    """

    severity: str
    code: str
    path: str
    message: str


class ConfigValidationError(ValueError):
    """Raised when ``PlantConfig.from_dict`` is given a payload that fails
    validation. Subclasses ``ValueError`` so existing ``except ValueError``
    paths keep working, but carries the full ``ValidationResult`` on
    ``.result`` so callers (Task 5's operator API) can surface every error
    and warning at once instead of just the first message.
    """

    def __init__(self, result: "ValidationResult") -> None:
        self.result = result
        first = result.errors[0].message if result.errors else "validation failed"
        super().__init__(first)


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of validating a PlantConfig or its raw-dict payload.

    ``errors`` are hard contradictions that must be fixed before the config
    can be used. ``warnings`` are advisory.

    Use ``.ok`` as the gate: a config with warnings only is still ``ok``.
    """

    errors: tuple[Issue, ...] = ()
    warnings: tuple[Issue, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def all_issues(self) -> tuple[Issue, ...]:
        return self.errors + self.warnings


def _err(code: str, path: str, message: str) -> Issue:
    return Issue(severity=ERROR, code=code, path=path, message=message)


def _warn(code: str, path: str, message: str) -> Issue:
    return Issue(severity=WARNING, code=code, path=path, message=message)


def _result(issues: list[Issue]) -> ValidationResult:
    errs = tuple(i for i in issues if i.severity == ERROR)
    warns = tuple(i for i in issues if i.severity == WARNING)
    return ValidationResult(errors=errs, warnings=warns)


# --------------------------------------------------------------------------- #
# Per-asset rule functions
# --------------------------------------------------------------------------- #
#
# Each takes the asset's typed fields as kwargs and a ``path`` prefix (e.g.
# ``"heat_pumps[0]"``). Returns a list of Issues. Same logic used by the
# dataclass __post_init__ (which raises on any error) and by the payload-level
# validator (which accumulates everything).
#
# Numeric inputs are accepted as ``float`` — the payload-level validator does
# the int/float coercion before calling these.


def check_asset_id(asset_id: Any, path: str) -> list[Issue]:
    """Asset id must be a non-empty string with no surrounding whitespace.

    Whitespace-padded ids would silently fail to match in parquet groupbys
    and in any "drop asset by id" operation downstream.
    """
    if not isinstance(asset_id, str):
        return [_err(Codes.BAD_ASSET_ID, path,
                     f"asset id must be a string, got {asset_id!r}")]
    if not asset_id or asset_id.strip() != asset_id:
        return [_err(
            Codes.BAD_ASSET_ID, path,
            f"asset id must be a non-empty string with no leading/trailing "
            f"whitespace, got {asset_id!r}",
        )]
    return []


def check_heat_pump(
    *, id: Any, p_el_min_mw: float, p_el_max_mw: float, cop: float, path: str
) -> list[Issue]:
    issues: list[Issue] = []
    issues.extend(check_asset_id(id, f"{path}.id"))
    if not (p_el_max_mw > 0):
        issues.append(_err(
            Codes.NEGATIVE_OR_ZERO, f"{path}.p_el_max_mw",
            f"p_el_max_mw={p_el_max_mw} for asset {id!r} must be > 0",
        ))
    if p_el_min_mw < 0:
        issues.append(_err(
            Codes.NEGATIVE, f"{path}.p_el_min_mw",
            f"heat pump {id!r}: p_el_min_mw={p_el_min_mw} must be >= 0",
        ))
    elif p_el_max_mw > 0 and p_el_min_mw > p_el_max_mw:
        issues.append(_err(
            Codes.MIN_EXCEEDS_MAX, f"{path}.p_el_min_mw",
            f"heat pump {id!r}: p_el_min_mw={p_el_min_mw} must be <= "
            f"p_el_max_mw={p_el_max_mw}",
        ))
    if not (cop > 0):
        issues.append(_err(
            Codes.NEGATIVE_OR_ZERO, f"{path}.cop",
            f"cop={cop} for asset {id!r} must be > 0",
        ))
    return issues


def check_boiler(
    *,
    id: Any,
    q_min_mw_th: float,
    q_max_mw_th: float,
    eff: float,
    min_up_steps: int,
    min_down_steps: int,
    path: str,
) -> list[Issue]:
    issues: list[Issue] = []
    issues.extend(check_asset_id(id, f"{path}.id"))
    if not (q_max_mw_th > 0):
        issues.append(_err(
            Codes.NEGATIVE_OR_ZERO, f"{path}.q_max_mw_th",
            f"q_max_mw_th={q_max_mw_th} for asset {id!r} must be > 0",
        ))
    if q_min_mw_th < 0:
        issues.append(_err(
            Codes.NEGATIVE, f"{path}.q_min_mw_th",
            f"boiler {id!r}: q_min_mw_th={q_min_mw_th} must be >= 0",
        ))
    elif q_max_mw_th > 0 and q_min_mw_th > q_max_mw_th:
        issues.append(_err(
            Codes.MIN_EXCEEDS_MAX, f"{path}.q_min_mw_th",
            f"boiler {id!r}: q_min_mw_th={q_min_mw_th} must be <= "
            f"q_max_mw_th={q_max_mw_th}",
        ))
    if not (0 < eff <= 1.5):
        issues.append(_err(
            Codes.EFF_OUT_OF_RANGE, f"{path}.eff",
            f"boiler {id!r}: eff={eff} out of (0, 1.5]",
        ))
    if min_up_steps < 1 or min_down_steps < 1:
        issues.append(_err(
            Codes.MIN_UP_DOWN_TOO_SMALL, f"{path}.min_up_steps",
            f"boiler {id!r}: min_up_steps/min_down_steps must be >= 1",
        ))
    issues.extend(_warn_long_min_up_down(id, min_up_steps, min_down_steps, path))
    return issues


def check_chp(
    *,
    id: Any,
    p_el_min_mw: float,
    p_el_max_mw: float,
    eff_el: float,
    eff_th: float,
    min_up_steps: int,
    min_down_steps: int,
    startup_cost_eur: float,
    path: str,
) -> list[Issue]:
    issues: list[Issue] = []
    issues.extend(check_asset_id(id, f"{path}.id"))
    if not (p_el_max_mw > 0):
        issues.append(_err(
            Codes.NEGATIVE_OR_ZERO, f"{path}.p_el_max_mw",
            f"p_el_max_mw={p_el_max_mw} for asset {id!r} must be > 0",
        ))
    if p_el_min_mw < 0:
        issues.append(_err(
            Codes.NEGATIVE, f"{path}.p_el_min_mw",
            f"CHP {id!r}: p_el_min_mw={p_el_min_mw} must be >= 0",
        ))
    elif p_el_max_mw > 0 and p_el_min_mw > p_el_max_mw:
        issues.append(_err(
            Codes.MIN_EXCEEDS_MAX, f"{path}.p_el_min_mw",
            f"CHP {id!r}: p_el_min_mw={p_el_min_mw} must be <= "
            f"p_el_max_mw={p_el_max_mw}",
        ))
    if not (0 < eff_el <= 1):
        issues.append(_err(
            Codes.EFF_OUT_OF_RANGE, f"{path}.eff_el",
            f"CHP {id!r}: eff_el={eff_el} out of (0, 1]",
        ))
    if not (0 < eff_th <= 1):
        issues.append(_err(
            Codes.EFF_OUT_OF_RANGE, f"{path}.eff_th",
            f"CHP {id!r}: eff_th={eff_th} out of (0, 1]",
        ))
    if min_up_steps < 1 or min_down_steps < 1:
        issues.append(_err(
            Codes.MIN_UP_DOWN_TOO_SMALL, f"{path}.min_up_steps",
            f"CHP {id!r}: min_up_steps/min_down_steps must be >= 1",
        ))
    if startup_cost_eur < 0:
        issues.append(_err(
            Codes.NEGATIVE, f"{path}.startup_cost_eur",
            f"CHP {id!r}: startup_cost_eur must be >= 0",
        ))
    issues.extend(_warn_long_min_up_down(id, min_up_steps, min_down_steps, path))
    # Warning: eff_el + eff_th > 1 is thermodynamically suspect on LHV. Each
    # is individually checked <= 1, so e.g. 0.6 + 0.6 = 1.2 sneaks through the
    # hard checks.
    if 0 < eff_el <= 1 and 0 < eff_th <= 1 and (eff_el + eff_th) > 1.0:
        issues.append(_warn(
            Codes.CHP_EFF_SUM_OVER_ONE, f"{path}.eff_th",
            f"CHP {id!r}: eff_el + eff_th = {eff_el + eff_th:.3f} > 1.0 "
            f"(thermodynamically suspect on LHV).",
        ))
    return issues


def check_storage(
    *,
    id: Any,
    capacity_mwh_th: float,
    floor_mwh_th: float,
    charge_max_mw_th: float,
    discharge_max_mw_th: float,
    loss_mwh_per_step: float,
    soc_init_mwh_th: float,
    dt_h: float,
    path: str,
) -> list[Issue]:
    """Storage rules. ``dt_h`` is needed for the loss-vs-charge warning.

    All checks are independent: a bad ``capacity_mwh_th`` does not suppress
    the negativity checks on ``charge_max_mw_th`` / ``discharge_max_mw_th`` /
    ``loss_mwh_per_step``. Only the *range* checks (floor, soc_init) are
    skipped when capacity is non-positive, since "in [0, capacity]" is
    meaningless then.
    """
    issues: list[Issue] = []
    issues.extend(check_asset_id(id, f"{path}.id"))
    capacity_ok = capacity_mwh_th > 0
    if not capacity_ok:
        issues.append(_err(
            Codes.NEGATIVE_OR_ZERO, f"{path}.capacity_mwh_th",
            f"capacity_mwh_th={capacity_mwh_th} for asset {id!r} must be > 0",
        ))
    if charge_max_mw_th < 0 or discharge_max_mw_th < 0:
        issues.append(_err(
            Codes.NEGATIVE, f"{path}.charge_max_mw_th",
            f"storage {id!r}: charge/discharge limits must be >= 0",
        ))
    if loss_mwh_per_step < 0:
        issues.append(_err(
            Codes.NEGATIVE, f"{path}.loss_mwh_per_step",
            f"storage {id!r}: loss_mwh_per_step must be >= 0",
        ))
    if capacity_ok and not (0 <= floor_mwh_th <= capacity_mwh_th):
        issues.append(_err(
            Codes.FLOOR_OUT_OF_RANGE, f"{path}.floor_mwh_th",
            f"storage {id!r}: floor_mwh_th={floor_mwh_th} must be in "
            f"[0, capacity_mwh_th={capacity_mwh_th}]",
        ))
    if capacity_ok and not (floor_mwh_th <= soc_init_mwh_th <= capacity_mwh_th):
        issues.append(_err(
            Codes.SOC_INIT_OUT_OF_RANGE, f"{path}.soc_init_mwh_th",
            f"storage {id!r}: soc_init_mwh_th={soc_init_mwh_th} must be "
            f"in [floor={floor_mwh_th}, capacity={capacity_mwh_th}]",
        ))

    if not capacity_ok:
        # Without capacity the warning thresholds aren't meaningful; skip
        # warnings but the errors above are all collected.
        return issues

    # Warnings (only when the hard rules above passed for the relevant field).
    if 0 <= floor_mwh_th <= capacity_mwh_th:
        usable = capacity_mwh_th - floor_mwh_th
        if usable < STORAGE_USABLE_RANGE_WARN_RATIO * capacity_mwh_th:
            issues.append(_warn(
                Codes.STORAGE_USABLE_RANGE_TINY, f"{path}.floor_mwh_th",
                f"storage {id!r}: usable range capacity-floor={usable:.3f} "
                f"is < {STORAGE_USABLE_RANGE_WARN_RATIO * 100:.0f}% of "
                f"capacity={capacity_mwh_th}; likely a typo in floor.",
            ))
    if charge_max_mw_th >= 0 and loss_mwh_per_step >= 0 and dt_h > 0:
        # Energy you can put in per step (MWh) vs energy lost per step (MWh).
        max_charge_per_step = charge_max_mw_th * dt_h
        if loss_mwh_per_step > max_charge_per_step:
            issues.append(_warn(
                Codes.STORAGE_LOSS_EXCEEDS_CHARGE, f"{path}.loss_mwh_per_step",
                f"storage {id!r}: loss_mwh_per_step={loss_mwh_per_step} "
                f"exceeds max charge per step "
                f"(charge_max_mw_th*dt_h={max_charge_per_step}); "
                f"storage drains faster than it can fill.",
            ))
    if discharge_max_mw_th == 0:
        issues.append(_warn(
            Codes.STORAGE_DISCHARGE_DISABLED, f"{path}.discharge_max_mw_th",
            f"storage {id!r}: discharge_max_mw_th=0; storage can be charged "
            f"but never discharged.",
        ))
    if charge_max_mw_th == 0:
        issues.append(_warn(
            Codes.STORAGE_CHARGE_DISABLED, f"{path}.charge_max_mw_th",
            f"storage {id!r}: charge_max_mw_th=0; storage can be discharged "
            f"but never refilled.",
        ))
    return issues


def _warn_long_min_up_down(
    asset_id: Any, min_up_steps: int, min_down_steps: int, path: str
) -> list[Issue]:
    if max(min_up_steps, min_down_steps) > MIN_UP_DOWN_WARN_STEPS:
        return [_warn(
            Codes.MIN_UP_OR_DOWN_VERY_LONG, f"{path}.min_up_steps",
            f"asset {asset_id!r}: min_up_steps={min_up_steps}, "
            f"min_down_steps={min_down_steps}; values > "
            f"{MIN_UP_DOWN_WARN_STEPS} steps (24h at 15-min grid) span most "
            f"of the default 35h MPC horizon — the optimizer can't usefully "
            f"toggle the unit within a commit cycle.",
        )]
    return []


# --------------------------------------------------------------------------- #
# Plant-level (cross-asset) rules
# --------------------------------------------------------------------------- #


def check_plant_globals(
    *,
    dt_h: float,
    gas_price_eur_mwh_hs: float,
    co2_factor_t_per_mwh_hs: float,
    co2_price_eur_per_t: float,
) -> list[Issue]:
    """Plant-wide scalars: dt_h must be 15-min, prices/CO2 non-negative."""
    issues: list[Issue] = []
    if dt_h != DT_H_REQUIRED:
        issues.append(_err(
            Codes.DT_H_NOT_QUARTERHOUR, "dt_h",
            f"dt_h must be {DT_H_REQUIRED} (the MILP grid is hardcoded to "
            f"15-min steps in model.py); got {dt_h}.",
        ))
    if gas_price_eur_mwh_hs < 0:
        issues.append(_err(
            Codes.NEGATIVE, "gas_price_eur_mwh_hs",
            "gas_price_eur_mwh_hs must be >= 0",
        ))
    if co2_factor_t_per_mwh_hs < 0 or co2_price_eur_per_t < 0:
        issues.append(_err(
            Codes.NEGATIVE, "co2_factor_t_per_mwh_hs",
            "co2 factor/price must be >= 0",
        ))
    return issues


def check_id_uniqueness(
    heat_pump_ids: list[str],
    boiler_ids: list[str],
    chp_ids: list[str],
    storage_ids: list[str],
) -> list[Issue]:
    """Asset ids must be globally unique across all four families."""
    issues: list[Issue] = []
    seen: dict[str, str] = {}
    families = (
        ("heat_pumps", heat_pump_ids),
        ("boilers", boiler_ids),
        ("chps", chp_ids),
        ("storages", storage_ids),
    )
    for family, ids in families:
        for idx, asset_id in enumerate(ids):
            if not isinstance(asset_id, str):
                continue  # id-syntax errors reported separately
            if asset_id in seen:
                issues.append(_err(
                    Codes.DUPLICATE_ASSET_ID, f"{family}[{idx}].id",
                    f"duplicate asset id {asset_id!r} (in {family} and {seen[asset_id]})",
                ))
            else:
                seen[asset_id] = family
    return issues


def check_has_heat_producer(
    n_heat_pumps: int, n_boilers: int, n_chps: int
) -> list[Issue]:
    if n_heat_pumps + n_boilers + n_chps == 0:
        return [_err(
            Codes.ZERO_HEAT_PRODUCERS, "",
            "PlantConfig has zero heat producers (HP+Boiler+CHP all empty); "
            "demand cannot be met",
        )]
    return []


def warn_no_storages(n_storages: int) -> list[Issue]:
    if n_storages == 0:
        return [_warn(
            Codes.NO_STORAGES, "storages",
            "PlantConfig has zero storages; the MPC has no inter-cycle "
            "thermal flexibility.",
        )]
    return []


# --------------------------------------------------------------------------- #
# Public entrypoints
# --------------------------------------------------------------------------- #


def validate_plant_config(cfg: "PlantConfig") -> ValidationResult:
    """Validate an already-constructed PlantConfig.

    Every construction path runs the same checks via ``__post_init__``, so
    on a config that was returned successfully ``errors`` will be empty.
    Returned for symmetry with ``validate_plant_payload`` and so callers
    can re-validate after structural surgery (e.g. ``dataclasses.replace``).
    """
    issues: list[Issue] = []

    issues.extend(check_plant_globals(
        dt_h=cfg.dt_h,
        gas_price_eur_mwh_hs=cfg.gas_price_eur_mwh_hs,
        co2_factor_t_per_mwh_hs=cfg.co2_factor_t_per_mwh_hs,
        co2_price_eur_per_t=cfg.co2_price_eur_per_t,
    ))

    for i, hp in enumerate(cfg.heat_pumps):
        issues.extend(check_heat_pump(
            id=hp.id, p_el_min_mw=hp.p_el_min_mw,
            p_el_max_mw=hp.p_el_max_mw, cop=hp.cop,
            path=f"heat_pumps[{i}]",
        ))
    for i, b in enumerate(cfg.boilers):
        issues.extend(check_boiler(
            id=b.id, q_min_mw_th=b.q_min_mw_th, q_max_mw_th=b.q_max_mw_th,
            eff=b.eff, min_up_steps=b.min_up_steps,
            min_down_steps=b.min_down_steps,
            path=f"boilers[{i}]",
        ))
    for i, c in enumerate(cfg.chps):
        issues.extend(check_chp(
            id=c.id, p_el_min_mw=c.p_el_min_mw, p_el_max_mw=c.p_el_max_mw,
            eff_el=c.eff_el, eff_th=c.eff_th,
            min_up_steps=c.min_up_steps, min_down_steps=c.min_down_steps,
            startup_cost_eur=c.startup_cost_eur,
            path=f"chps[{i}]",
        ))
    for i, s in enumerate(cfg.storages):
        issues.extend(check_storage(
            id=s.id, capacity_mwh_th=s.capacity_mwh_th,
            floor_mwh_th=s.floor_mwh_th,
            charge_max_mw_th=s.charge_max_mw_th,
            discharge_max_mw_th=s.discharge_max_mw_th,
            loss_mwh_per_step=s.loss_mwh_per_step,
            soc_init_mwh_th=s.soc_init_mwh_th,
            dt_h=cfg.dt_h,
            path=f"storages[{i}]",
        ))

    issues.extend(check_id_uniqueness(
        [hp.id for hp in cfg.heat_pumps],
        [b.id for b in cfg.boilers],
        [c.id for c in cfg.chps],
        [s.id for s in cfg.storages],
    ))
    issues.extend(check_has_heat_producer(
        len(cfg.heat_pumps), len(cfg.boilers), len(cfg.chps),
    ))
    issues.extend(warn_no_storages(len(cfg.storages)))

    return _result(issues)


# Top-level and per-family field sets. Single source of truth: ``from_dict``
# in config.py asks this module for validation, so the schema only lives here.
_TOP_LEVEL_FIELDS: frozenset[str] = frozenset({
    "schema_version", "dt_h", "gas_price_eur_mwh_hs",
    "co2_factor_t_per_mwh_hs", "co2_price_eur_per_t",
    "heat_pumps", "boilers", "chps", "storages",
})
_HP_FIELDS: frozenset[str] = frozenset({"id", "p_el_min_mw", "p_el_max_mw", "cop"})
_BOILER_FIELDS: frozenset[str] = frozenset({
    "id", "q_min_mw_th", "q_max_mw_th", "eff", "min_up_steps", "min_down_steps",
})
_CHP_FIELDS: frozenset[str] = frozenset({
    "id", "p_el_min_mw", "p_el_max_mw", "eff_el", "eff_th",
    "min_up_steps", "min_down_steps", "startup_cost_eur",
})
_STORAGE_FIELDS: frozenset[str] = frozenset({
    "id", "capacity_mwh_th", "floor_mwh_th", "charge_max_mw_th",
    "discharge_max_mw_th", "loss_mwh_per_step", "soc_init_mwh_th",
})


def _coerce_number(
    payload: dict, key: str, issues: list[Issue], *, path: str | None = None,
) -> float | None:
    """Return ``payload[key]`` as a float, recording NON_NUMERIC if it isn't.

    Strict: rejects bool (``True``/``False`` are int subclasses but never
    config numbers) and stringy numbers like ``"3.14"`` (JSON should already
    be typed; a string here is a typo signal we don't want to swallow).
    Returns None if the key is missing or rejected; callers skip value-range
    checks when None.

    ``path`` overrides the issue path (used for asset-level fields like
    ``"boilers[0].q_max_mw_th"``); defaults to ``key`` for top-level scalars.
    """
    if key not in payload:
        return None
    v = payload[key]
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        issues.append(_err(
            Codes.NON_NUMERIC, path if path is not None else key,
            f"{key} must be a number (int or float), got "
            f"{type(v).__name__}={v!r}",
        ))
        return None
    return float(v)


def _coerce_int(
    payload: dict, key: str, issues: list[Issue], *, path: str,
) -> int | None:
    """Return ``payload[key]`` as an int, recording NON_NUMERIC otherwise.

    Strict: rejects bool and floats (``4.0`` is fine in JSON for a step
    count, but accepting it would mask ``4.5``-style typos). Asset path
    is required because every caller is asset-level.
    """
    if key not in payload:
        return None
    v = payload[key]
    if isinstance(v, bool) or not isinstance(v, int):
        issues.append(_err(
            Codes.NON_NUMERIC, path,
            f"{key} must be an int, got {type(v).__name__}={v!r}",
        ))
        return None
    return v


def _check_hp_dict(item: dict, path: str, dt_h: float | None) -> list[Issue]:
    issues: list[Issue] = []
    p_min = _coerce_number(item, "p_el_min_mw", issues, path=f"{path}.p_el_min_mw")
    p_max = _coerce_number(item, "p_el_max_mw", issues, path=f"{path}.p_el_max_mw")
    cop = _coerce_number(item, "cop", issues, path=f"{path}.cop")
    if None not in (p_min, p_max, cop):
        issues.extend(check_heat_pump(
            id=item["id"], p_el_min_mw=p_min, p_el_max_mw=p_max,
            cop=cop, path=path,
        ))
    else:
        issues.extend(check_asset_id(item.get("id"), f"{path}.id"))
    return issues


def _check_boiler_dict(item: dict, path: str, dt_h: float | None) -> list[Issue]:
    issues: list[Issue] = []
    q_min = _coerce_number(item, "q_min_mw_th", issues, path=f"{path}.q_min_mw_th")
    q_max = _coerce_number(item, "q_max_mw_th", issues, path=f"{path}.q_max_mw_th")
    eff = _coerce_number(item, "eff", issues, path=f"{path}.eff")
    mu = _coerce_int(item, "min_up_steps", issues, path=f"{path}.min_up_steps")
    md = _coerce_int(item, "min_down_steps", issues, path=f"{path}.min_down_steps")
    if None not in (q_min, q_max, eff, mu, md):
        issues.extend(check_boiler(
            id=item["id"], q_min_mw_th=q_min, q_max_mw_th=q_max, eff=eff,
            min_up_steps=mu, min_down_steps=md, path=path,
        ))
    else:
        issues.extend(check_asset_id(item.get("id"), f"{path}.id"))
    return issues


def _check_chp_dict(item: dict, path: str, dt_h: float | None) -> list[Issue]:
    issues: list[Issue] = []
    p_min = _coerce_number(item, "p_el_min_mw", issues, path=f"{path}.p_el_min_mw")
    p_max = _coerce_number(item, "p_el_max_mw", issues, path=f"{path}.p_el_max_mw")
    eff_el = _coerce_number(item, "eff_el", issues, path=f"{path}.eff_el")
    eff_th = _coerce_number(item, "eff_th", issues, path=f"{path}.eff_th")
    mu = _coerce_int(item, "min_up_steps", issues, path=f"{path}.min_up_steps")
    md = _coerce_int(item, "min_down_steps", issues, path=f"{path}.min_down_steps")
    su = _coerce_number(item, "startup_cost_eur", issues,
                        path=f"{path}.startup_cost_eur")
    if None not in (p_min, p_max, eff_el, eff_th, mu, md, su):
        issues.extend(check_chp(
            id=item["id"], p_el_min_mw=p_min, p_el_max_mw=p_max,
            eff_el=eff_el, eff_th=eff_th, min_up_steps=mu,
            min_down_steps=md, startup_cost_eur=su, path=path,
        ))
    else:
        issues.extend(check_asset_id(item.get("id"), f"{path}.id"))
    return issues


def _check_storage_dict(item: dict, path: str, dt_h: float | None) -> list[Issue]:
    issues: list[Issue] = []
    cap = _coerce_number(item, "capacity_mwh_th", issues, path=f"{path}.capacity_mwh_th")
    floor = _coerce_number(item, "floor_mwh_th", issues, path=f"{path}.floor_mwh_th")
    ch = _coerce_number(item, "charge_max_mw_th", issues, path=f"{path}.charge_max_mw_th")
    dis = _coerce_number(item, "discharge_max_mw_th", issues, path=f"{path}.discharge_max_mw_th")
    loss = _coerce_number(item, "loss_mwh_per_step", issues, path=f"{path}.loss_mwh_per_step")
    soc = _coerce_number(item, "soc_init_mwh_th", issues, path=f"{path}.soc_init_mwh_th")
    # If dt_h failed to parse at plant level, fall back to the required value
    # so the storage check still runs — the loss-vs-charge warning will use
    # the only legal grid (the value the user has to pick anyway).
    effective_dt_h = dt_h if dt_h is not None else DT_H_REQUIRED
    if None not in (cap, floor, ch, dis, loss, soc):
        issues.extend(check_storage(
            id=item["id"], capacity_mwh_th=cap, floor_mwh_th=floor,
            charge_max_mw_th=ch, discharge_max_mw_th=dis,
            loss_mwh_per_step=loss, soc_init_mwh_th=soc, dt_h=effective_dt_h,
            path=path,
        ))
    else:
        issues.extend(check_asset_id(item.get("id"), f"{path}.id"))
    return issues


@dataclass(frozen=True)
class _Family:
    """Per-family wiring for the payload-level loop. ``check`` does the
    type-coerce-then-rule-check step for one asset dict; type errors emerge
    as NON_NUMERIC and the rule call is skipped for that asset."""

    key: str               # top-level key in payload (e.g. "heat_pumps")
    fields: frozenset[str]
    label: str             # human label used in "<label> entry" messages
    check: Callable[[dict, str, float | None], list[Issue]]


_FAMILIES: tuple[_Family, ...] = (
    _Family("heat_pumps", _HP_FIELDS, "heat_pump", _check_hp_dict),
    _Family("boilers", _BOILER_FIELDS, "boiler", _check_boiler_dict),
    _Family("chps", _CHP_FIELDS, "chp", _check_chp_dict),
    _Family("storages", _STORAGE_FIELDS, "storage", _check_storage_dict),
)


def validate_plant_payload(
    payload: Any, expected_schema_version: int
) -> ValidationResult:
    """Validate a raw dict (e.g. ``json.loads(...)``). Collects all issues.

    Does *not* construct dataclasses, so it survives a payload where any
    number of fields are wrong simultaneously — caller gets the full list
    in one shot. This is the function the operator API (Task 5) calls when
    the operator submits a candidate config and expects a full diff of
    errors and warnings.
    """
    issues: list[Issue] = []

    if not isinstance(payload, dict):
        return _result([_err(
            Codes.PAYLOAD_NOT_DICT, "",
            f"plant config must be a JSON object, got {type(payload).__name__}",
        )])

    # Schema version. ``bool`` is rejected explicitly because in Python
    # ``True == 1`` / ``False == 0`` would let ``true`` masquerade as
    # schema_version 1.
    version = payload.get("schema_version")
    if version is None:
        issues.append(_err(
            Codes.SCHEMA_VERSION_MISSING, "schema_version",
            "plant config payload is missing schema_version. "
            f"Expected {expected_schema_version}.",
        ))
    elif isinstance(version, bool) or not isinstance(version, int) or version != expected_schema_version:
        issues.append(_err(
            Codes.SCHEMA_VERSION_MISMATCH, "schema_version",
            f"plant config schema_version={version!r}, expected "
            f"int {expected_schema_version}. No migration available.",
        ))

    # Unknown top-level keys
    extra = set(payload) - _TOP_LEVEL_FIELDS
    if extra:
        issues.append(_err(
            Codes.UNKNOWN_TOP_LEVEL_KEYS, "",
            f"plant config has unknown top-level keys: {sorted(extra)}. "
            f"Expected: {sorted(_TOP_LEVEL_FIELDS)}",
        ))

    # Required scalar fields (the four asset lists default to []).
    for required in (
        "dt_h", "gas_price_eur_mwh_hs",
        "co2_factor_t_per_mwh_hs", "co2_price_eur_per_t",
    ):
        if required not in payload:
            issues.append(_err(
                Codes.MISSING_REQUIRED_FIELD, required,
                f"plant config missing required field: {required!r}",
            ))

    # Plant globals (only check if present-and-numeric; otherwise NON_NUMERIC).
    dt_h = _coerce_number(payload, "dt_h", issues)
    gas_price = _coerce_number(payload, "gas_price_eur_mwh_hs", issues)
    co2_factor = _coerce_number(payload, "co2_factor_t_per_mwh_hs", issues)
    co2_price = _coerce_number(payload, "co2_price_eur_per_t", issues)
    if None not in (dt_h, gas_price, co2_factor, co2_price):
        issues.extend(check_plant_globals(
            dt_h=dt_h, gas_price_eur_mwh_hs=gas_price,
            co2_factor_t_per_mwh_hs=co2_factor,
            co2_price_eur_per_t=co2_price,
        ))

    # Per-family assets.
    ids_by_family: dict[str, list[str]] = {fam.key: [] for fam in _FAMILIES}
    # Count valid (list-shaped) family entries separately so the
    # has-heat-producer check and no-storages warning are driven by the same
    # data the loop just inspected — avoids a second pass that has to re-do
    # the "is this a list?" guard.
    count_by_family: dict[str, int] = {fam.key: 0 for fam in _FAMILIES}
    for fam in _FAMILIES:
        raw_list = payload.get(fam.key, [])
        if not isinstance(raw_list, list):
            issues.append(_err(
                Codes.ASSET_FIELDS_WRONG, fam.key,
                f"{fam.key} must be a list, got {type(raw_list).__name__}",
            ))
            continue
        count_by_family[fam.key] = len(raw_list)
        for idx, item in enumerate(raw_list):
            path = f"{fam.key}[{idx}]"
            if not isinstance(item, dict):
                issues.append(_err(
                    Codes.ASSET_NOT_A_DICT, path,
                    f"{fam.label} entry must be a dict, got "
                    f"{type(item).__name__}",
                ))
                continue
            extra_keys = set(item) - fam.fields
            missing_keys = fam.fields - set(item)
            if extra_keys or missing_keys:
                issues.append(_err(
                    Codes.ASSET_FIELDS_WRONG, path,
                    f"{fam.label} entry id={item.get('id', '?')!r} has wrong "
                    f"fields. missing={sorted(missing_keys)}, "
                    f"extra={sorted(extra_keys)}",
                ))
                # Don't run value checks if the shape is wrong: we'd just
                # cascade KeyError-equivalents into the user's report.
                # Still track the id for uniqueness if it parses.
                aid = item.get("id")
                if isinstance(aid, str):
                    ids_by_family[fam.key].append(aid)
                continue

            # Shape is right: run value checks on this asset.
            issues.extend(fam.check(item, path, dt_h))
            aid = item.get("id")
            if isinstance(aid, str):
                ids_by_family[fam.key].append(aid)

    # Cross-asset checks.
    issues.extend(check_id_uniqueness(
        ids_by_family["heat_pumps"], ids_by_family["boilers"],
        ids_by_family["chps"], ids_by_family["storages"],
    ))
    issues.extend(check_has_heat_producer(
        count_by_family["heat_pumps"],
        count_by_family["boilers"],
        count_by_family["chps"],
    ))
    issues.extend(warn_no_storages(count_by_family["storages"]))

    return _result(issues)
