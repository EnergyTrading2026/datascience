"""Plant configuration: per-asset parameters in lists, validated.

Each asset (heat pump, boiler, CHP, storage) has its own dataclass with a
stable ``id`` field. ``PlantConfig`` is the top-level container with one list
per family — counts may be zero, IDs must be unique across the whole config.

``PlantConfig.legacy_default()`` reproduces the old single-of-each-asset
hardcoded plant from `docs/optimization/optimization_problem.md` exactly. This
is the regression-pin anchor.

``RuntimeConfig`` is solver/operational tuning — change per deployment.

All sanity checks live in ``config_validation`` (the same module Tasks 3-5's
operator API will call). Per-asset ``__post_init__`` delegates to the
per-asset rule function and raises ``ValueError`` on the first error;
``PlantConfig.__post_init__`` runs the full validator for cross-asset and
global checks. ``from_dict`` first runs ``validate_plant_payload`` so callers
get a collect-all ``ConfigValidationError`` (carries the full
``ValidationResult``) before any dataclass is constructed.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from optimization.config_validation import (
    DT_H_REQUIRED,
    ConfigValidationError,
    Issue,
    check_boiler,
    check_chp,
    check_heat_pump,
    check_storage,
    validate_plant_config,
    validate_plant_payload,
)

# Bump on every breaking change to the on-disk plant_config.json schema.
# Distinct from the state schema_version — config and state evolve separately.
CONFIG_SCHEMA_VERSION = 1


def _raise_first_error(issues: list[Issue]) -> None:
    """Raise ``ValueError`` with the first error's message if any are present.

    Per-asset ``__post_init__`` uses this to keep the "fail loud on first
    contradiction" behavior. Operator code that wants the full list of
    errors and warnings goes through ``validate_plant_payload`` directly.
    """
    for issue in issues:
        if issue.severity == "error":
            raise ValueError(issue.message)


@dataclass(frozen=True)
class HeatPumpParams:
    """Single heat pump. No min-up/down (fast-cycling assumed)."""

    id: str
    p_el_min_mw: float
    p_el_max_mw: float
    cop: float

    def __post_init__(self) -> None:
        _raise_first_error(check_heat_pump(
            id=self.id,
            p_el_min_mw=self.p_el_min_mw,
            p_el_max_mw=self.p_el_max_mw,
            cop=self.cop,
            path="",
        ))


@dataclass(frozen=True)
class BoilerParams:
    """Single condensing boiler. Min-up/down in 15-min steps."""

    id: str
    q_min_mw_th: float
    q_max_mw_th: float
    eff: float
    min_up_steps: int
    min_down_steps: int

    def __post_init__(self) -> None:
        _raise_first_error(check_boiler(
            id=self.id,
            q_min_mw_th=self.q_min_mw_th,
            q_max_mw_th=self.q_max_mw_th,
            eff=self.eff,
            min_up_steps=self.min_up_steps,
            min_down_steps=self.min_down_steps,
            path="",
        ))


@dataclass(frozen=True)
class CHPParams:
    """Single CHP unit. Min-up/down in 15-min steps; startup cost in EUR."""

    id: str
    p_el_min_mw: float
    p_el_max_mw: float
    eff_el: float
    eff_th: float
    min_up_steps: int
    min_down_steps: int
    startup_cost_eur: float

    def __post_init__(self) -> None:
        _raise_first_error(check_chp(
            id=self.id,
            p_el_min_mw=self.p_el_min_mw,
            p_el_max_mw=self.p_el_max_mw,
            eff_el=self.eff_el,
            eff_th=self.eff_th,
            min_up_steps=self.min_up_steps,
            min_down_steps=self.min_down_steps,
            startup_cost_eur=self.startup_cost_eur,
            path="",
        ))

    @property
    def heat_power_ratio(self) -> float:
        """Q_th = ratio * P_el for this CHP (= eff_th / eff_el)."""
        return self.eff_th / self.eff_el


@dataclass(frozen=True)
class StorageParams:
    """Single thermal storage. Hard floor enforced every step.

    The loss-vs-charge warning is dt_h-dependent, but a standalone
    ``StorageParams`` has no plant context — so this path checks it against
    ``DT_H_REQUIRED`` (the only legal grid; any other dt_h is rejected at the
    plant level). When ``PlantConfig`` is validated, the warning is
    re-evaluated against the actual ``cfg.dt_h`` so a mismatch can't slip
    through.
    """

    id: str
    capacity_mwh_th: float
    floor_mwh_th: float
    charge_max_mw_th: float
    discharge_max_mw_th: float
    loss_mwh_per_step: float
    soc_init_mwh_th: float

    def __post_init__(self) -> None:
        _raise_first_error(check_storage(
            id=self.id,
            capacity_mwh_th=self.capacity_mwh_th,
            floor_mwh_th=self.floor_mwh_th,
            charge_max_mw_th=self.charge_max_mw_th,
            discharge_max_mw_th=self.discharge_max_mw_th,
            loss_mwh_per_step=self.loss_mwh_per_step,
            soc_init_mwh_th=self.soc_init_mwh_th,
            dt_h=DT_H_REQUIRED,
            path="",
        ))


@dataclass(frozen=True)
class PlantConfig:
    """Whole-plant config: globals + per-asset lists.

    Any list may be empty (e.g. a customer with no CHP). IDs must be unique
    across the entire config so dispatch and state can address assets globally.
    """

    dt_h: float
    gas_price_eur_mwh_hs: float
    co2_factor_t_per_mwh_hs: float
    co2_price_eur_per_t: float

    heat_pumps: tuple[HeatPumpParams, ...]
    boilers: tuple[BoilerParams, ...]
    chps: tuple[CHPParams, ...]
    storages: tuple[StorageParams, ...]

    def __post_init__(self) -> None:
        # Frozen dataclass: coerce list inputs to tuples without mutating self.
        # Must happen before validation since the validator iterates each tuple.
        for fname in ("heat_pumps", "boilers", "chps", "storages"):
            v = getattr(self, fname)
            if not isinstance(v, tuple):
                object.__setattr__(self, fname, tuple(v))
        # Per-asset __post_init__ already validated each item; this re-runs
        # the full validator to catch cross-asset errors (uniqueness,
        # has-producer, globals like dt_h). Per-asset checks are cheap, so we
        # accept the redundancy in exchange for one entrypoint with one rule
        # set. Raises ConfigValidationError (a ValueError subclass) so direct
        # construction surfaces the full ValidationResult, same as from_dict.
        result = validate_plant_config(self)
        if result.errors:
            raise ConfigValidationError(result)

    @property
    def gas_cost_eur_mwh_hs(self) -> float:
        """Effective fuel cost = gas + CO2 (EUR / MWh_Hs)."""
        return self.gas_price_eur_mwh_hs + self.co2_factor_t_per_mwh_hs * self.co2_price_eur_per_t

    def all_unit_ids(self) -> list[str]:
        """IDs of unit-commitment assets (HP+boiler+CHP). Storage has its own state."""
        return (
            [hp.id for hp in self.heat_pumps]
            + [b.id for b in self.boilers]
            + [c.id for c in self.chps]
        )

    @classmethod
    def legacy_default(cls) -> "PlantConfig":
        """The hardcoded 1-of-each-asset plant from the original spec.

        Reproduces the old `PlantParams()` defaults exactly: same IDs, same
        physics, same economics. The regression pin (test_regression_pin.py)
        locks the MILP behavior under this config.
        """
        return cls(
            dt_h=0.25,
            gas_price_eur_mwh_hs=35.0,
            co2_factor_t_per_mwh_hs=0.201,
            co2_price_eur_per_t=60.0,
            heat_pumps=(
                HeatPumpParams(id="hp", p_el_min_mw=1.0, p_el_max_mw=8.0, cop=3.5),
            ),
            boilers=(
                BoilerParams(
                    id="boiler",
                    q_min_mw_th=2.0,
                    q_max_mw_th=12.0,
                    eff=0.97,
                    min_up_steps=4,
                    min_down_steps=4,
                ),
            ),
            chps=(
                CHPParams(
                    id="chp",
                    p_el_min_mw=2.0,
                    p_el_max_mw=6.0,
                    eff_el=0.40,
                    eff_th=0.48,
                    min_up_steps=8,
                    min_down_steps=8,
                    startup_cost_eur=600.0,
                ),
            ),
            storages=(
                StorageParams(
                    id="storage",
                    capacity_mwh_th=200.0,
                    floor_mwh_th=50.0,
                    charge_max_mw_th=15.0,
                    discharge_max_mw_th=15.0,
                    loss_mwh_per_step=0.000125,
                    soc_init_mwh_th=200.0,
                ),
            ),
        )

    def to_dict(self) -> dict:
        """JSON-friendly dict with ``schema_version``. Round-trips via from_dict."""
        return {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "dt_h": self.dt_h,
            "gas_price_eur_mwh_hs": self.gas_price_eur_mwh_hs,
            "co2_factor_t_per_mwh_hs": self.co2_factor_t_per_mwh_hs,
            "co2_price_eur_per_t": self.co2_price_eur_per_t,
            "heat_pumps": [asdict(hp) for hp in self.heat_pumps],
            "boilers": [asdict(b) for b in self.boilers],
            "chps": [asdict(c) for c in self.chps],
            "storages": [asdict(s) for s in self.storages],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PlantConfig":
        """Construct from a dict (e.g. parsed from JSON).

        Runs ``validate_plant_payload`` upfront so this path uses the same
        rule set as the operator API. On failure raises
        ``ConfigValidationError`` (a ``ValueError`` subclass) which carries
        the full ``ValidationResult`` on ``.result`` — callers wanting every
        error and warning catch this; legacy ``except ValueError`` still
        works and sees the first error's message.
        """
        result = validate_plant_payload(payload, expected_schema_version=CONFIG_SCHEMA_VERSION)
        if not result.ok:
            raise ConfigValidationError(result)
        return cls(
            dt_h=float(payload["dt_h"]),
            gas_price_eur_mwh_hs=float(payload["gas_price_eur_mwh_hs"]),
            co2_factor_t_per_mwh_hs=float(payload["co2_factor_t_per_mwh_hs"]),
            co2_price_eur_per_t=float(payload["co2_price_eur_per_t"]),
            heat_pumps=tuple(
                HeatPumpParams(**item) for item in payload.get("heat_pumps", [])
            ),
            boilers=tuple(
                BoilerParams(**item) for item in payload.get("boilers", [])
            ),
            chps=tuple(
                CHPParams(**item) for item in payload.get("chps", [])
            ),
            storages=tuple(
                StorageParams(**item) for item in payload.get("storages", [])
            ),
        )

    def to_json(self, path: Path) -> None:
        """Atomic-write the config to JSON (tmp-then-replace), analogous to
        DispatchState.save."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2))
        os.replace(tmp, path)

    @classmethod
    def from_json(cls, path: Path) -> "PlantConfig":
        """Load config from JSON. FileNotFoundError if missing."""
        return cls.from_dict(json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class RuntimeConfig:
    """Operational tuning. Tweak per deployment without touching plant physics."""

    horizon_hours_target: int = 35     # ideal forward-look (35h matches mpc_prototype)
    horizon_hours_min: int = 11        # covers the 13:00 pre-EPEX-clearing cycle (11h until midnight)
    commit_hours: int = 1              # hourly cadence -> 1h commit, 4 intervals
    solver_time_limit_s: int = 30
    solver_mip_gap: float = 0.005

    # Robust-planning factor: solver sees forecast * this. >=1 inflates demand.
    # Set to 1.0 for honest forecast; the noise eval suggested ~1.10 reduces realized
    # cost slightly under MAPE=10%. Decide per real-data eval.
    demand_safety_factor: float = 1.0
