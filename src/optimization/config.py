"""Plant configuration: per-asset parameters in lists, validated.

Each asset (heat pump, boiler, CHP, storage) has its own dataclass with a
stable ``id`` field. ``PlantConfig`` is the top-level container with one list
per family — counts may be zero, IDs must be unique across the whole config.

``PlantConfig.legacy_default()`` reproduces the old single-of-each-asset
hardcoded plant from `docs/optimization/optimization_problem.md` exactly. This
is the regression-pin anchor.

``RuntimeConfig`` is solver/operational tuning — change per deployment.

``__post_init__`` and ``from_dict`` raise ``ValueError`` on any problem.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Bump on every breaking change to the on-disk plant_config.json schema.
# Distinct from the state schema_version — config and state evolve separately.
CONFIG_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class HeatPumpParams:
    """Single heat pump. No min-up/down (fast-cycling assumed)."""

    id: str
    p_el_min_mw: float
    p_el_max_mw: float
    cop: float

    def __post_init__(self) -> None:
        _check_id(self.id)
        _check_pos(self.p_el_max_mw, "p_el_max_mw", self.id)
        if self.p_el_min_mw < 0 or self.p_el_min_mw > self.p_el_max_mw:
            raise ValueError(
                f"heat pump {self.id!r}: p_el_min_mw={self.p_el_min_mw} must be in "
                f"[0, p_el_max_mw={self.p_el_max_mw}]"
            )
        _check_pos(self.cop, "cop", self.id)


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
        _check_id(self.id)
        _check_pos(self.q_max_mw_th, "q_max_mw_th", self.id)
        if self.q_min_mw_th < 0 or self.q_min_mw_th > self.q_max_mw_th:
            raise ValueError(
                f"boiler {self.id!r}: q_min_mw_th={self.q_min_mw_th} must be in "
                f"[0, q_max_mw_th={self.q_max_mw_th}]"
            )
        if not 0 < self.eff <= 1.5:
            raise ValueError(f"boiler {self.id!r}: eff={self.eff} out of (0, 1.5]")
        if self.min_up_steps < 1 or self.min_down_steps < 1:
            raise ValueError(
                f"boiler {self.id!r}: min_up_steps/min_down_steps must be >= 1"
            )


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
        _check_id(self.id)
        _check_pos(self.p_el_max_mw, "p_el_max_mw", self.id)
        if self.p_el_min_mw < 0 or self.p_el_min_mw > self.p_el_max_mw:
            raise ValueError(
                f"CHP {self.id!r}: p_el_min_mw={self.p_el_min_mw} must be in "
                f"[0, p_el_max_mw={self.p_el_max_mw}]"
            )
        if not 0 < self.eff_el <= 1:
            raise ValueError(f"CHP {self.id!r}: eff_el={self.eff_el} out of (0, 1]")
        if not 0 < self.eff_th <= 1:
            raise ValueError(f"CHP {self.id!r}: eff_th={self.eff_th} out of (0, 1]")
        if self.min_up_steps < 1 or self.min_down_steps < 1:
            raise ValueError(f"CHP {self.id!r}: min_up_steps/min_down_steps must be >= 1")
        if self.startup_cost_eur < 0:
            raise ValueError(f"CHP {self.id!r}: startup_cost_eur must be >= 0")

    @property
    def heat_power_ratio(self) -> float:
        """Q_th = ratio * P_el for this CHP (= eff_th / eff_el)."""
        return self.eff_th / self.eff_el


@dataclass(frozen=True)
class StorageParams:
    """Single thermal storage. Hard floor enforced every step."""

    id: str
    capacity_mwh_th: float
    floor_mwh_th: float
    charge_max_mw_th: float
    discharge_max_mw_th: float
    loss_mwh_per_step: float
    soc_init_mwh_th: float

    def __post_init__(self) -> None:
        _check_id(self.id)
        _check_pos(self.capacity_mwh_th, "capacity_mwh_th", self.id)
        if not 0 <= self.floor_mwh_th <= self.capacity_mwh_th:
            raise ValueError(
                f"storage {self.id!r}: floor_mwh_th={self.floor_mwh_th} must be in "
                f"[0, capacity_mwh_th={self.capacity_mwh_th}]"
            )
        if self.charge_max_mw_th < 0 or self.discharge_max_mw_th < 0:
            raise ValueError(f"storage {self.id!r}: charge/discharge limits must be >= 0")
        if self.loss_mwh_per_step < 0:
            raise ValueError(f"storage {self.id!r}: loss_mwh_per_step must be >= 0")
        if not self.floor_mwh_th <= self.soc_init_mwh_th <= self.capacity_mwh_th:
            raise ValueError(
                f"storage {self.id!r}: soc_init_mwh_th={self.soc_init_mwh_th} must be "
                f"in [floor={self.floor_mwh_th}, capacity={self.capacity_mwh_th}]"
            )


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
        if self.dt_h <= 0:
            raise ValueError(f"dt_h must be > 0, got {self.dt_h}")
        if self.gas_price_eur_mwh_hs < 0:
            raise ValueError("gas_price_eur_mwh_hs must be >= 0")
        if self.co2_factor_t_per_mwh_hs < 0 or self.co2_price_eur_per_t < 0:
            raise ValueError("co2 factor/price must be >= 0")
        # Frozen dataclass: coerce list inputs to tuples without mutating self.
        for fname in ("heat_pumps", "boilers", "chps", "storages"):
            v = getattr(self, fname)
            if not isinstance(v, tuple):
                object.__setattr__(self, fname, tuple(v))
        # Global ID uniqueness
        seen: dict[str, str] = {}
        for family, items in (
            ("heat_pumps", self.heat_pumps),
            ("boilers", self.boilers),
            ("chps", self.chps),
            ("storages", self.storages),
        ):
            for it in items:
                if it.id in seen:
                    raise ValueError(
                        f"duplicate asset id {it.id!r} (in {family} and {seen[it.id]})"
                    )
                seen[it.id] = family
        # At least one heat producer (otherwise demand can never be met)
        if not (self.heat_pumps or self.boilers or self.chps):
            raise ValueError(
                "PlantConfig has zero heat producers (HP+Boiler+CHP all empty); "
                "demand cannot be met"
            )

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
        """Construct from a dict (e.g. parsed from JSON). Validates schema_version
        and per-asset shape; physical bounds are enforced by each dataclass'
        ``__post_init__``.
        """
        version = payload.get("schema_version")
        if version is None:
            raise ValueError(
                "plant config payload is missing schema_version. "
                f"Expected {CONFIG_SCHEMA_VERSION}."
            )
        if version != CONFIG_SCHEMA_VERSION:
            raise ValueError(
                f"plant config schema_version={version}, expected "
                f"{CONFIG_SCHEMA_VERSION}. No migration available."
            )
        try:
            return cls(
                dt_h=float(payload["dt_h"]),
                gas_price_eur_mwh_hs=float(payload["gas_price_eur_mwh_hs"]),
                co2_factor_t_per_mwh_hs=float(payload["co2_factor_t_per_mwh_hs"]),
                co2_price_eur_per_t=float(payload["co2_price_eur_per_t"]),
                heat_pumps=tuple(
                    HeatPumpParams(**_check_keys(item, _HP_FIELDS, "heat_pump"))
                    for item in payload.get("heat_pumps", [])
                ),
                boilers=tuple(
                    BoilerParams(**_check_keys(item, _BOILER_FIELDS, "boiler"))
                    for item in payload.get("boilers", [])
                ),
                chps=tuple(
                    CHPParams(**_check_keys(item, _CHP_FIELDS, "chp"))
                    for item in payload.get("chps", [])
                ),
                storages=tuple(
                    StorageParams(**_check_keys(item, _STORAGE_FIELDS, "storage"))
                    for item in payload.get("storages", [])
                ),
            )
        except KeyError as e:
            raise ValueError(f"plant config missing required field: {e.args[0]!r}") from e

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


_HP_FIELDS = {"id", "p_el_min_mw", "p_el_max_mw", "cop"}
_BOILER_FIELDS = {"id", "q_min_mw_th", "q_max_mw_th", "eff", "min_up_steps", "min_down_steps"}
_CHP_FIELDS = {
    "id", "p_el_min_mw", "p_el_max_mw", "eff_el", "eff_th",
    "min_up_steps", "min_down_steps", "startup_cost_eur",
}
_STORAGE_FIELDS = {
    "id", "capacity_mwh_th", "floor_mwh_th", "charge_max_mw_th",
    "discharge_max_mw_th", "loss_mwh_per_step", "soc_init_mwh_th",
}


def _check_keys(item: dict, expected: set[str], family: str) -> dict:
    """Raise on unknown or missing keys so typos in plant_config.json fail loud."""
    if not isinstance(item, dict):
        raise ValueError(f"{family} entry must be a dict, got {type(item).__name__}")
    extra = set(item) - expected
    missing = expected - set(item)
    if extra or missing:
        raise ValueError(
            f"{family} entry id={item.get('id', '?')!r} has wrong fields. "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    return item


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


def _check_id(asset_id: str) -> None:
    if not isinstance(asset_id, str):
        raise ValueError(f"asset id must be a string, got {asset_id!r}")
    # Reject empty, whitespace-only, and leading/trailing-whitespace IDs:
    # an id like " hp " would silently fail to match "hp" in downstream
    # parquet groupbys.
    if not asset_id or asset_id.strip() != asset_id:
        raise ValueError(
            f"asset id must be a non-empty string with no leading/trailing "
            f"whitespace, got {asset_id!r}"
        )


def _check_pos(value: float, name: str, asset_id: str) -> None:
    if not value > 0:
        raise ValueError(f"{name}={value} for asset {asset_id!r} must be > 0")
