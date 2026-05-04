"""Plant constants from `docs/optimization/optimization_problem.md` + runtime knobs.

PlantParams are physics/economics — change only when the spec changes.
RuntimeConfig is solver/operational tuning — change per deployment.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PlantParams:
    """Physical and economic constants. Source: docs/optimization/optimization_problem.md."""

    # Time grid
    dt_h: float = 0.25  # 15-min step in hours

    # Fuel + CO2
    gas_price_eur_mwh_hs: float = 35.0
    co2_factor_t_per_mwh_hs: float = 0.201
    co2_price_eur_per_t: float = 60.0

    # Storage
    sto_capacity_mwh_th: float = 200.0
    sto_floor_mwh_th: float = 50.0  # hard floor, every t (spec)
    sto_charge_max_mw_th: float = 15.0
    sto_discharge_max_mw_th: float = 15.0
    sto_loss_mwh_per_step: float = 0.000125
    sto_soc_init_mwh_th: float = 200.0  # default cold-start

    # Heat pump (no min-run/down)
    hp_p_el_min_mw: float = 1.0
    hp_p_el_max_mw: float = 8.0
    hp_cop: float = 3.5

    # Condensing boiler (1h min-run/down)
    boiler_q_min_mw_th: float = 2.0
    boiler_q_max_mw_th: float = 12.0
    boiler_eff: float = 0.97
    boiler_min_up_steps: int = 4   # 1h
    boiler_min_down_steps: int = 4

    # CHP (2h min-run/down)
    chp_p_el_min_mw: float = 2.0
    chp_p_el_max_mw: float = 6.0
    chp_eff_el: float = 0.40
    chp_eff_th: float = 0.48
    chp_min_up_steps: int = 8   # 2h
    chp_min_down_steps: int = 8
    chp_startup_cost_eur: float = 600.0

    @property
    def gas_cost_eur_mwh_hs(self) -> float:
        """Effective fuel cost = gas + CO2 (EUR / MWh_Hs)."""
        return self.gas_price_eur_mwh_hs + self.co2_factor_t_per_mwh_hs * self.co2_price_eur_per_t

    @property
    def chp_heat_power_ratio(self) -> float:
        """Q_th = ratio * P_el for CHP (= eff_th / eff_el = 1.2)."""
        return self.chp_eff_th / self.chp_eff_el


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
