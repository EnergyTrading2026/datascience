"""MILP builder. Direct port of the validated formulation from
`notebooks/optimization/mpc_prototype.ipynb` (cell 3), with the spec-mandated
hard floor on storage SoC enabled by default.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from optimization.config import PlantParams
from optimization.state import TIS_LONG, DispatchState

INT_PER_HOUR = 4  # 15-min grid (model is always 15-min internally)

Resolution = Literal["hour", "quarterhour"]
_REPEAT_FACTOR: dict[str, int] = {"hour": INT_PER_HOUR, "quarterhour": 1}


def _validate_inputs(forecast: pd.Series, prices: pd.Series) -> int:
    if len(forecast) == 0:
        raise ValueError("empty forecast")
    if len(forecast) != len(prices):
        raise ValueError(
            f"forecast and prices must have equal length; got "
            f"forecast={len(forecast)}, prices={len(prices)}"
        )
    if forecast.index.tz is None or prices.index.tz is None:
        raise ValueError("forecast and prices must have tz-aware index")
    # Don't strictly require identical indices — allow Berlin/UTC equivalents — but
    # the same wall-clock instant must align.
    if not forecast.index.equals(prices.index):
        # Convert both to UTC for the comparison; if still different, fail.
        if not forecast.index.tz_convert("UTC").equals(prices.index.tz_convert("UTC")):
            raise ValueError("forecast and prices indices do not align (compared in UTC)")
    if forecast.isna().any() or prices.isna().any():
        raise ValueError("forecast/prices contain NaN")
    return len(forecast)


def build_model(
    forecast_demand_mw_th: pd.Series,
    da_prices_eur_mwh: pd.Series,
    state: DispatchState,
    params: PlantParams,
    demand_safety_factor: float = 1.0,
    resolution: Resolution = "hour",
) -> pyo.ConcreteModel:
    """Build a dispatch MILP over the joint horizon of forecast and prices.

    The MILP is the validated 15-min formulation from mpc_prototype.ipynb cell 3:
    energy balance, storage dynamics, unit min-up/min-down, CHP startup cost.
    Inputs are broadcast onto the 15-min internal grid via a per-resolution
    repeat factor (hour -> 4, quarterhour -> 1). Forecast and prices must be
    at the same resolution.

    Args:
        forecast_demand_mw_th: demand (MW_th), tz-aware index at `resolution` granularity.
        da_prices_eur_mwh: DA prices (EUR/MWh_el), tz-aware index aligned to forecast.
        state: carry-over state from previous commit (or DispatchState.cold_start).
        params: plant constants.
        demand_safety_factor: planner sees demand * factor (>=1 = robust planning).
        resolution: 'hour' or 'quarterhour'. Determines the repeat factor used
            to broadcast inputs onto the 15-min model grid.

    Returns:
        pyomo.ConcreteModel ready to be solved by `solve.solve(...)`. Stashes
        `m._params`, `m._horizon_hours`, `m._horizon_start` for downstream use.
    """
    if resolution not in _REPEAT_FACTOR:
        raise ValueError(f"resolution must be 'hour' or 'quarterhour'; got {resolution!r}")
    slots = _validate_inputs(forecast_demand_mw_th, da_prices_eur_mwh)
    p = params
    dt = p.dt_h
    repeat_factor = _REPEAT_FACTOR[resolution]
    T = slots * repeat_factor
    horizon_hours = T // INT_PER_HOUR
    demand_15 = np.repeat(
        np.asarray(forecast_demand_mw_th.to_numpy(), float) * demand_safety_factor,
        repeat_factor,
    )
    price_15 = np.repeat(np.asarray(da_prices_eur_mwh.to_numpy(), float), repeat_factor)

    m = pyo.ConcreteModel("DistrictHeatingMILP")
    m.T_set = pyo.RangeSet(1, T)
    m.demand = pyo.Param(m.T_set, initialize={t: demand_15[t - 1] for t in range(1, T + 1)})
    m.da_price = pyo.Param(m.T_set, initialize={t: price_15[t - 1] for t in range(1, T + 1)})

    # Heat pump
    m.z_hp = pyo.Var(m.T_set, within=pyo.Binary)
    m.P_hp_el = pyo.Var(m.T_set, within=pyo.NonNegativeReals, bounds=(0, p.hp_p_el_max_mw))
    m.Q_hp = pyo.Var(m.T_set, within=pyo.NonNegativeReals)

    # Boiler
    m.z_boiler = pyo.Var(m.T_set, within=pyo.Binary)
    m.s_boiler = pyo.Var(m.T_set, within=pyo.Binary)
    m.d_boiler = pyo.Var(m.T_set, within=pyo.Binary)
    m.Q_boiler = pyo.Var(m.T_set, within=pyo.NonNegativeReals, bounds=(0, p.boiler_q_max_mw_th))
    m.F_boiler = pyo.Var(m.T_set, within=pyo.NonNegativeReals)

    # CHP
    m.z_chp = pyo.Var(m.T_set, within=pyo.Binary)
    m.s_chp = pyo.Var(m.T_set, within=pyo.Binary)
    m.d_chp = pyo.Var(m.T_set, within=pyo.Binary)
    m.P_chp_el = pyo.Var(m.T_set, within=pyo.NonNegativeReals, bounds=(0, p.chp_p_el_max_mw))
    m.Q_chp = pyo.Var(m.T_set, within=pyo.NonNegativeReals)
    m.F_chp = pyo.Var(m.T_set, within=pyo.NonNegativeReals)

    # Storage. SoC bounds enforce hard floor (params.sto_floor_mwh_th, default 50).
    m.y_sto = pyo.Var(m.T_set, within=pyo.Binary)
    m.Q_charge = pyo.Var(m.T_set, within=pyo.NonNegativeReals, bounds=(0, p.sto_charge_max_mw_th))
    m.Q_discharge = pyo.Var(
        m.T_set, within=pyo.NonNegativeReals, bounds=(0, p.sto_discharge_max_mw_th)
    )
    m.SoC = pyo.Var(
        pyo.RangeSet(0, T),
        within=pyo.NonNegativeReals,
        bounds=(p.sto_floor_mwh_th, p.sto_capacity_mwh_th),
    )

    # Objective: fuel + grid - chp revenue + chp startup
    gas_cost = p.gas_cost_eur_mwh_hs
    m.obj = pyo.Objective(
        rule=lambda m: sum(
            m.P_hp_el[t] * m.da_price[t] * dt
            + m.F_boiler[t] * gas_cost * dt
            + m.F_chp[t] * gas_cost * dt
            + p.chp_startup_cost_eur * m.s_chp[t]
            - m.P_chp_el[t] * m.da_price[t] * dt
            for t in m.T_set
        ),
        sense=pyo.minimize,
    )

    # Energy balance (must be met every t, even with hard floor on storage).
    m.energy_balance = pyo.Constraint(
        m.T_set,
        rule=lambda m, t: m.Q_hp[t] + m.Q_boiler[t] + m.Q_chp[t] + m.Q_discharge[t] - m.Q_charge[t]
        == m.demand[t],
    )
    # SoC initial condition
    m.soc_init = pyo.Constraint(expr=m.SoC[0] == state.sto_soc_mwh_th)
    # Charge/discharge mutually exclusive (via binary y_sto)
    m.charge_excl = pyo.Constraint(
        m.T_set, rule=lambda m, t: m.Q_charge[t] <= p.sto_charge_max_mw_th * m.y_sto[t]
    )
    m.discharge_excl = pyo.Constraint(
        m.T_set,
        rule=lambda m, t: m.Q_discharge[t] <= p.sto_discharge_max_mw_th * (1 - m.y_sto[t]),
    )
    # SoC dynamics
    m.soc_dyn = pyo.Constraint(
        m.T_set,
        rule=lambda m, t: m.SoC[t]
        == m.SoC[t - 1] + m.Q_charge[t] * dt - m.Q_discharge[t] * dt - p.sto_loss_mwh_per_step,
    )

    # Heat pump: power-output coupling, no min-run/down
    m.hp_min = pyo.Constraint(
        m.T_set, rule=lambda m, t: m.P_hp_el[t] >= p.hp_p_el_min_mw * m.z_hp[t]
    )
    m.hp_max = pyo.Constraint(
        m.T_set, rule=lambda m, t: m.P_hp_el[t] <= p.hp_p_el_max_mw * m.z_hp[t]
    )
    m.hp_th = pyo.Constraint(m.T_set, rule=lambda m, t: m.Q_hp[t] == p.hp_cop * m.P_hp_el[t])

    # Boiler: output, fuel, min-up/down
    m.b_min = pyo.Constraint(
        m.T_set, rule=lambda m, t: m.Q_boiler[t] >= p.boiler_q_min_mw_th * m.z_boiler[t]
    )
    m.b_max = pyo.Constraint(
        m.T_set, rule=lambda m, t: m.Q_boiler[t] <= p.boiler_q_max_mw_th * m.z_boiler[t]
    )
    m.b_fuel = pyo.Constraint(
        m.T_set, rule=lambda m, t: m.F_boiler[t] == m.Q_boiler[t] / p.boiler_eff
    )
    z_b0 = state.boiler_on
    b_tis = state.boiler_time_in_state_steps
    if z_b0 == 1 and b_tis < p.boiler_min_up_steps:
        for t in range(1, min(p.boiler_min_up_steps - b_tis + 1, T + 1)):
            m.z_boiler[t].fix(1)
    elif z_b0 == 0 and b_tis < p.boiler_min_down_steps:
        for t in range(1, min(p.boiler_min_down_steps - b_tis + 1, T + 1)):
            m.z_boiler[t].fix(0)
    m.b_start = pyo.Constraint(
        m.T_set,
        rule=lambda m, t: m.s_boiler[t]
        >= m.z_boiler[t] - (z_b0 if t == 1 else m.z_boiler[t - 1]),
    )
    m.b_shut = pyo.Constraint(
        m.T_set,
        rule=lambda m, t: m.d_boiler[t]
        >= (z_b0 if t == 1 else m.z_boiler[t - 1]) - m.z_boiler[t],
    )
    Lub, Ldb = p.boiler_min_up_steps, p.boiler_min_down_steps
    m.b_minup = pyo.Constraint(
        m.T_set,
        rule=lambda m, t: sum(m.s_boiler[i] for i in range(max(1, t - Lub + 1), t + 1))
        <= m.z_boiler[t],
    )
    m.b_mindn = pyo.Constraint(
        m.T_set,
        rule=lambda m, t: sum(m.d_boiler[i] for i in range(max(1, t - Ldb + 1), t + 1))
        <= 1 - m.z_boiler[t],
    )

    # CHP: output, fuel, min-up/down, startup cost (in objective)
    m.c_min = pyo.Constraint(
        m.T_set, rule=lambda m, t: m.P_chp_el[t] >= p.chp_p_el_min_mw * m.z_chp[t]
    )
    m.c_max = pyo.Constraint(
        m.T_set, rule=lambda m, t: m.P_chp_el[t] <= p.chp_p_el_max_mw * m.z_chp[t]
    )
    m.c_th = pyo.Constraint(
        m.T_set, rule=lambda m, t: m.Q_chp[t] == p.chp_heat_power_ratio * m.P_chp_el[t]
    )
    m.c_fuel = pyo.Constraint(
        m.T_set, rule=lambda m, t: m.F_chp[t] == m.P_chp_el[t] / p.chp_eff_el
    )
    z_c0 = state.chp_on
    c_tis = state.chp_time_in_state_steps
    if z_c0 == 1 and c_tis < p.chp_min_up_steps:
        for t in range(1, min(p.chp_min_up_steps - c_tis + 1, T + 1)):
            m.z_chp[t].fix(1)
    elif z_c0 == 0 and c_tis < p.chp_min_down_steps:
        for t in range(1, min(p.chp_min_down_steps - c_tis + 1, T + 1)):
            m.z_chp[t].fix(0)
    m.c_start = pyo.Constraint(
        m.T_set,
        rule=lambda m, t: m.s_chp[t] >= m.z_chp[t] - (z_c0 if t == 1 else m.z_chp[t - 1]),
    )
    m.c_shut = pyo.Constraint(
        m.T_set,
        rule=lambda m, t: m.d_chp[t] >= (z_c0 if t == 1 else m.z_chp[t - 1]) - m.z_chp[t],
    )
    Luc, Ldc = p.chp_min_up_steps, p.chp_min_down_steps
    m.c_minup = pyo.Constraint(
        m.T_set,
        rule=lambda m, t: sum(m.s_chp[i] for i in range(max(1, t - Luc + 1), t + 1))
        <= m.z_chp[t],
    )
    m.c_mindn = pyo.Constraint(
        m.T_set,
        rule=lambda m, t: sum(m.d_chp[i] for i in range(max(1, t - Ldc + 1), t + 1))
        <= 1 - m.z_chp[t],
    )

    # Stash for downstream extract_dispatch / extract_state
    m._params = p
    m._init_state = state
    m._horizon_hours = horizon_hours
    m._T = T
    m._horizon_start = forecast_demand_mw_th.index[0]
    return m
