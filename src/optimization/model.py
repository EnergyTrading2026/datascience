"""MILP builder. Indexed Pyomo formulation: variables, constraints, and the
objective are all built per-asset over Pyomo Sets keyed by asset id, so the
same code handles 0..N heat pumps, boilers, CHPs, and storages.

Configured with a single ``PlantConfig.legacy_default()``, this reproduces the
old hardcoded MILP exactly (verified by ``test_regression_pin``).
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from optimization.config import (
    BoilerParams,
    CHPParams,
    HeatPumpParams,
    PlantConfig,
    StorageParams,
)
from optimization.state import DispatchState

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
    if not forecast.index.equals(prices.index):
        if not forecast.index.tz_convert("UTC").equals(prices.index.tz_convert("UTC")):
            raise ValueError("forecast and prices indices do not align (compared in UTC)")
    if forecast.isna().any() or prices.isna().any():
        raise ValueError("forecast/prices contain NaN")
    return len(forecast)


def build_model(
    forecast_demand_mw_th: pd.Series,
    da_prices_eur_mwh: pd.Series,
    state: DispatchState,
    config: PlantConfig,
    demand_safety_factor: float = 1.0,
    resolution: Resolution = "hour",
) -> pyo.ConcreteModel:
    """Build a dispatch MILP over the joint horizon of forecast and prices.

    Args:
        forecast_demand_mw_th: demand (MW_th), tz-aware index at `resolution` granularity.
        da_prices_eur_mwh: DA prices (EUR/MWh_el), tz-aware index aligned to forecast.
        state: carry-over state from previous commit (or DispatchState.cold_start).
            Must have an entry for every asset in ``config``.
        config: plant configuration (asset lists + globals).
        demand_safety_factor: planner sees demand * factor (>=1 = robust planning).
        resolution: 'hour' or 'quarterhour'. Determines the repeat factor used
            to broadcast inputs onto the 15-min model grid.

    Returns:
        pyomo.ConcreteModel ready to be solved by `solve.solve(...)`. Stashes
        ``m._config``, ``m._init_state``, ``m._horizon_hours``, ``m._T``,
        ``m._horizon_start`` for downstream use.
    """
    if resolution not in _REPEAT_FACTOR:
        raise ValueError(f"resolution must be 'hour' or 'quarterhour'; got {resolution!r}")
    state.covers(config)

    slots = _validate_inputs(forecast_demand_mw_th, da_prices_eur_mwh)
    dt = config.dt_h
    repeat_factor = _REPEAT_FACTOR[resolution]
    T = slots * repeat_factor
    horizon_hours = T // INT_PER_HOUR
    demand_15 = np.repeat(
        np.asarray(forecast_demand_mw_th.to_numpy(), float) * demand_safety_factor,
        repeat_factor,
    )
    price_15 = np.repeat(np.asarray(da_prices_eur_mwh.to_numpy(), float), repeat_factor)

    # By-id lookups for closure-friendly bounds + rules.
    hp_by_id: dict[str, HeatPumpParams] = {hp.id: hp for hp in config.heat_pumps}
    boiler_by_id: dict[str, BoilerParams] = {b.id: b for b in config.boilers}
    chp_by_id: dict[str, CHPParams] = {c.id: c for c in config.chps}
    storage_by_id: dict[str, StorageParams] = {s.id: s for s in config.storages}

    m = pyo.ConcreteModel("DistrictHeatingMILP")

    m.T_set = pyo.RangeSet(1, T)
    m.T0_set = pyo.RangeSet(0, T)  # SoC lives on T+1 timestamps (initial + T endings)

    # Asset sets: ordered so Pyomo iteration is deterministic. Insertion order
    # equals config order, which equals legacy_default order — important for the
    # regression pin (HiGHS branching can depend on var-creation order).
    m.HP_SET = pyo.Set(initialize=[hp.id for hp in config.heat_pumps], ordered=True)
    m.BOILER_SET = pyo.Set(initialize=[b.id for b in config.boilers], ordered=True)
    m.CHP_SET = pyo.Set(initialize=[c.id for c in config.chps], ordered=True)
    m.STORAGE_SET = pyo.Set(initialize=[s.id for s in config.storages], ordered=True)

    m.demand = pyo.Param(m.T_set, initialize={t: demand_15[t - 1] for t in range(1, T + 1)})
    m.da_price = pyo.Param(m.T_set, initialize={t: price_15[t - 1] for t in range(1, T + 1)})

    # ---------------------- Variables ----------------------

    # Heat pumps
    m.z_hp = pyo.Var(m.HP_SET, m.T_set, within=pyo.Binary)
    m.P_hp_el = pyo.Var(
        m.HP_SET, m.T_set,
        within=pyo.NonNegativeReals,
        bounds=lambda m, i, t: (0, hp_by_id[i].p_el_max_mw),
    )
    m.Q_hp = pyo.Var(m.HP_SET, m.T_set, within=pyo.NonNegativeReals)

    # Boilers
    m.z_boiler = pyo.Var(m.BOILER_SET, m.T_set, within=pyo.Binary)
    m.s_boiler = pyo.Var(m.BOILER_SET, m.T_set, within=pyo.Binary)
    m.d_boiler = pyo.Var(m.BOILER_SET, m.T_set, within=pyo.Binary)
    m.Q_boiler = pyo.Var(
        m.BOILER_SET, m.T_set,
        within=pyo.NonNegativeReals,
        bounds=lambda m, i, t: (0, boiler_by_id[i].q_max_mw_th),
    )
    m.F_boiler = pyo.Var(m.BOILER_SET, m.T_set, within=pyo.NonNegativeReals)

    # CHPs
    m.z_chp = pyo.Var(m.CHP_SET, m.T_set, within=pyo.Binary)
    m.s_chp = pyo.Var(m.CHP_SET, m.T_set, within=pyo.Binary)
    m.d_chp = pyo.Var(m.CHP_SET, m.T_set, within=pyo.Binary)
    m.P_chp_el = pyo.Var(
        m.CHP_SET, m.T_set,
        within=pyo.NonNegativeReals,
        bounds=lambda m, i, t: (0, chp_by_id[i].p_el_max_mw),
    )
    m.Q_chp = pyo.Var(m.CHP_SET, m.T_set, within=pyo.NonNegativeReals)
    m.F_chp = pyo.Var(m.CHP_SET, m.T_set, within=pyo.NonNegativeReals)

    # Storages
    m.y_sto = pyo.Var(m.STORAGE_SET, m.T_set, within=pyo.Binary)
    m.Q_charge = pyo.Var(
        m.STORAGE_SET, m.T_set,
        within=pyo.NonNegativeReals,
        bounds=lambda m, s, t: (0, storage_by_id[s].charge_max_mw_th),
    )
    m.Q_discharge = pyo.Var(
        m.STORAGE_SET, m.T_set,
        within=pyo.NonNegativeReals,
        bounds=lambda m, s, t: (0, storage_by_id[s].discharge_max_mw_th),
    )
    m.SoC = pyo.Var(
        m.STORAGE_SET, m.T0_set,
        within=pyo.NonNegativeReals,
        bounds=lambda m, s, t: (storage_by_id[s].floor_mwh_th, storage_by_id[s].capacity_mwh_th),
    )

    # ---------------------- Disabled assets ----------------------
    # Operator-forced off-switch: the asset stays registered (and keeps its
    # state) but every binary commit variable is pinned to 0, which through
    # the per-asset min/max constraints below forces P_el / Q / fuel to 0
    # too. Storages additionally get their charge/discharge flows fixed to 0
    # and a frozen SoC dynamic (see _soc_dyn) so a disabled storage doesn't
    # silently drift below floor over multi-cycle disables — turning the
    # asset off is a pause, not a slow drain.
    #
    # Start (s_*) and shutdown (d_*) markers are pinned to 0 as well: if the
    # prior state had the unit on (state.on==1) and disable forces z=0, the
    # ``_b_shut`` / ``_c_shut`` rules would otherwise emit d=1 at t=1, which
    # downstream dispatch consumers would read as the optimizer's choice to
    # shut down. With s_/d_ pinned to 0 the start/shut rules are skipped at
    # the constraint level (below) so no z_prev=1 + z=0 contradiction fires,
    # and the extracted dispatch reports the asset as "untouched, off" — the
    # honest representation of an operator-forced disable.
    #
    # The per-asset min-up/down init-state-fixing blocks below explicitly
    # skip disabled units: forcing z=1 to honor a residual min-up while
    # also forcing z=0 to disable is infeasible.
    disabled_ids: frozenset[str] = frozenset(config.disabled_asset_ids)
    for hp in config.heat_pumps:
        if hp.id in disabled_ids:
            for t in m.T_set:
                m.z_hp[hp.id, t].fix(0)
    for b in config.boilers:
        if b.id in disabled_ids:
            for t in m.T_set:
                m.z_boiler[b.id, t].fix(0)
                m.s_boiler[b.id, t].fix(0)
                m.d_boiler[b.id, t].fix(0)
    for c in config.chps:
        if c.id in disabled_ids:
            for t in m.T_set:
                m.z_chp[c.id, t].fix(0)
                m.s_chp[c.id, t].fix(0)
                m.d_chp[c.id, t].fix(0)
    for s in config.storages:
        if s.id in disabled_ids:
            for t in m.T_set:
                m.Q_charge[s.id, t].fix(0.0)
                m.Q_discharge[s.id, t].fix(0.0)

    # ---------------------- Objective ----------------------

    gas_cost = config.gas_cost_eur_mwh_hs

    def _obj(m: pyo.ConcreteModel) -> pyo.Expression:
        cost_hp = sum(
            m.P_hp_el[i, t] * m.da_price[t] * dt for i in m.HP_SET for t in m.T_set
        )
        cost_boiler = sum(
            m.F_boiler[i, t] * gas_cost * dt for i in m.BOILER_SET for t in m.T_set
        )
        cost_chp_fuel = sum(
            m.F_chp[i, t] * gas_cost * dt for i in m.CHP_SET for t in m.T_set
        )
        cost_chp_start = sum(
            chp_by_id[i].startup_cost_eur * m.s_chp[i, t]
            for i in m.CHP_SET for t in m.T_set
        )
        rev_chp = sum(
            m.P_chp_el[i, t] * m.da_price[t] * dt
            for i in m.CHP_SET for t in m.T_set
        )
        return cost_hp + cost_boiler + cost_chp_fuel + cost_chp_start - rev_chp

    m.obj = pyo.Objective(rule=_obj, sense=pyo.minimize)

    # ---------------------- Energy balance ----------------------

    def _balance(m: pyo.ConcreteModel, t: int) -> pyo.Expression:
        hp_q = sum(m.Q_hp[i, t] for i in m.HP_SET)
        boiler_q = sum(m.Q_boiler[i, t] for i in m.BOILER_SET)
        chp_q = sum(m.Q_chp[i, t] for i in m.CHP_SET)
        sto_net = sum(
            m.Q_discharge[s, t] - m.Q_charge[s, t] for s in m.STORAGE_SET
        )
        return hp_q + boiler_q + chp_q + sto_net == m.demand[t]

    m.energy_balance = pyo.Constraint(m.T_set, rule=_balance)

    # ---------------------- Heat pump constraints ----------------------

    m.hp_min = pyo.Constraint(
        m.HP_SET, m.T_set,
        rule=lambda m, i, t: m.P_hp_el[i, t] >= hp_by_id[i].p_el_min_mw * m.z_hp[i, t],
    )
    m.hp_max = pyo.Constraint(
        m.HP_SET, m.T_set,
        rule=lambda m, i, t: m.P_hp_el[i, t] <= hp_by_id[i].p_el_max_mw * m.z_hp[i, t],
    )
    m.hp_th = pyo.Constraint(
        m.HP_SET, m.T_set,
        rule=lambda m, i, t: m.Q_hp[i, t] == hp_by_id[i].cop * m.P_hp_el[i, t],
    )

    # ---------------------- Boiler constraints ----------------------

    m.b_min = pyo.Constraint(
        m.BOILER_SET, m.T_set,
        rule=lambda m, i, t: m.Q_boiler[i, t] >= boiler_by_id[i].q_min_mw_th * m.z_boiler[i, t],
    )
    m.b_max = pyo.Constraint(
        m.BOILER_SET, m.T_set,
        rule=lambda m, i, t: m.Q_boiler[i, t] <= boiler_by_id[i].q_max_mw_th * m.z_boiler[i, t],
    )
    m.b_fuel = pyo.Constraint(
        m.BOILER_SET, m.T_set,
        rule=lambda m, i, t: m.F_boiler[i, t] == m.Q_boiler[i, t] / boiler_by_id[i].eff,
    )

    # Initial-state fixing: if the unit just started/stopped, the first few
    # timesteps are forced to keep its commitment until min-up/down clears.
    # Disabled boilers are already fixed to z=0 above; skipping them here
    # avoids the infeasible fix(1) + fix(0) collision when a unit is
    # disabled mid-run.
    for b in config.boilers:
        if b.id in disabled_ids:
            continue
        init = state.units[b.id]
        if init.on == 1 and init.time_in_state_steps < b.min_up_steps:
            for t in range(1, min(b.min_up_steps - init.time_in_state_steps + 1, T + 1)):
                m.z_boiler[b.id, t].fix(1)
        elif init.on == 0 and init.time_in_state_steps < b.min_down_steps:
            for t in range(1, min(b.min_down_steps - init.time_in_state_steps + 1, T + 1)):
                m.z_boiler[b.id, t].fix(0)

    def _b_start(m: pyo.ConcreteModel, i: str, t: int) -> pyo.Expression:
        if i in disabled_ids:
            return pyo.Constraint.Skip
        z_prev = state.units[i].on if t == 1 else m.z_boiler[i, t - 1]
        return m.s_boiler[i, t] >= m.z_boiler[i, t] - z_prev

    def _b_shut(m: pyo.ConcreteModel, i: str, t: int) -> pyo.Expression:
        if i in disabled_ids:
            return pyo.Constraint.Skip
        z_prev = state.units[i].on if t == 1 else m.z_boiler[i, t - 1]
        return m.d_boiler[i, t] >= z_prev - m.z_boiler[i, t]

    m.b_start = pyo.Constraint(m.BOILER_SET, m.T_set, rule=_b_start)
    m.b_shut = pyo.Constraint(m.BOILER_SET, m.T_set, rule=_b_shut)

    def _b_minup(m: pyo.ConcreteModel, i: str, t: int) -> pyo.Expression:
        L = boiler_by_id[i].min_up_steps
        return sum(m.s_boiler[i, k] for k in range(max(1, t - L + 1), t + 1)) <= m.z_boiler[i, t]

    def _b_mindn(m: pyo.ConcreteModel, i: str, t: int) -> pyo.Expression:
        L = boiler_by_id[i].min_down_steps
        return (
            sum(m.d_boiler[i, k] for k in range(max(1, t - L + 1), t + 1))
            <= 1 - m.z_boiler[i, t]
        )

    m.b_minup = pyo.Constraint(m.BOILER_SET, m.T_set, rule=_b_minup)
    m.b_mindn = pyo.Constraint(m.BOILER_SET, m.T_set, rule=_b_mindn)

    # ---------------------- CHP constraints ----------------------

    m.c_min = pyo.Constraint(
        m.CHP_SET, m.T_set,
        rule=lambda m, i, t: m.P_chp_el[i, t] >= chp_by_id[i].p_el_min_mw * m.z_chp[i, t],
    )
    m.c_max = pyo.Constraint(
        m.CHP_SET, m.T_set,
        rule=lambda m, i, t: m.P_chp_el[i, t] <= chp_by_id[i].p_el_max_mw * m.z_chp[i, t],
    )
    m.c_th = pyo.Constraint(
        m.CHP_SET, m.T_set,
        rule=lambda m, i, t: m.Q_chp[i, t] == chp_by_id[i].heat_power_ratio * m.P_chp_el[i, t],
    )
    m.c_fuel = pyo.Constraint(
        m.CHP_SET, m.T_set,
        rule=lambda m, i, t: m.F_chp[i, t] == m.P_chp_el[i, t] / chp_by_id[i].eff_el,
    )

    for c in config.chps:
        if c.id in disabled_ids:
            continue
        init = state.units[c.id]
        if init.on == 1 and init.time_in_state_steps < c.min_up_steps:
            for t in range(1, min(c.min_up_steps - init.time_in_state_steps + 1, T + 1)):
                m.z_chp[c.id, t].fix(1)
        elif init.on == 0 and init.time_in_state_steps < c.min_down_steps:
            for t in range(1, min(c.min_down_steps - init.time_in_state_steps + 1, T + 1)):
                m.z_chp[c.id, t].fix(0)

    def _c_start(m: pyo.ConcreteModel, i: str, t: int) -> pyo.Expression:
        if i in disabled_ids:
            return pyo.Constraint.Skip
        z_prev = state.units[i].on if t == 1 else m.z_chp[i, t - 1]
        return m.s_chp[i, t] >= m.z_chp[i, t] - z_prev

    def _c_shut(m: pyo.ConcreteModel, i: str, t: int) -> pyo.Expression:
        if i in disabled_ids:
            return pyo.Constraint.Skip
        z_prev = state.units[i].on if t == 1 else m.z_chp[i, t - 1]
        return m.d_chp[i, t] >= z_prev - m.z_chp[i, t]

    m.c_start = pyo.Constraint(m.CHP_SET, m.T_set, rule=_c_start)
    m.c_shut = pyo.Constraint(m.CHP_SET, m.T_set, rule=_c_shut)

    def _c_minup(m: pyo.ConcreteModel, i: str, t: int) -> pyo.Expression:
        L = chp_by_id[i].min_up_steps
        return sum(m.s_chp[i, k] for k in range(max(1, t - L + 1), t + 1)) <= m.z_chp[i, t]

    def _c_mindn(m: pyo.ConcreteModel, i: str, t: int) -> pyo.Expression:
        L = chp_by_id[i].min_down_steps
        return (
            sum(m.d_chp[i, k] for k in range(max(1, t - L + 1), t + 1))
            <= 1 - m.z_chp[i, t]
        )

    m.c_minup = pyo.Constraint(m.CHP_SET, m.T_set, rule=_c_minup)
    m.c_mindn = pyo.Constraint(m.CHP_SET, m.T_set, rule=_c_mindn)

    # ---------------------- Storage constraints ----------------------

    m.soc_init = pyo.Constraint(
        m.STORAGE_SET,
        rule=lambda m, s: m.SoC[s, 0] == state.storages[s].soc_mwh_th,
    )
    m.charge_excl = pyo.Constraint(
        m.STORAGE_SET, m.T_set,
        rule=lambda m, s, t: m.Q_charge[s, t]
        <= storage_by_id[s].charge_max_mw_th * m.y_sto[s, t],
    )
    m.discharge_excl = pyo.Constraint(
        m.STORAGE_SET, m.T_set,
        rule=lambda m, s, t: m.Q_discharge[s, t]
        <= storage_by_id[s].discharge_max_mw_th * (1 - m.y_sto[s, t]),
    )
    def _soc_dyn(m: pyo.ConcreteModel, s: str, t: int) -> pyo.Expression:
        if s in disabled_ids:
            # Disabled storage: charge/discharge fixed to 0 above, and the
            # SoC is frozen across the horizon. Skipping the loss term is a
            # deliberate operator-facing contract — "off = pause" — so that
            # a multi-cycle disable cannot quietly drain SoC below floor.
            return m.SoC[s, t] == m.SoC[s, t - 1]
        return (
            m.SoC[s, t]
            == m.SoC[s, t - 1]
            + m.Q_charge[s, t] * dt
            - m.Q_discharge[s, t] * dt
            - storage_by_id[s].loss_mwh_per_step
        )

    m.soc_dyn = pyo.Constraint(m.STORAGE_SET, m.T_set, rule=_soc_dyn)

    # Stash for downstream extract_dispatch / extract_state
    m._config = config
    m._init_state = state
    m._horizon_hours = horizon_hours
    m._T = T
    m._horizon_start = forecast_demand_mw_th.index[0]
    return m
