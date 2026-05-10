"""Extract committed dispatch + carry-over state from a solved model.

The committed dispatch is the FIRST `n_intervals` worth of decisions —
this is what the solver locks in for the current cycle. Everything beyond
that is "advisory" and will be re-planned in the next hourly solve.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd
import pyomo.environ as pyo

from src.optimization.config import PlantParams
from src.optimization.state import TIS_LONG, DispatchState

INT_PER_HOUR = 4


@dataclass
class Dispatch:
    """Committed setpoints for the first `n_intervals` of a solved model.

    Each setpoint list has length n_intervals at 15-min resolution.
    soc_trajectory has length n_intervals + 1 (initial SoC + each step end).
    """

    solve_time: pd.Timestamp     # tz-aware, when solve happened
    commit_start: pd.Timestamp   # tz-aware, first interval start (== model._horizon_start)
    n_intervals: int

    hp_p_el_mw: list[float]
    boiler_q_th_mw: list[float]
    chp_p_el_mw: list[float]
    sto_charge_mw_th: list[float]
    sto_discharge_mw_th: list[float]

    soc_trajectory_mwh_th: list[float]
    expected_cost_eur: float

    def to_dataframe(self) -> pd.DataFrame:
        """Tabular view: one row per 15-min step, indexed by absolute timestamp.

        Columns are setpoints + soc_end. soc_end of step k = soc_trajectory[k+1].
        """
        idx = pd.date_range(
            self.commit_start,
            periods=self.n_intervals,
            freq="15min",
            tz=self.commit_start.tz,
        )
        return pd.DataFrame(
            {
                "hp_p_el_mw": self.hp_p_el_mw,
                "boiler_q_th_mw": self.boiler_q_th_mw,
                "chp_p_el_mw": self.chp_p_el_mw,
                "sto_charge_mw_th": self.sto_charge_mw_th,
                "sto_discharge_mw_th": self.sto_discharge_mw_th,
                "soc_end_mwh_th": self.soc_trajectory_mwh_th[1:],
            },
            index=idx,
        )


def _v(var, i: int) -> float:
    return float(pyo.value(var[i]))


def _bin(var, i: int) -> int:
    return int(round(pyo.value(var[i])))


def extract_dispatch(
    model: pyo.ConcreteModel,
    n_intervals: int,
    solve_time: pd.Timestamp,
) -> Dispatch:
    """Pull committed setpoints from intervals 1..n_intervals of the solved model.

    Args:
        model: solved model (build_model has stashed _params, _T, _horizon_start).
        n_intervals: number of 15-min intervals to commit (4 = 1h commit).
        solve_time: when the solve happened (Berlin tz-aware).

    Returns:
        Dispatch with setpoint lists + cost breakdown.
    """
    if not 1 <= n_intervals <= model._T:
        raise ValueError(f"n_intervals={n_intervals} out of [1, {model._T}]")

    p: PlantParams = model._params
    dt = p.dt_h
    ts = list(range(1, n_intervals + 1))

    hp_p_el = [_v(model.P_hp_el, t) for t in ts]
    boiler_q = [_v(model.Q_boiler, t) for t in ts]
    chp_p_el = [_v(model.P_chp_el, t) for t in ts]
    s_chp = [_bin(model.s_chp, t) for t in ts]
    sto_chg = [_v(model.Q_charge, t) for t in ts]
    sto_dis = [_v(model.Q_discharge, t) for t in ts]
    soc = [float(pyo.value(model.SoC[t])) for t in range(0, n_intervals + 1)]

    cost_hp = sum(hp_p_el[i] * float(pyo.value(model.da_price[ts[i]])) * dt for i in range(n_intervals))
    cost_b = sum(boiler_q[i] / p.boiler_eff * p.gas_cost_eur_mwh_hs * dt for i in range(n_intervals))
    cost_cf = sum(chp_p_el[i] / p.chp_eff_el * p.gas_cost_eur_mwh_hs * dt for i in range(n_intervals))
    cost_cs = sum(s_chp) * p.chp_startup_cost_eur
    rev_chp = sum(
        chp_p_el[i] * float(pyo.value(model.da_price[ts[i]])) * dt for i in range(n_intervals)
    )
    expected_cost = cost_hp + cost_b + cost_cf + cost_cs - rev_chp

    return Dispatch(
        solve_time=solve_time,
        commit_start=model._horizon_start,
        n_intervals=n_intervals,
        hp_p_el_mw=hp_p_el,
        boiler_q_th_mw=boiler_q,
        chp_p_el_mw=chp_p_el,
        sto_charge_mw_th=sto_chg,
        sto_discharge_mw_th=sto_dis,
        soc_trajectory_mwh_th=soc,
        expected_cost_eur=expected_cost,
    )


def extract_state(
    model: pyo.ConcreteModel,
    t_end: int,
    commit_end_time: pd.Timestamp,
) -> DispatchState:
    """Carry-over state at end of interval t_end.

    Handles the subtle case where the run extends back into the carried-over
    initial state with no switch in [1, t_end] — in that case the time-in-state
    accumulates across the boundary. (Direct port of mpc_prototype.ipynb cell 3
    `extract_state` logic, validated against the notebook self-test.)
    """
    T = model._T
    if not 1 <= t_end <= T:
        raise ValueError(f"t_end={t_end} out of [1,{T}]")

    init: DispatchState = model._init_state

    def tis(var, z0: int, tis0: int) -> tuple[int, int]:
        z_final = _bin(var, t_end)
        count = 0
        for t in range(t_end, 0, -1):
            if _bin(var, t) == z_final:
                count += 1
            else:
                break
        if count == t_end and z0 == z_final:
            return z_final, min(count + tis0, TIS_LONG)
        return z_final, count

    z_h = _bin(model.z_hp, t_end)
    z_b, b_tis = tis(model.z_boiler, init.boiler_on, init.boiler_time_in_state_steps)
    z_c, c_tis = tis(model.z_chp, init.chp_on, init.chp_time_in_state_steps)

    return DispatchState(
        timestamp=commit_end_time,
        sto_soc_mwh_th=float(pyo.value(model.SoC[t_end])),
        hp_on=z_h,
        boiler_on=z_b,
        boiler_time_in_state_steps=b_tis,
        chp_on=z_c,
        chp_time_in_state_steps=c_tis,
    )
