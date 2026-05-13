"""Extract committed dispatch + carry-over state from a solved model.

The committed dispatch is the FIRST `n_intervals` worth of decisions —
this is what the solver locks in for the current cycle. Everything beyond
that is "advisory" and will be re-planned in the next hourly solve.

All per-asset quantities are dicts keyed by asset id, matching ``PlantConfig``.
``Dispatch.to_dataframe()`` returns a long-format frame
(timestamp, asset_id, family, quantity, value) so the row count scales with
the asset count instead of the column count.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import pyomo.environ as pyo

from optimization.config import PlantConfig
from optimization.state import TIS_LONG, DispatchState, StorageState, UnitState

INT_PER_HOUR = 4


@dataclass
class Dispatch:
    """Committed setpoints for the first `n_intervals` of a solved model.

    Each per-asset value is a list of length n_intervals at 15-min resolution.
    ``soc_trajectory_mwh_th[asset_id]`` has length n_intervals + 1
    (initial SoC + each step end).
    """

    solve_time: pd.Timestamp     # tz-aware, when solve happened
    commit_start: pd.Timestamp   # tz-aware, first interval start (== model._horizon_start)
    n_intervals: int

    hp_p_el_mw: dict[str, list[float]] = field(default_factory=dict)
    boiler_q_th_mw: dict[str, list[float]] = field(default_factory=dict)
    chp_p_el_mw: dict[str, list[float]] = field(default_factory=dict)
    sto_charge_mw_th: dict[str, list[float]] = field(default_factory=dict)
    sto_discharge_mw_th: dict[str, list[float]] = field(default_factory=dict)
    soc_trajectory_mwh_th: dict[str, list[float]] = field(default_factory=dict)

    expected_cost_eur: float = 0.0

    def to_dataframe(self) -> pd.DataFrame:
        """Long-format tabular view of the committed dispatch.

        Index: tz-aware DatetimeIndex (non-unique) at 15-min resolution.
        Columns: ``asset_id``, ``family``, ``quantity``, ``value``.

        ``quantity`` is one of: ``p_el_mw`` (hp/chp), ``q_th_mw`` (boiler),
        ``charge_mw_th`` / ``discharge_mw_th`` / ``soc_end_mwh_th`` (storage).
        ``soc_end_mwh_th[t]`` is the SoC at the end of step ``t`` (= soc_trajectory[1:]).

        Long over wide because dashboard charting and per-asset aggregation
        (``df.groupby(['family','quantity'])['value'].sum()``) stay constant
        as the asset count grows.
        """
        idx = pd.date_range(
            self.commit_start,
            periods=self.n_intervals,
            freq="15min",
            tz=self.commit_start.tz,
        )
        rows: list[dict] = []

        def _emit(family: str, quantity: str, by_asset: dict[str, list[float]]) -> None:
            for asset_id, vals in by_asset.items():
                for ts, v in zip(idx, vals, strict=True):
                    rows.append({
                        "timestamp": ts,
                        "asset_id": asset_id,
                        "family": family,
                        "quantity": quantity,
                        "value": float(v),
                    })

        _emit("hp", "p_el_mw", self.hp_p_el_mw)
        _emit("boiler", "q_th_mw", self.boiler_q_th_mw)
        _emit("chp", "p_el_mw", self.chp_p_el_mw)
        _emit("storage", "charge_mw_th", self.sto_charge_mw_th)
        _emit("storage", "discharge_mw_th", self.sto_discharge_mw_th)
        _emit(
            "storage", "soc_end_mwh_th",
            {a: vals[1:] for a, vals in self.soc_trajectory_mwh_th.items()},
        )

        if not rows:
            return pd.DataFrame(
                columns=["asset_id", "family", "quantity", "value"],
                index=pd.DatetimeIndex([], tz=self.commit_start.tz, name="timestamp"),
            )
        df = pd.DataFrame(rows).set_index("timestamp")
        return df


def _v(var, asset_id: str, t: int) -> float:
    return float(pyo.value(var[asset_id, t]))


def _bin(var, asset_id: str, t: int) -> int:
    return int(round(pyo.value(var[asset_id, t])))


def extract_dispatch(
    model: pyo.ConcreteModel,
    n_intervals: int,
    solve_time: pd.Timestamp,
) -> Dispatch:
    """Pull committed setpoints from intervals 1..n_intervals of the solved model.

    Args:
        model: solved model (build_model has stashed _config, _T, _horizon_start).
        n_intervals: number of 15-min intervals to commit (4 = 1h commit).
        solve_time: when the solve happened (Berlin tz-aware).

    Returns:
        Dispatch with per-asset setpoint dicts + cost breakdown.
    """
    if not 1 <= n_intervals <= model._T:
        raise ValueError(f"n_intervals={n_intervals} out of [1, {model._T}]")

    cfg: PlantConfig = model._config
    dt = cfg.dt_h
    ts = list(range(1, n_intervals + 1))

    hp_p_el = {
        hp.id: [_v(model.P_hp_el, hp.id, t) for t in ts] for hp in cfg.heat_pumps
    }
    boiler_q = {
        b.id: [_v(model.Q_boiler, b.id, t) for t in ts] for b in cfg.boilers
    }
    chp_p_el = {
        c.id: [_v(model.P_chp_el, c.id, t) for t in ts] for c in cfg.chps
    }
    s_chp = {
        c.id: [_bin(model.s_chp, c.id, t) for t in ts] for c in cfg.chps
    }
    sto_chg = {
        s.id: [_v(model.Q_charge, s.id, t) for t in ts] for s in cfg.storages
    }
    sto_dis = {
        s.id: [_v(model.Q_discharge, s.id, t) for t in ts] for s in cfg.storages
    }
    soc = {
        s.id: [float(pyo.value(model.SoC[s.id, t])) for t in range(0, n_intervals + 1)]
        for s in cfg.storages
    }

    gas_cost = cfg.gas_cost_eur_mwh_hs
    cost_hp = sum(
        hp_p_el[hp.id][i] * float(pyo.value(model.da_price[ts[i]])) * dt
        for hp in cfg.heat_pumps for i in range(n_intervals)
    )
    cost_b = sum(
        boiler_q[b.id][i] / b.eff * gas_cost * dt
        for b in cfg.boilers for i in range(n_intervals)
    )
    cost_cf = sum(
        chp_p_el[c.id][i] / c.eff_el * gas_cost * dt
        for c in cfg.chps for i in range(n_intervals)
    )
    cost_cs = sum(
        sum(s_chp[c.id]) * c.startup_cost_eur for c in cfg.chps
    )
    rev_chp = sum(
        chp_p_el[c.id][i] * float(pyo.value(model.da_price[ts[i]])) * dt
        for c in cfg.chps for i in range(n_intervals)
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
    accumulates across the boundary.
    """
    T = model._T
    if not 1 <= t_end <= T:
        raise ValueError(f"t_end={t_end} out of [1,{T}]")

    cfg: PlantConfig = model._config
    init: DispatchState = model._init_state

    def _tis_for(var, asset_id: str) -> tuple[int, int]:
        """Return (final on/off, time-in-state at t_end) for one asset."""
        z_final = _bin(var, asset_id, t_end)
        count = 0
        for t in range(t_end, 0, -1):
            if _bin(var, asset_id, t) == z_final:
                count += 1
            else:
                break
        z0 = init.units[asset_id].on
        tis0 = init.units[asset_id].time_in_state_steps
        if count == t_end and z0 == z_final:
            return z_final, min(count + tis0, TIS_LONG)
        return z_final, count

    units: dict[str, UnitState] = {}
    for hp in cfg.heat_pumps:
        # HPs have no min-up/down — pin tis to TIS_LONG so the field can never
        # be load-bearing if a future change adds an HP UC constraint.
        z_final = _bin(model.z_hp, hp.id, t_end)
        units[hp.id] = UnitState(on=z_final, time_in_state_steps=TIS_LONG)
    for b in cfg.boilers:
        z, tis = _tis_for(model.z_boiler, b.id)
        units[b.id] = UnitState(on=z, time_in_state_steps=tis)
    for c in cfg.chps:
        z, tis = _tis_for(model.z_chp, c.id)
        units[c.id] = UnitState(on=z, time_in_state_steps=tis)

    storages: dict[str, StorageState] = {
        s.id: StorageState(soc_mwh_th=float(pyo.value(model.SoC[s.id, t_end])))
        for s in cfg.storages
    }

    return DispatchState(
        timestamp=commit_end_time,
        units=units,
        storages=storages,
    )
