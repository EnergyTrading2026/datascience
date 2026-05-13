"""One-shot harvest script for the pre-refactor MILP goldens.

ARCHIVED: this file targets the **pre-refactor** API (PlantParams, flat
DispatchState fields). It is intentionally NOT runnable on this branch
(feat/modular-assets) or any later branch — it imports symbols
(`PlantParams`, `DispatchState(sto_soc_mwh_th=..., hp_on=..., ...)`) that no
longer exist. It lives under scripts/_archive/ so static-import linters and
"run all scripts" CI checks skip it.

Re-harvesting workflow (only if the regression pin has legitimately drifted
and needs re-baselining against the original solver):
  1. git checkout 8751915  # last pre-refactor commit
  2. copy this file to scripts/ on that checkout (8751915 doesn't have
     scripts/_archive/) and run: python scripts/harvest_regression_pin.py
  3. paste objective + dispatch + final state into test_regression_pin.py
  4. git checkout feat/modular-assets (or wherever the pin test lives)

Re-running should not be necessary in normal operation — drift in the goldens
means the new code's behavior changed, which is exactly what the pin is
designed to detect.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from optimization.config import PlantParams, RuntimeConfig
from optimization.dispatch import extract_dispatch, extract_state
from optimization.model import build_model
from optimization.solve import solve
from optimization.state import DispatchState


def make_inputs() -> tuple[pd.Series, pd.Series, DispatchState, PlantParams, RuntimeConfig]:
    idx = pd.date_range("2026-01-01 00:00:00", periods=35, freq="1h", tz="Europe/Berlin")
    # Demand profile: cold-morning peak, midday valley, evening peak.
    hours = np.arange(35) % 24
    demand = 8.0 + 4.0 * np.sin((hours - 6) / 24 * 2 * np.pi)  # 4..12 MW_th
    forecast = pd.Series(demand, index=idx, name="demand_mw_th")
    # Price profile: cheap night, expensive day, very expensive evening peak.
    price = 40.0 + 30.0 * np.sin((hours - 9) / 24 * 2 * np.pi) + 5.0 * (hours == 18)
    prices = pd.Series(price, index=idx, name="price_eur_mwh")
    state = DispatchState(
        timestamp=pd.Timestamp("2026-01-01 00:00:00", tz="Europe/Berlin"),
        sto_soc_mwh_th=120.0,
        hp_on=0,
        boiler_on=1,
        boiler_time_in_state_steps=8,
        chp_on=0,
        chp_time_in_state_steps=20,
    )
    params = PlantParams()
    runtime = RuntimeConfig()
    return forecast, prices, state, params, runtime


def main() -> None:
    forecast, prices, state, params, runtime = make_inputs()
    model = build_model(forecast, prices, state, params, resolution="hour")
    result = solve(model, time_limit_s=runtime.solver_time_limit_s, mip_gap=runtime.solver_mip_gap)
    n_intervals = runtime.commit_hours * 4
    dispatch = extract_dispatch(model, n_intervals=n_intervals, solve_time=forecast.index[0])
    new_state = extract_state(
        model,
        t_end=n_intervals,
        commit_end_time=forecast.index[0] + pd.Timedelta(hours=runtime.commit_hours),
    )

    out = {
        "objective_eur": result.objective_eur,
        "expected_cost_eur": dispatch.expected_cost_eur,
        "hp_p_el_mw": dispatch.hp_p_el_mw,
        "boiler_q_th_mw": dispatch.boiler_q_th_mw,
        "chp_p_el_mw": dispatch.chp_p_el_mw,
        "sto_charge_mw_th": dispatch.sto_charge_mw_th,
        "sto_discharge_mw_th": dispatch.sto_discharge_mw_th,
        "soc_trajectory_mwh_th": dispatch.soc_trajectory_mwh_th,
        "new_state": {
            "sto_soc_mwh_th": new_state.sto_soc_mwh_th,
            "hp_on": new_state.hp_on,
            "boiler_on": new_state.boiler_on,
            "boiler_time_in_state_steps": new_state.boiler_time_in_state_steps,
            "chp_on": new_state.chp_on,
            "chp_time_in_state_steps": new_state.chp_time_in_state_steps,
        },
    }
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
