"""Smoke tests for the backtest harness.

Verifies:
  - end-to-end MPC backtest over a short range succeeds and produces records
  - the strategy_fn hook works (custom strategies plug in correctly)
  - StrategyInfeasibleError is recorded and the harness recovers
  - write_outputs produces the expected files
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from optimization.backtest import (
    BacktestResult,
    StrategyInfeasibleError,
    StrategyResult,
    mpc_strategy,
    run_backtest,
    write_outputs,
)
from optimization.config import PlantParams, RuntimeConfig
from optimization.dispatch import Dispatch
from optimization.state import DispatchState


@pytest.fixture
def constant_inputs_long():
    """48h of constant demand and price — long enough for several MPC cycles."""
    idx = pd.date_range("2026-01-01 00:00:00", periods=48, freq="1h", tz="Europe/Berlin")
    demand = pd.Series(10.0, index=idx, name="demand_mw_th")
    price = pd.Series(60.0, index=idx, name="price_eur_mwh")
    return demand, price


def test_mpc_backtest_three_cycles(constant_inputs_long):
    """3 hourly MPC cycles on constant inputs: harness loop + state carry-over works."""
    demand, prices = constant_inputs_long
    start = demand.index[0]
    end = start + pd.Timedelta(hours=3)

    result = run_backtest(demand=demand, prices=prices, start=start, end=end, log_every=0)

    assert isinstance(result, BacktestResult)
    assert len(result.records) == 3
    assert result.summary["n_cycles"] == 3
    assert result.summary["n_infeasible"] == 0
    assert result.summary["n_skipped_short"] == 0
    # Constant 10 MW_th demand for 3h fits entirely inside the 200 MWh storage,
    # so the optimal plan runs no units and the expected cost is 0.
    assert result.summary["total_expected_cost_eur"] == pytest.approx(0.0, abs=1e-6)

    # Dispatch log: 3 cycles * 4 intervals/h * 1h commit = 12 rows
    assert len(result.dispatch_log) == 3 * 4
    expected_cols = {
        "hp_p_el_mw", "boiler_q_th_mw", "chp_p_el_mw",
        "sto_charge_mw_th", "sto_discharge_mw_th", "soc_end_mwh_th",
    }
    assert expected_cols <= set(result.dispatch_log.columns)

    # SoC stays within bounds (default floor=50, capacity=200)
    assert result.summary["soc_end"]["min_mwh"] >= 50.0 - 1e-6
    assert result.summary["soc_end"]["max_mwh"] <= 200.0 + 1e-6


def test_strategy_fn_hook_is_called(constant_inputs_long):
    """Harness must use the strategy_fn argument for every cycle."""
    demand, prices = constant_inputs_long
    start = demand.index[0]
    end = start + pd.Timedelta(hours=2)
    runtime = RuntimeConfig()
    params = PlantParams()

    calls: list[pd.Timestamp] = []

    def stub_strategy(forecast, prices_, state, params_, runtime_, solve_time):
        calls.append(solve_time)
        # Reuse mpc_strategy to produce a valid result without re-implementing it.
        return mpc_strategy(forecast, prices_, state, params_, runtime_, solve_time)

    result = run_backtest(
        demand=demand, prices=prices, start=start, end=end,
        params=params, runtime=runtime, strategy_fn=stub_strategy, log_every=0,
    )
    assert len(calls) == 2
    assert calls == [start, start + pd.Timedelta(hours=1)]
    assert len(result.records) == 2


def test_infeasible_strategy_recorded_and_recovered(constant_inputs_long):
    """StrategyInfeasibleError must be counted and the next cycle cold-starts."""
    demand, prices = constant_inputs_long
    start = demand.index[0]
    end = start + pd.Timedelta(hours=3)

    call_count = {"n": 0}

    def flaky_strategy(forecast, prices_, state, params_, runtime_, solve_time):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise StrategyInfeasibleError("simulated failure")
        return mpc_strategy(forecast, prices_, state, params_, runtime_, solve_time)

    result = run_backtest(
        demand=demand, prices=prices, start=start, end=end,
        strategy_fn=flaky_strategy, log_every=0,
    )
    assert result.summary["n_infeasible"] == 1
    assert result.summary["n_cycles"] == 2
    assert call_count["n"] == 3


def test_skip_when_horizon_too_short():
    """Horizon below RuntimeConfig.horizon_hours_min must skip, not crash."""
    # Only 8h of data, min horizon is 12h -> first cycle skipped.
    idx = pd.date_range("2026-01-01 00:00:00", periods=8, freq="1h", tz="Europe/Berlin")
    demand = pd.Series(10.0, index=idx, name="demand_mw_th")
    prices = pd.Series(60.0, index=idx, name="price_eur_mwh")

    result = run_backtest(
        demand=demand, prices=prices,
        start=idx[0], end=idx[0] + pd.Timedelta(hours=2),
        log_every=0,
    )
    assert result.summary["n_cycles"] == 0
    assert result.summary["n_skipped_short"] == 2


def test_write_outputs_creates_files(tmp_path, constant_inputs_long):
    demand, prices = constant_inputs_long
    start = demand.index[0]
    result = run_backtest(
        demand=demand, prices=prices,
        start=start, end=start + pd.Timedelta(hours=2),
        log_every=0,
    )

    out_dir = tmp_path / "backtest"
    write_outputs(result, out_dir)

    assert (out_dir / "summary.json").exists()
    assert (out_dir / "records.parquet").exists()
    assert (out_dir / "dispatch_log.parquet").exists()

    payload = json.loads((out_dir / "summary.json").read_text())
    assert "summary" in payload and "params" in payload and "runtime" in payload
    assert payload["summary"]["n_cycles"] == 2

    records = pd.read_parquet(out_dir / "records.parquet")
    assert len(records) == 2
    dispatch = pd.read_parquet(out_dir / "dispatch_log.parquet")
    assert len(dispatch) == 2 * 4
