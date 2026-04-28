"""Shared fixtures for optimization tests."""
from __future__ import annotations

import pandas as pd
import pytest

from optimization.config import PlantParams, RuntimeConfig
from optimization.state import DispatchState


@pytest.fixture
def params() -> PlantParams:
    return PlantParams()


@pytest.fixture
def params_no_floor() -> PlantParams:
    """Notebook self-test reference uses bounds (0, 200). For regression checks."""
    return PlantParams(sto_floor_mwh_th=0.0)


@pytest.fixture
def runtime() -> RuntimeConfig:
    return RuntimeConfig()


@pytest.fixture
def cold_state() -> DispatchState:
    return DispatchState.cold_start(pd.Timestamp("2026-01-01 00:00:00", tz="Europe/Berlin"))


@pytest.fixture
def constant_inputs_35h() -> tuple[pd.Series, pd.Series]:
    """Trivial 35h horizon: constant demand and price. For smoke tests of build_model."""
    idx = pd.date_range("2026-01-01 00:00:00", periods=35, freq="1h", tz="Europe/Berlin")
    demand = pd.Series(10.0, index=idx, name="demand_mw_th")
    price = pd.Series(60.0, index=idx, name="price_eur_mwh")
    return demand, price
