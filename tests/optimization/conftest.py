"""Shared fixtures for optimization tests."""
from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from optimization.config import PlantConfig, RuntimeConfig
from optimization.state import DispatchState


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: tests that exercise the solver at the documented scaling envelope",
    )


@pytest.fixture
def config() -> PlantConfig:
    return PlantConfig.legacy_default()


@pytest.fixture
def config_no_floor() -> PlantConfig:
    """Notebook self-test reference uses bounds (0, 200). For regression checks."""
    base = PlantConfig.legacy_default()
    storages = tuple(replace(s, floor_mwh_th=0.0) for s in base.storages)
    return replace(base, storages=storages)


@pytest.fixture
def runtime() -> RuntimeConfig:
    return RuntimeConfig()


@pytest.fixture
def cold_state(config) -> DispatchState:
    return DispatchState.cold_start(config, pd.Timestamp("2026-01-01 00:00:00", tz="Europe/Berlin"))


@pytest.fixture
def constant_inputs_35h() -> tuple[pd.Series, pd.Series]:
    """Trivial 35h horizon: constant demand and price. For smoke tests of build_model."""
    idx = pd.date_range("2026-01-01 00:00:00", periods=35, freq="1h", tz="Europe/Berlin")
    demand = pd.Series(10.0, index=idx, name="demand_mw_th")
    price = pd.Series(60.0, index=idx, name="price_eur_mwh")
    return demand, price
