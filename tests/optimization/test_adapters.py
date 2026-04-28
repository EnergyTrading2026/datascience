"""Adapter tests — SMARD CSV slicing + forecast parquet schema validation."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from optimization.adapters import forecast as forecast_io
from optimization.adapters import smard as smard_io


# ---------------- SMARD ----------------
PRICES_CSV = Path("data/Gro_handelspreise_202403010000_202603020000_Stunde.csv")


@pytest.mark.skipif(not PRICES_CSV.exists(), reason="SMARD CSV not present in data/")
def test_smard_returns_hourly_series_anchored_at_request():
    at_time = pd.Timestamp("2025-12-01 13:00:00", tz="Europe/Berlin")
    s = smard_io.get_published_prices(at_time, PRICES_CSV)
    assert isinstance(s, pd.Series)
    assert s.name == "price_eur_mwh"
    assert s.index.tz is not None
    assert s.index[0] == at_time
    # Hourly frequency
    if len(s) >= 2:
        assert (s.index[1] - s.index[0]) == pd.Timedelta(hours=1)
    assert not s.isna().any()


@pytest.mark.skipif(not PRICES_CSV.exists(), reason="SMARD CSV not present in data/")
def test_smard_anchor_rounds_up_to_next_full_hour():
    at_time = pd.Timestamp("2025-12-01 13:30:00", tz="Europe/Berlin")
    s = smard_io.get_published_prices(at_time, PRICES_CSV)
    assert s.index[0] == pd.Timestamp("2025-12-01 14:00:00", tz="Europe/Berlin")


def test_smard_rejects_naive_timestamp():
    with pytest.raises(ValueError, match="tz-aware"):
        smard_io.get_published_prices(pd.Timestamp("2025-12-01 13:00:00"), Path("ignored"))


# ---------------- Forecast ----------------
def _write_forecast(tmp_path, idx, values, col=forecast_io.DEMAND_COLUMN):
    path = tmp_path / "forecast.parquet"
    pd.DataFrame({col: values}, index=idx).to_parquet(path)
    return path


def test_forecast_loads_valid_parquet(tmp_path):
    idx = pd.date_range("2026-01-01 13:00:00", periods=48, freq="1h", tz="Europe/Berlin")
    path = _write_forecast(tmp_path, idx, [10.0] * 48)
    s = forecast_io.load_forecast(path, pd.Timestamp("2026-01-01 13:00:00", tz="Europe/Berlin"))
    assert len(s) == 48
    assert s.name == forecast_io.DEMAND_COLUMN
    assert s.iloc[0] == 10.0


def test_forecast_anchor_rounds_up(tmp_path):
    idx = pd.date_range("2026-01-01 13:00:00", periods=48, freq="1h", tz="Europe/Berlin")
    path = _write_forecast(tmp_path, idx, list(range(48)))
    # Request at 13:30 -> anchor 14:00 -> first value should be 1 (the second row)
    s = forecast_io.load_forecast(path, pd.Timestamp("2026-01-01 13:30:00", tz="Europe/Berlin"))
    assert s.iloc[0] == 1.0


def test_forecast_rejects_naive_index(tmp_path):
    idx = pd.date_range("2026-01-01", periods=24, freq="1h")  # no tz
    path = _write_forecast(tmp_path, idx, [10.0] * 24)
    with pytest.raises(forecast_io.ForecastSchemaError, match="tz-aware"):
        forecast_io.load_forecast(path, pd.Timestamp("2026-01-01", tz="Europe/Berlin"))


def test_forecast_rejects_missing_column(tmp_path):
    idx = pd.date_range("2026-01-01", periods=24, freq="1h", tz="Europe/Berlin")
    path = _write_forecast(tmp_path, idx, [10.0] * 24, col="wrong_column")
    with pytest.raises(forecast_io.ForecastSchemaError, match="demand_mw_th"):
        forecast_io.load_forecast(path, pd.Timestamp("2026-01-01", tz="Europe/Berlin"))


def test_forecast_rejects_nan(tmp_path):
    idx = pd.date_range("2026-01-01", periods=24, freq="1h", tz="Europe/Berlin")
    vals = [10.0] * 24
    vals[5] = float("nan")
    path = _write_forecast(tmp_path, idx, vals)
    with pytest.raises(forecast_io.ForecastSchemaError, match="NaN"):
        forecast_io.load_forecast(path, pd.Timestamp("2026-01-01", tz="Europe/Berlin"))


def test_forecast_rejects_non_hourly(tmp_path):
    idx = pd.date_range("2026-01-01", periods=24, freq="30min", tz="Europe/Berlin")
    path = _write_forecast(tmp_path, idx, [10.0] * 24)
    with pytest.raises(forecast_io.ForecastSchemaError, match="hourly"):
        forecast_io.load_forecast(path, pd.Timestamp("2026-01-01", tz="Europe/Berlin"))
