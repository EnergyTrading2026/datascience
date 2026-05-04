"""Adapter tests — SMARD CSV slicing + forecast parquet schema validation."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from optimization.adapters import forecast as forecast_io
from optimization.adapters import smard as smard_io
from optimization.adapters import smard_live as smard_live_io


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
    with pytest.raises(forecast_io.ForecastSchemaError, match="hour"):
        forecast_io.load_forecast(path, pd.Timestamp("2026-01-01", tz="Europe/Berlin"))


# ---------------- Forecast at quarter-hour resolution ----------------
def test_forecast_loads_quarterhour(tmp_path):
    idx = pd.date_range("2026-01-01 13:00:00", periods=140, freq="15min", tz="Europe/Berlin")
    path = _write_forecast(tmp_path, idx, [10.0] * 140)
    s = forecast_io.load_forecast(
        path,
        pd.Timestamp("2026-01-01 13:00:00", tz="Europe/Berlin"),
        resolution="quarterhour",
    )
    assert len(s) == 140
    assert (s.index[1] - s.index[0]) == pd.Timedelta(minutes=15)


def test_forecast_quarterhour_anchor_rounds_up(tmp_path):
    idx = pd.date_range("2026-01-01 13:00:00", periods=140, freq="15min", tz="Europe/Berlin")
    path = _write_forecast(tmp_path, idx, list(range(140)))
    # Request at 13:07 -> anchor 13:15 -> first value should be 1 (the second row)
    s = forecast_io.load_forecast(
        path,
        pd.Timestamp("2026-01-01 13:07:00", tz="Europe/Berlin"),
        resolution="quarterhour",
    )
    assert s.iloc[0] == 1.0


def test_forecast_quarterhour_rejects_hourly_file(tmp_path):
    idx = pd.date_range("2026-01-01 13:00:00", periods=24, freq="1h", tz="Europe/Berlin")
    path = _write_forecast(tmp_path, idx, [10.0] * 24)
    with pytest.raises(forecast_io.ForecastSchemaError, match="quarterhour"):
        forecast_io.load_forecast(
            path,
            pd.Timestamp("2026-01-01 13:00:00", tz="Europe/Berlin"),
            resolution="quarterhour",
        )


def test_forecast_rejects_invalid_resolution(tmp_path):
    idx = pd.date_range("2026-01-01", periods=24, freq="1h", tz="Europe/Berlin")
    path = _write_forecast(tmp_path, idx, [10.0] * 24)
    with pytest.raises(ValueError, match="resolution"):
        forecast_io.load_forecast(
            path,
            pd.Timestamp("2026-01-01", tz="Europe/Berlin"),
            resolution="weekly",  # type: ignore[arg-type]
        )


# ---------------- SMARD live adapter (mocked HTTP) ----------------
def _index_payload(bucket_starts_utc: list[pd.Timestamp]) -> dict:
    """Build a minimal index_*.json response."""
    return {"timestamps": [int(b.timestamp() * 1000) for b in bucket_starts_utc]}


def _bucket_payload(start_utc: pd.Timestamp, n: int, step: pd.Timedelta, base_price: float) -> dict:
    """Build a minimal {filter}_{region}_{resolution}_{bucket_ms}.json response.

    Generates n slots starting at start_utc; price varies as base + i.
    """
    series = []
    for i in range(n):
        ts_ms = int((start_utc + i * step).timestamp() * 1000)
        series.append([ts_ms, base_price + i])
    return {"series": series}


@pytest.fixture
def mock_smard_hour(monkeypatch):
    """Mock the live SMARD HTTP layer with a single deterministic week of hourly data.

    Returns the URLs that were requested, for assertions.
    """
    bucket_start = pd.Timestamp("2026-04-27 22:00:00", tz="UTC")  # Sunday 22:00 UTC = Mon 00:00 Berlin
    requested: list[str] = []

    def fake_get(url: str) -> dict:
        requested.append(url)
        if url.endswith("index_hour.json"):
            return _index_payload([bucket_start])
        if "_hour_" in url:
            return _bucket_payload(bucket_start, 168, pd.Timedelta(hours=1), base_price=50.0)
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(smard_live_io, "_http_get_json_retrying", fake_get)
    return requested


@pytest.fixture
def mock_smard_quarterhour(monkeypatch):
    bucket_start = pd.Timestamp("2026-04-27 22:00:00", tz="UTC")
    requested: list[str] = []

    def fake_get(url: str) -> dict:
        requested.append(url)
        if url.endswith("index_quarterhour.json"):
            return _index_payload([bucket_start])
        if "_quarterhour_" in url:
            return _bucket_payload(bucket_start, 672, pd.Timedelta(minutes=15), base_price=50.0)
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(smard_live_io, "_http_get_json_retrying", fake_get)
    return requested


def test_smard_live_returns_hourly_series_anchored(mock_smard_hour):
    at_time = pd.Timestamp("2026-04-28 13:00:00", tz="Europe/Berlin")
    s = smard_live_io.get_published_prices(at_time, resolution="hour", horizon_h=12)
    assert isinstance(s, pd.Series)
    assert s.name == "price_eur_mwh"
    assert s.index.tz is not None
    assert s.index[0] == at_time
    assert (s.index[1] - s.index[0]) == pd.Timedelta(hours=1)
    assert len(s) == 12
    assert not s.isna().any()


def test_smard_live_returns_quarterhour_series_anchored(mock_smard_quarterhour):
    at_time = pd.Timestamp("2026-04-28 13:00:00", tz="Europe/Berlin")
    s = smard_live_io.get_published_prices(at_time, resolution="quarterhour", horizon_h=2)
    assert (s.index[1] - s.index[0]) == pd.Timedelta(minutes=15)
    assert len(s) == 8  # 2h * 4 slots/h
    assert s.index[0] == at_time


def test_smard_live_anchor_rounds_up_hour(mock_smard_hour):
    at_time = pd.Timestamp("2026-04-28 13:30:00", tz="Europe/Berlin")
    s = smard_live_io.get_published_prices(at_time, resolution="hour", horizon_h=4)
    assert s.index[0] == pd.Timestamp("2026-04-28 14:00:00", tz="Europe/Berlin")


def test_smard_live_anchor_rounds_up_quarterhour(mock_smard_quarterhour):
    at_time = pd.Timestamp("2026-04-28 13:07:00", tz="Europe/Berlin")
    s = smard_live_io.get_published_prices(at_time, resolution="quarterhour", horizon_h=1)
    assert s.index[0] == pd.Timestamp("2026-04-28 13:15:00", tz="Europe/Berlin")


def test_smard_live_rejects_naive_timestamp():
    with pytest.raises(ValueError, match="tz-aware"):
        smard_live_io.get_published_prices(pd.Timestamp("2026-04-28 13:00:00"))


def test_smard_live_rejects_invalid_resolution():
    with pytest.raises(ValueError, match="resolution"):
        smard_live_io.get_published_prices(
            pd.Timestamp("2026-04-28 13:00:00", tz="Europe/Berlin"),
            resolution="daily",  # type: ignore[arg-type]
        )


def test_smard_live_drops_trailing_nulls(monkeypatch):
    """Unpublished slots come back as null and must be stripped."""
    bucket_start = pd.Timestamp("2026-04-27 22:00:00", tz="UTC")
    step = pd.Timedelta(hours=1)

    def fake_get(url: str) -> dict:
        if url.endswith("index_hour.json"):
            return _index_payload([bucket_start])
        # 168 slots in the week, but slots 24+ are null (auction not cleared yet).
        series = []
        for i in range(168):
            ts_ms = int((bucket_start + i * step).timestamp() * 1000)
            series.append([ts_ms, 42.0 if i < 24 else None])
        return {"series": series}

    monkeypatch.setattr(smard_live_io, "_http_get_json_retrying", fake_get)

    at_time = pd.Timestamp("2026-04-28 00:00:00", tz="Europe/Berlin")
    s = smard_live_io.get_published_prices(at_time, resolution="hour", horizon_h=48)
    # Bucket starts Mon 00:00 Berlin (= Sun 22:00 UTC); the 24 published slots
    # cover the whole bucket day. Anchor at 00:00 Berlin -> 24 published values.
    assert len(s) == 24
    assert not s.isna().any()


def test_smard_live_retries_on_5xx_then_succeeds(monkeypatch):
    """5xx errors must be retried (mapped to SmardApiUnavailableError), not propagated."""
    bucket_start = pd.Timestamp("2026-04-27 22:00:00", tz="UTC")
    calls = {"n": 0}

    def fake_inner(url: str) -> dict:
        # Patches the inner single-shot fetcher: simulates what the real
        # _http_get_json does for 5xx (translate to SmardApiUnavailableError)
        # so that the retry loop in _http_get_json_retrying can catch it.
        calls["n"] += 1
        if calls["n"] <= 2:
            raise smard_live_io.SmardApiUnavailableError(f"HTTP 503 from {url}")
        if url.endswith("index_hour.json"):
            return _index_payload([bucket_start])
        return _bucket_payload(bucket_start, 168, pd.Timedelta(hours=1), base_price=50.0)

    monkeypatch.setattr(smard_live_io, "_http_get_json", fake_inner)
    monkeypatch.setattr(smard_live_io, "_RETRY_DELAYS_S", (0.0, 0.0, 0.0))

    at_time = pd.Timestamp("2026-04-28 13:00:00", tz="Europe/Berlin")
    s = smard_live_io.get_published_prices(at_time, resolution="hour", horizon_h=4)
    assert len(s) == 4
    # 2 retries for the index call (calls 1,2 fail; 3 = index), then 1 bucket call (4).
    assert calls["n"] == 4


def test_smard_live_raises_after_retry_exhaustion(monkeypatch):
    def always_fail(url: str) -> dict:
        raise smard_live_io.SmardApiUnavailableError(f"HTTP 503 from {url}")

    monkeypatch.setattr(smard_live_io, "_http_get_json", always_fail)
    monkeypatch.setattr(smard_live_io, "_RETRY_DELAYS_S", (0.0, 0.0, 0.0))

    at_time = pd.Timestamp("2026-04-28 13:00:00", tz="Europe/Berlin")
    with pytest.raises(smard_live_io.SmardApiUnavailableError):
        smard_live_io.get_published_prices(at_time, resolution="hour", horizon_h=4)


def test_smard_live_does_not_retry_on_4xx(monkeypatch):
    """4xx surfaces as SmardSchemaError; no retry loop."""
    calls = {"n": 0}

    def fake_inner(url: str) -> dict:
        calls["n"] += 1
        # Real _http_get_json maps 4xx to SmardSchemaError; mirror that here.
        raise smard_live_io.SmardSchemaError(f"HTTP 404 from {url}")

    monkeypatch.setattr(smard_live_io, "_http_get_json", fake_inner)

    at_time = pd.Timestamp("2026-04-28 13:00:00", tz="Europe/Berlin")
    with pytest.raises(smard_live_io.SmardSchemaError):
        smard_live_io.get_published_prices(at_time, resolution="hour", horizon_h=4)
    assert calls["n"] == 1  # No retry on 4xx


def test_smard_live_http_layer_translates_errors():
    """Direct test of the inner _http_get_json: maps urllib errors to SmardLive types."""
    import urllib.error
    from unittest.mock import patch

    # 5xx -> SmardApiUnavailableError
    err_5xx = urllib.error.HTTPError("http://x", 503, "down", {}, None)
    with patch("urllib.request.urlopen", side_effect=err_5xx):
        with pytest.raises(smard_live_io.SmardApiUnavailableError):
            smard_live_io._http_get_json("http://x")

    # 4xx -> SmardSchemaError
    err_4xx = urllib.error.HTTPError("http://x", 404, "missing", {}, None)
    with patch("urllib.request.urlopen", side_effect=err_4xx):
        with pytest.raises(smard_live_io.SmardSchemaError):
            smard_live_io._http_get_json("http://x")

    # network error -> SmardApiUnavailableError
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("dns fail")):
        with pytest.raises(smard_live_io.SmardApiUnavailableError):
            smard_live_io._http_get_json("http://x")


def test_smard_live_raises_on_malformed_payload(monkeypatch):
    def fake_get(url: str) -> dict:
        return {"unexpected_key": []}

    monkeypatch.setattr(smard_live_io, "_http_get_json_retrying", fake_get)

    at_time = pd.Timestamp("2026-04-28 13:00:00", tz="Europe/Berlin")
    with pytest.raises(smard_live_io.SmardSchemaError, match="timestamps"):
        smard_live_io.get_published_prices(at_time, resolution="hour", horizon_h=4)
