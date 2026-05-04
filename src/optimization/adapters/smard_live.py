"""SMARD live DA-price adapter — pulls directly from the chart_data JSON API.

The CSV-based `adapters/smard.py` stays in place for backtest (historical bulk
data lives in a CSV). This adapter is the production fetcher: no DevOps in
between, just the live SMARD endpoint.

API reference: https://www.smard.de/app/chart_data
- Filter 4169 = DE/LU day-ahead price (verified to return real
  quarter-hour-DA-auction prices when called with resolution='quarterhour';
  hourly resolution is the volume-weighted aggregate).
- Region 'DE' covers the post-2018-10-01 DE/LU bidding zone.
- Data is served in weekly buckets; an index endpoint lists available bucket
  start timestamps (UTC ms).

Auction schedule: EPEX clears at ~12:45 CET; SMARD mirrors with delay. From
~13:45 Berlin local, "tomorrow 00:00–24:00" is reliably available. Before then,
only "today's remaining hours" are guaranteed published. The caller (`run.py`)
reconciles via horizon length — this adapter just returns whatever is live.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Literal

import pandas as pd

logger = logging.getLogger(__name__)

BASE = "https://www.smard.de/app/chart_data"
FILTER = 4169
REGION = "DE"
BERLIN = "Europe/Berlin"

Resolution = Literal["hour", "quarterhour"]
_FLOOR_FREQ: dict[str, str] = {"hour": "1h", "quarterhour": "15min"}

# Three retries with exponential backoff (1s, 2s, 4s). Keeps the worst-case
# wall-clock cost ~7s, well within the hourly cycle's budget.
_RETRY_DELAYS_S: tuple[float, ...] = (1.0, 2.0, 4.0)
_HTTP_TIMEOUT_S = 15.0
_DEFAULT_HORIZON_H = 48  # covers MPC's 35h target + headroom (one extra week-bucket worst case)


class SmardLiveError(RuntimeError):
    """Base for live SMARD adapter errors."""


class SmardApiUnavailableError(SmardLiveError):
    """SMARD endpoint unreachable or returning 5xx after retry exhaustion.

    `run.py` treats this as recoverable (exit 1) — DevOps' scheduler retries
    the next cycle.
    """


class SmardSchemaError(SmardLiveError):
    """SMARD returned 200 but the payload doesn't match the expected schema.

    Non-recoverable from the optimizer's perspective — likely an API contract
    change. `run.py` lets this bubble to exit 3 (page).
    """


def _http_get_json(url: str) -> dict:
    """Single HTTP GET, returns parsed JSON. Raises typed SmardLive errors.

    Module-level so tests can monkeypatch this single function instead of
    digging into urllib.
    """
    try:
        with urllib.request.urlopen(url, timeout=_HTTP_TIMEOUT_S) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        if 500 <= e.code < 600:
            raise SmardApiUnavailableError(f"HTTP {e.code} from {url}") from e
        raise SmardSchemaError(f"HTTP {e.code} from {url}") from e
    except urllib.error.URLError as e:
        raise SmardApiUnavailableError(f"network error fetching {url}: {e.reason}") from e
    except (json.JSONDecodeError, ValueError) as e:
        raise SmardSchemaError(f"non-JSON response from {url}: {e}") from e


def _http_get_json_retrying(url: str) -> dict:
    """Retry _http_get_json on transient errors only (5xx / network)."""
    last_err: SmardApiUnavailableError | None = None
    for attempt, delay in enumerate((0.0, *_RETRY_DELAYS_S)):
        if delay > 0:
            time.sleep(delay)
        try:
            return _http_get_json(url)
        except SmardApiUnavailableError as e:
            last_err = e
            logger.warning("SMARD attempt %d/%d failed: %s",
                           attempt + 1, len(_RETRY_DELAYS_S) + 1, e)
    assert last_err is not None
    raise last_err


def _fetch_week(bucket_ms: int, resolution: Resolution) -> pd.DataFrame:
    url = f"{BASE}/{FILTER}/{REGION}/{FILTER}_{REGION}_{resolution}_{bucket_ms}.json"
    payload = _http_get_json_retrying(url)
    series = payload.get("series")
    if not isinstance(series, list):
        raise SmardSchemaError(f"missing or invalid 'series' in {url}")
    df = pd.DataFrame(series, columns=["ts_ms", "price_eur_mwh"])
    df["timestamp_utc"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    return df[["timestamp_utc", "price_eur_mwh"]]


def _fetch_window(
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
    resolution: Resolution,
) -> pd.DataFrame:
    """Pull every weekly bucket overlapping [start_utc, end_utc) and slice.

    Returns timestamps in UTC, prices possibly null (caller decides how to
    handle null tail = unpublished slots).
    """
    index_url = f"{BASE}/{FILTER}/{REGION}/index_{resolution}.json"
    index = _http_get_json_retrying(index_url)
    timestamps = index.get("timestamps")
    if not isinstance(timestamps, list):
        raise SmardSchemaError(f"missing or invalid 'timestamps' in {index_url}")
    buckets = pd.to_datetime(timestamps, unit="ms", utc=True)
    one_week = pd.Timedelta(days=7)
    needed_ms = [
        int(b.timestamp() * 1000)
        for b in buckets
        if (b + one_week) > start_utc and b <= end_utc
    ]
    if not needed_ms:
        return pd.DataFrame(columns=["timestamp_utc", "price_eur_mwh"])
    frames = [_fetch_week(b, resolution) for b in needed_ms]
    df = pd.concat(frames, ignore_index=True).sort_values("timestamp_utc")
    df = df[(df.timestamp_utc >= start_utc) & (df.timestamp_utc < end_utc)]
    return df.reset_index(drop=True)


def get_published_prices(
    at_time: pd.Timestamp,
    resolution: Resolution = "hour",
    horizon_h: int = _DEFAULT_HORIZON_H,
) -> pd.Series:
    """Return DA prices live from SMARD, anchored at the next slot boundary.

    Args:
        at_time: tz-aware Berlin timestamp (current solve time). The first
            index value is the smallest slot boundary >= at_time.
        resolution: 'hour' or 'quarterhour'. Must match what build_model
            expects (and what the demand forecast was generated at).
        horizon_h: forward window in hours to fetch. Default 48 covers the
            MPC's 35h target plus headroom. Returned series may be shorter
            if the next-day auction has not cleared yet.

    Returns:
        pd.Series[float], tz-aware Europe/Berlin index, name='price_eur_mwh'.
        Index gap is 1h ('hour') or 15min ('quarterhour'). Trailing nulls
        (= unpublished slots) are dropped — caller does horizon reconciliation.

    Raises:
        ValueError: at_time is not tz-aware, or resolution is invalid.
        SmardApiUnavailableError: SMARD endpoint unreachable after retries.
        SmardSchemaError: response payload doesn't match expected schema.
    """
    if at_time.tzinfo is None:
        raise ValueError(f"at_time must be tz-aware; got {at_time!r}")
    if resolution not in ("hour", "quarterhour"):
        raise ValueError(f"resolution must be 'hour' or 'quarterhour'; got {resolution!r}")
    if horizon_h <= 0:
        raise ValueError(f"horizon_h must be positive; got {horizon_h}")

    freq = _FLOOR_FREQ[resolution]
    at_berlin = at_time.tz_convert(BERLIN)
    floored = at_berlin.floor(freq)
    anchor = floored if floored == at_berlin else at_berlin.ceil(freq)
    end = anchor + pd.Timedelta(hours=horizon_h)

    df = _fetch_window(anchor.tz_convert("UTC"), end.tz_convert("UTC"), resolution)

    s = pd.Series(
        pd.to_numeric(df["price_eur_mwh"], errors="coerce").to_numpy(dtype=float),
        index=pd.DatetimeIndex(df["timestamp_utc"]).tz_convert(BERLIN),
        name="price_eur_mwh",
    )
    s = s.dropna()
    s.index.name = "timestamp_berlin"
    return s
