"""Demand forecast adapter — interface to the forecasting team's output.

V1 placeholder schema (REVIEW with forecasting team before relying on this in prod):
  - File format: parquet, with tz-aware DatetimeIndex
  - Index: hourly, tz-aware Europe/Berlin
  - Column: 'demand_mw_th' (float, MW thermal)
  - Length: at least the requested horizon, starting at the next full hour after at_time
  - No NaN

Validation is strict — schema violations raise loudly so DevOps gets paged
rather than feeding garbage to the optimizer.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

BERLIN = "Europe/Berlin"
DEMAND_COLUMN = "demand_mw_th"


class ForecastSchemaError(ValueError):
    """Forecast file violates the agreed schema."""


def load_forecast(path: Path, at_time: pd.Timestamp) -> pd.Series:
    """Load + validate demand forecast from forecasting team.

    Args:
        path: file path to forecast artifact (.parquet).
        at_time: tz-aware Berlin timestamp (current solve time). The forecast must
            cover the next full hour >= at_time onward.

    Returns:
        pd.Series[float], hourly, tz-aware Europe/Berlin index, name='demand_mw_th'.
        Index starts at the next full hour >= at_time.

    Raises:
        FileNotFoundError, ForecastSchemaError, ValueError.
    """
    if at_time.tzinfo is None:
        raise ValueError(f"at_time must be tz-aware; got {at_time!r}")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"forecast file not found: {path}")

    df = pd.read_parquet(path)

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ForecastSchemaError(f"index must be DatetimeIndex; got {type(df.index).__name__}")
    if df.index.tz is None:
        raise ForecastSchemaError("index must be tz-aware")
    if DEMAND_COLUMN not in df.columns:
        raise ForecastSchemaError(
            f"missing required column '{DEMAND_COLUMN}'; got {list(df.columns)}"
        )

    s = df[DEMAND_COLUMN].astype(float)
    s.index = s.index.tz_convert(BERLIN)
    s = s.sort_index()
    s.name = DEMAND_COLUMN

    if len(s) >= 2:
        gaps = s.index.to_series().diff().dropna()
        if not (gaps == pd.Timedelta(hours=1)).all():
            raise ForecastSchemaError("forecast index is not hourly-uniform")

    if s.isna().any():
        raise ForecastSchemaError("forecast contains NaN")

    at_berlin = at_time.tz_convert(BERLIN)
    if at_berlin.minute == 0 and at_berlin.second == 0 and at_berlin.microsecond == 0:
        anchor = at_berlin
    else:
        anchor = at_berlin.ceil("1h")

    sliced = s.loc[anchor:]
    if len(sliced) == 0:
        raise ForecastSchemaError(
            f"forecast does not cover anchor {anchor}; latest forecast point is {s.index[-1]}"
        )
    return sliced
