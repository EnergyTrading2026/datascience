"""Demand forecast adapter — interface to the forecasting team's output.

V1 placeholder schema (REVIEW with forecasting team before relying on this in prod):
  - File format: parquet, with tz-aware DatetimeIndex
  - Index gap: 1h ('hour' resolution) or 15min ('quarterhour' resolution)
  - Index: tz-aware Europe/Berlin
  - Column: 'demand_mw_th' (float, MW thermal)
  - Length: at least the requested horizon, starting at the next slot boundary after at_time
  - No NaN

Validation is strict: schema violations raise loudly rather than feeding
garbage to the optimizer.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

BERLIN = "Europe/Berlin"
DEMAND_COLUMN = "demand_mw_th"

Resolution = Literal["hour", "quarterhour"]
_FLOOR_FREQ: dict[str, str] = {"hour": "1h", "quarterhour": "15min"}
_EXPECTED_GAP: dict[str, pd.Timedelta] = {
    "hour": pd.Timedelta(hours=1),
    "quarterhour": pd.Timedelta(minutes=15),
}


class ForecastSchemaError(ValueError):
    """Forecast file violates the agreed schema."""


def load_forecast(
    path: Path,
    at_time: pd.Timestamp,
    resolution: Resolution = "hour",
) -> pd.Series:
    """Load + validate demand forecast from forecasting team.

    Args:
        path: file path to forecast artifact (.parquet).
        at_time: tz-aware Berlin timestamp (current solve time). Forecast must
            cover the next slot boundary >= at_time onward.
        resolution: 'hour' (1h gap) or 'quarterhour' (15min gap). Must match
            the resolution at which the SMARD adapter and the model are run.

    Returns:
        pd.Series[float], tz-aware Europe/Berlin index, name='demand_mw_th'.
        Index starts at the next slot boundary >= at_time.

    Raises:
        FileNotFoundError, ForecastSchemaError, ValueError.
    """
    if at_time.tzinfo is None:
        raise ValueError(f"at_time must be tz-aware; got {at_time!r}")
    if resolution not in _EXPECTED_GAP:
        raise ValueError(f"resolution must be 'hour' or 'quarterhour'; got {resolution!r}")
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

    expected_gap = _EXPECTED_GAP[resolution]
    if len(s) >= 2:
        gaps = s.index.to_series().diff().dropna()
        if not (gaps == expected_gap).all():
            raise ForecastSchemaError(
                f"forecast index is not {resolution}-uniform "
                f"(expected gap {expected_gap})"
            )

    if s.isna().any():
        raise ForecastSchemaError("forecast contains NaN")

    freq = _FLOOR_FREQ[resolution]
    at_berlin = at_time.tz_convert(BERLIN)
    floored = at_berlin.floor(freq)
    anchor = floored if floored == at_berlin else at_berlin.ceil(freq)

    sliced = s.loc[anchor:]
    if len(sliced) == 0:
        raise ForecastSchemaError(
            f"forecast does not cover anchor {anchor}; latest forecast point is {s.index[-1]}"
        )
    return sliced
