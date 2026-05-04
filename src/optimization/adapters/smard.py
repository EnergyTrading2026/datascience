"""SMARD DA-price adapter.

Day-ahead auction publishes once per day (~12:45 CET) for the next calendar day.
Forward visibility from solve time t:
  - t < ~13:00 Berlin -> prices available up to today 24:00      (1-12h ahead)
  - t >= ~13:00 Berlin -> prices available up to tomorrow 24:00  (11-35h ahead)

Caller (run.py) reconciles available DA horizon with the demand-forecast horizon.

Used for backtest / replay against historical SMARD CSV exports. The
production fetcher is `smard_live.py`, which pulls live from the SMARD API.

Reference: notebooks/optimization/smard_live_prices.ipynb (validation),
mpc_prototype.ipynb cell 5 (parser).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

BERLIN = "Europe/Berlin"
PRICE_COLUMN = "Deutschland/Luxemburg [€/MWh] Berechnete Auflösungen"


class InsufficientPriceHorizonError(RuntimeError):
    """Raised when DA prices end before the requested horizon."""


def _load_csv(path: Path) -> pd.Series:
    """Parse SMARD CSV (semicolon-separated, comma-decimal, Berlin local) into hourly Series."""
    df = pd.read_csv(path, sep=";", decimal=",", low_memory=False)
    ts = pd.to_datetime(df["Datum von"], format="%d.%m.%Y %H:%M").dt.tz_localize(
        BERLIN, ambiguous="infer"
    )
    s = pd.Series(
        pd.to_numeric(df[PRICE_COLUMN], errors="coerce").values,
        index=ts,
        name="price_eur_mwh",
    )
    s = s.groupby(s.index).first().sort_index().asfreq("1h")
    s.index.name = "timestamp_berlin"
    return s


def get_published_prices(at_time: pd.Timestamp, csv_path: Path) -> pd.Series:
    """Return hourly DA prices available at `at_time`, anchored at the next full hour.

    Args:
        at_time: tz-aware Berlin timestamp (current solve time). The first index
            value of the returned series is the smallest hour boundary >= at_time.
        csv_path: path to SMARD-format CSV file (semicolon-separated,
            comma-decimal, Berlin local). For backtest only — point to a
            downloaded SMARD export.

    Returns:
        pd.Series[float], hourly, tz-aware Europe/Berlin index, name='price_eur_mwh'.
        Length depends on what's published in the CSV at the time of read.
        NaN values are dropped — caller checks resulting length vs. needed horizon.

    Raises:
        FileNotFoundError: if csv_path does not exist.
        ValueError: if at_time is not tz-aware.
    """
    if at_time.tzinfo is None:
        raise ValueError(f"at_time must be tz-aware; got {at_time!r}")
    series = _load_csv(Path(csv_path))
    at_berlin = at_time.tz_convert(BERLIN)
    # Anchor: smallest full hour >= at_berlin.
    if at_berlin.minute == 0 and at_berlin.second == 0 and at_berlin.microsecond == 0:
        anchor = at_berlin
    else:
        anchor = at_berlin.ceil("1h")
    return series.loc[anchor:].dropna()
