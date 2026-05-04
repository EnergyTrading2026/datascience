"""Live integration tests against the real SMARD chart_data API.

Skipped by default (`-m 'not live'` in pytest config). Run explicitly with:

    uv run pytest tests/optimization/test_live_smard_integration.py -m live

These tests exercise the full data path against production SMARD endpoints.
They detect regressions caused by upstream API changes (new schema, renamed
filters, broken endpoints, changed JSON structure) that mocked unit tests
cannot catch.

Assertions are intentionally loose on values (live prices change daily) and
strict on structure (tz-awareness, gap, no-NaN, sensible bounds).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from optimization.adapters import smard_live
from optimization.adapters.forecast import DEMAND_COLUMN
from optimization.run import run_one_cycle


# Sanity bounds for German DA prices (€/MWh). Historical extremes during the
# 2022 gas crisis hit ~700 €/MWh; pre-crisis floor includes negative midday
# values down to ~-100. Anything outside [-500, 1500] would be a data error.
_PRICE_LOWER = -500.0
_PRICE_UPPER = 1500.0


@pytest.mark.live
def test_live_qh_returns_sensible_structure():
    """smard_live with quarterhour: tz-aware, 15-min gap, no NaN, sensible values."""
    now = pd.Timestamp.now(tz="Europe/Berlin")
    s = smard_live.get_published_prices(now, resolution="quarterhour", horizon_h=12)

    assert isinstance(s, pd.Series)
    assert s.name == "price_eur_mwh"
    assert s.index.tz is not None
    assert str(s.index.tz) == "Europe/Berlin"
    assert (s.index[1] - s.index[0]) == pd.Timedelta(minutes=15)
    assert not s.isna().any()
    # At least 3h of QH data should always be available regardless of EPEX schedule
    assert len(s) >= 12, f"expected >=12 QH slots, got {len(s)}"
    assert _PRICE_LOWER < s.min(), f"min price {s.min()} below sanity floor"
    assert s.max() < _PRICE_UPPER, f"max price {s.max()} above sanity ceiling"
    # Real DA prices are not constant over 12 slots
    assert s.std() > 0.1, "QH price series suspiciously constant"


@pytest.mark.live
def test_live_hour_returns_sensible_structure():
    """smard_live with hour: same checks, hourly gap."""
    now = pd.Timestamp.now(tz="Europe/Berlin")
    s = smard_live.get_published_prices(now, resolution="hour", horizon_h=12)

    assert (s.index[1] - s.index[0]) == pd.Timedelta(hours=1)
    assert not s.isna().any()
    assert len(s) >= 3
    assert _PRICE_LOWER < s.min() and s.max() < _PRICE_UPPER


@pytest.mark.live
def test_live_qh_aggregates_consistently_with_hourly():
    """Cross-check: arithmetic mean of 4 QH slots ≈ SMARD's published hourly value.

    The hourly value is volume-weighted, not arithmetic-averaged, so small
    deviations are expected. Tolerance of 10 €/MWh is loose enough for normal
    weighting differences but tight enough to catch a structural mismatch
    (e.g. if the API ever returned a different price product under the same filter).

    Filter out partially-published hours (last hour of QH stream may have <4
    slots if EPEX is mid-publication) — those would falsely diverge.
    """
    now = pd.Timestamp.now(tz="Europe/Berlin")
    sh = smard_live.get_published_prices(now, resolution="hour", horizon_h=6)
    sq = smard_live.get_published_prices(now, resolution="quarterhour", horizon_h=6)

    counts = sq.groupby(sq.index.floor("1h")).count()
    full_hours = counts[counts == 4].index
    if len(full_hours) < 3:
        pytest.skip(
            "not enough fully-published QH hours to cross-check "
            f"(got {len(full_hours)} full hours)"
        )
    sq_aggregated = sq.groupby(sq.index.floor("1h")).mean().loc[full_hours]
    common = sh.index.intersection(sq_aggregated.index)
    assert len(common) >= 3, "not enough overlap to cross-check"
    diff = (sh.loc[common] - sq_aggregated.loc[common]).abs()
    assert diff.max() < 10.0, (
        f"hourly vs QH-mean diverge by {diff.max():.2f} €/MWh — "
        "structural mismatch suspected"
    )


@pytest.mark.live
def test_live_full_hybrid_cycle(tmp_path: Path):
    """End-to-end: hourly synthetic forecast + real QH SMARD prices -> dispatch parquet."""
    solve_time = pd.Timestamp.now(tz="Europe/Berlin").floor("1h")
    idx = pd.date_range(solve_time, periods=35, freq="1h", tz="Europe/Berlin")
    forecast_path = tmp_path / "forecast.parquet"
    pd.DataFrame({DEMAND_COLUMN: [10.0] * 35}, index=idx).to_parquet(forecast_path)

    rc = run_one_cycle(
        solve_time=solve_time,
        forecast_path=forecast_path,
        prices_path=None,
        state_in=None,
        state_out=tmp_path / "state.json",
        dispatch_out=tmp_path / "dispatch.parquet",
        cold_start=True,
    )
    if rc == 1:
        pytest.skip(
            "run_one_cycle returned exit 1 (likely horizon too short before "
            "EPEX clears for next day). Re-run after ~13:45 Berlin local."
        )
    assert rc == 0

    df = pd.read_parquet(tmp_path / "dispatch.parquet")
    assert len(df) == 4  # 1h commit at QH model = 4 slots
    assert (df.index[1] - df.index[0]) == pd.Timedelta(minutes=15)
    expected_cols = {
        "hp_p_el_mw", "boiler_q_th_mw", "chp_p_el_mw",
        "sto_charge_mw_th", "sto_discharge_mw_th", "soc_end_mwh_th",
    }
    assert expected_cols <= set(df.columns)
    # SoC must respect the storage floor (50) and capacity (200)
    assert df["soc_end_mwh_th"].between(50.0 - 1e-6, 200.0 + 1e-6).all()
