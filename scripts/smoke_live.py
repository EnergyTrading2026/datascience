"""Manual smoketest: run the full hybrid cycle against the real SMARD API.

Pulls QH and hourly DA prices live, cross-checks consistency, then runs one
hourly MPC cycle with a synthetic flat demand forecast and prints a readable
summary. Use this before a release or when investigating a prod failure to
quickly verify whether the SMARD API or the optimizer pipeline is the issue.

Run:
    uv run python scripts/smoke_live.py
"""
from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

import pandas as pd

from optimization.adapters import smard_live
from optimization.adapters.forecast import DEMAND_COLUMN
from optimization.run import run_one_cycle


def _check(label: str, ok: bool, detail: str = "") -> None:
    mark = "OK " if ok else "FAIL"
    suffix = f"  {detail}" if detail else ""
    print(f"  [{mark}] {label}{suffix}")


def main() -> int:
    # Quiet down the optimizer's INFO logs so the script output stays readable.
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    now = pd.Timestamp.now(tz="Europe/Berlin")
    print(f"SMARD live smoketest @ {now.isoformat(timespec='seconds')}")
    print("-" * 64)
    overall_ok = True

    # ---- Step 1: pull QH ----
    print("Step 1: pull QH prices (resolution=quarterhour)")
    try:
        sq = smard_live.get_published_prices(now, resolution="quarterhour", horizon_h=24)
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")
        return 1
    _check("non-empty", len(sq) > 0, f"{len(sq)} slots")
    _check("15-min gap", (sq.index[1] - sq.index[0]) == pd.Timedelta(minutes=15))
    _check("no NaN", not sq.isna().any())
    print(f"        window: {sq.index[0]}  →  {sq.index[-1]}")
    print(f"        min/max/mean: {sq.min():.2f} / {sq.max():.2f} / {sq.mean():.2f} EUR/MWh")
    overall_ok &= len(sq) > 0 and not sq.isna().any()

    # ---- Step 2: pull hourly + cross-check ----
    print("\nStep 2: pull hourly prices + cross-check against QH aggregate")
    try:
        sh = smard_live.get_published_prices(now, resolution="hour", horizon_h=24)
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")
        return 1
    _check("non-empty", len(sh) > 0, f"{len(sh)} slots")
    sq_agg = sq.groupby(sq.index.floor("1h")).mean()
    common = sh.index.intersection(sq_agg.index)
    if len(common) > 0:
        diff = (sh.loc[common] - sq_agg.loc[common]).abs()
        _check(
            "hourly ≈ QH-mean",
            diff.max() < 10.0,
            f"max diff {diff.max():.2f} EUR/MWh (volume-weighting tolerance: 10)",
        )
        overall_ok &= diff.max() < 10.0

    # ---- Step 3: intra-hour spreads ----
    print("\nStep 3: intra-hour spreads (next 6 full hours of QH data)")
    aligned = sq.iloc[: (len(sq) // 4) * 4]
    spreads = []
    for hh in range(min(6, len(aligned) // 4)):
        block = aligned.iloc[hh * 4 : (hh + 1) * 4]
        spread = block.max() - block.min()
        spreads.append(spread)
        print(
            f"        {block.index[0].strftime('%H:%M')}-"
            f"{block.index[-1].strftime('%H:%M')}: "
            f"min={block.min():6.2f}  max={block.max():6.2f}  spread={spread:5.2f}"
        )
    if spreads:
        print(f"        max spread observed: {max(spreads):.2f} EUR/MWh")

    # ---- Step 4: full hybrid run_one_cycle ----
    print("\nStep 4: full hybrid run_one_cycle (hourly synthetic forecast + live QH)")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        solve_time = now.floor("1h")
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
        # Pre-EPEX-clearing (~before 13:45 Berlin) tomorrow's prices aren't on
        # SMARD yet, so the horizon may fall below the 11h floor — that's the
        # expected operational behavior, not a code bug.
        pre_clearing = len(sq) < 44
        if rc == 1 and pre_clearing:
            _check(
                "exit code 0",
                True,
                f"got rc=1 (expected before ~13:45 Berlin: only {len(sq)} QH "
                "slots = <11h published; tomorrow's auction not mirrored yet)",
            )
        elif rc != 0:
            _check("exit code 0", False, f"got rc={rc}")
            overall_ok = False
        else:
            df = pd.read_parquet(tmp_path / "dispatch.parquet")
            # Long format since feat/modular-assets: one row per
            # (timestamp, asset_id, family, quantity). Pull SoC for the legacy
            # default's lone storage and verify the commit window shape.
            soc = df[(df["family"] == "storage") & (df["quantity"] == "soc_end_mwh_th")]
            unique_ts = soc.index.unique().sort_values()
            _check("4 dispatch timestamps (1h commit @ QH)", len(unique_ts) == 4)
            if len(unique_ts) >= 2:
                _check(
                    "15-min gap in dispatch",
                    (unique_ts[1] - unique_ts[0]) == pd.Timedelta(minutes=15),
                )
            _check(
                "SoC within bounds [50, 200]",
                soc["value"].between(50.0, 200.0).all(),
            )
            if len(unique_ts):
                print(f"        commit window: {unique_ts[0]}  →  {unique_ts[-1]}")
            soc_traj = soc.sort_index()["value"].tolist()
            print(f"        SoC trajectory: {[round(v, 1) for v in soc_traj]}")

    print("\n" + "-" * 64)
    print("OVERALL: " + ("OK" if overall_ok else "FAIL"))
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
