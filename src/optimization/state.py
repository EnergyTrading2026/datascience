"""DispatchState — carry-over between hourly solves.

Critical: this is the SAFETY-CRITICAL data structure for hourly MPC. Every hour's
solve depends on the realized state at the end of the previous hour's commit.
Bugs here surface as silently wrong dispatch (units restarting unnecessarily,
storage drifting from reality).
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

# Time-in-state sentinel for "long enough that min-up/down is non-binding".
TIS_LONG = 999


@dataclass
class DispatchState:
    """State at a 15-min interval boundary, carried into the next solve.

    Time-in-state is in 15-min steps (not hours), matching the MILP grid.
    """

    timestamp: pd.Timestamp          # tz-aware Berlin, end of last commit window
    sto_soc_mwh_th: float            # realized storage SoC

    hp_on: int                       # 0 or 1
    boiler_on: int                   # 0 or 1
    boiler_time_in_state_steps: int  # how long boiler has been in current state
    chp_on: int                      # 0 or 1
    chp_time_in_state_steps: int

    @classmethod
    def cold_start(cls, timestamp: pd.Timestamp, sto_soc_mwh_th: float = 200.0) -> "DispatchState":
        """Default state when no prior commit exists (e.g., first deployment)."""
        return cls(
            timestamp=timestamp,
            sto_soc_mwh_th=sto_soc_mwh_th,
            hp_on=0,
            boiler_on=0,
            boiler_time_in_state_steps=TIS_LONG,
            chp_on=0,
            chp_time_in_state_steps=TIS_LONG,
        )

    def save(self, path: Path) -> None:
        """Serialize to JSON via atomic tmp-then-replace.

        Atomic = either the file is fully written or the previous version stays
        untouched. Important because a partial state file would crash the next
        hourly run.
        """
        payload = asdict(self)
        payload["timestamp"] = self.timestamp.isoformat()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> "DispatchState":
        """Deserialize from JSON. Raises FileNotFoundError if missing —
        caller decides whether to fall back to cold_start()."""
        path = Path(path)
        payload = json.loads(path.read_text())
        ts = pd.Timestamp(payload["timestamp"])
        if ts.tzinfo is None:
            raise ValueError(f"State timestamp must be tz-aware, got {ts!r}")
        return cls(
            timestamp=ts,
            sto_soc_mwh_th=float(payload["sto_soc_mwh_th"]),
            hp_on=int(payload["hp_on"]),
            boiler_on=int(payload["boiler_on"]),
            boiler_time_in_state_steps=int(payload["boiler_time_in_state_steps"]),
            chp_on=int(payload["chp_on"]),
            chp_time_in_state_steps=int(payload["chp_time_in_state_steps"]),
        )
