"""Thin HiGHS wrapper. Fails loudly on infeasibility — never silently fall back."""
from __future__ import annotations

import time
from dataclasses import dataclass

import pyomo.environ as pyo


@dataclass
class SolveResult:
    feasible: bool
    status: str
    objective_eur: float | None
    solve_time_s: float


class SolverInfeasibleError(RuntimeError):
    """Raised when HiGHS reports infeasible/unknown — caller must handle."""


_OK = (
    pyo.TerminationCondition.optimal,
    pyo.TerminationCondition.feasible,
)


def solve(
    model: pyo.ConcreteModel,
    time_limit_s: int = 30,
    mip_gap: float = 0.005,
) -> SolveResult:
    """Solve `model` with HiGHS via Pyomo. Raises SolverInfeasibleError on infeasibility.

    Use `mip_gap=0.005` (0.5%) by default — matches the eval-validated setting.
    """
    solver = pyo.SolverFactory("appsi_highs")
    solver.options["time_limit"] = time_limit_s
    solver.options["mip_rel_gap"] = mip_gap

    t0 = time.time()
    res = solver.solve(model, tee=False)
    elapsed = time.time() - t0

    cond = res.solver.termination_condition
    status_str = str(cond)
    if cond not in _OK:
        raise SolverInfeasibleError(f"HiGHS terminated with status={status_str}")

    return SolveResult(
        feasible=True,
        status=status_str,
        objective_eur=float(pyo.value(model.obj)),
        solve_time_s=elapsed,
    )
