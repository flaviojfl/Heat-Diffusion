"""
Convergence tests for heat diffusion solvers.

Verifies that each solver achieves its expected order of accuracy
by refining the spatial grid and measuring the L2 error against
the analytical solution.

Expected convergence rates:
- FTCS:           O(Δt)  + O(Δx²)  → 1st order in time
- Crank-Nicolson: O(Δt²) + O(Δx²)  → 2nd order in time and space
- θ-method:       same as CN for θ=0.5
"""

from __future__ import annotations

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from heat_diffusion.config import Dimension, SimulationConfig, SolverType
from heat_diffusion.solvers import make_solver
from heat_diffusion.utils.analytical import analytical_1d_sine, compute_l2_error

ALPHA = 1.0e-3
LX = 1.0
T_END = 0.05  # Short run for speed


def ic_sine(x: np.ndarray) -> np.ndarray:
    return np.sin(np.pi * x / LX)


def analytical(x: np.ndarray, t: float) -> np.ndarray:
    return analytical_1d_sine(x, t, ALPHA, LX)


def run_convergence(
    solver_type: SolverType,
    nx_values: list[int],
    t_end: float = T_END,
) -> tuple[list[float], list[float]]:
    """Run solver at multiple resolutions and return (step_sizes, errors)."""
    step_sizes = []
    errors = []

    for nx in nx_values:
        dx = LX / (nx - 1)
        cfg = SimulationConfig(
            alpha=ALPHA,
            L_x=LX,
            nx=nx,
            t_end=t_end,
            solver_type=solver_type,
            dimension=Dimension.ONE_D,
            save_every=1000,  # Only save endpoints
            verbose=False,
        )
        solver = make_solver(cfg, ic_sine, analytical_solution=analytical)
        result = solver.solve()

        x = np.linspace(0, LX, nx)
        u_num = result.final_state
        u_exact = analytical(x, result.times[-1])
        err = compute_l2_error(u_num, u_exact, dx)
        step_sizes.append(dx)
        errors.append(err)

    return step_sizes, errors


def estimate_order(step_sizes: list[float], errors: list[float]) -> float:
    """Estimate convergence order from log-log slope."""
    log_h = np.log(step_sizes)
    log_e = np.log(errors)
    coeffs = np.polyfit(log_h, log_e, 1)
    return float(coeffs[0])


# ──────────────────────────────────────────────────────────────────────
# Spatial convergence: grid refinement
# ──────────────────────────────────────────────────────────────────────

NX_VALUES = [20, 30, 50, 80, 120]


@pytest.mark.slow
def test_spatial_convergence_ftcs() -> None:
    """FTCS should achieve at least O(Δx^1.5) spatial convergence."""
    step_sizes, errors = run_convergence(SolverType.FTCS, NX_VALUES)
    order = estimate_order(step_sizes, errors)
    assert order >= 1.5, f"FTCS spatial order {order:.2f} < 1.5"


@pytest.mark.slow
def test_spatial_convergence_crank_nicolson() -> None:
    """Crank-Nicolson should achieve O(Δx²) spatial convergence."""
    step_sizes, errors = run_convergence(SolverType.CRANK_NICOLSON, NX_VALUES)
    order = estimate_order(step_sizes, errors)
    assert order >= 1.8, (
        f"Crank-Nicolson spatial order {order:.2f} < 1.8 "
        f"(expected ~2.0). errors={errors}"
    )


# ──────────────────────────────────────────────────────────────────────
# Temporal convergence: time step refinement
# ──────────────────────────────────────────────────────────────────────

def run_temporal_convergence(
    solver_type: SolverType,
    dt_values: list[float],
    nx: int = 200,
) -> tuple[list[float], list[float]]:
    """Run at multiple dt values with fine spatial grid, measure time error."""
    errors = []
    x = np.linspace(0, LX, nx)
    dx = LX / (nx - 1)

    for dt in dt_values:
        cfg = SimulationConfig(
            alpha=ALPHA,
            L_x=LX,
            nx=nx,
            t_end=T_END,
            dt=dt,
            solver_type=solver_type,
            dimension=Dimension.ONE_D,
            save_every=10000,
            verbose=False,
        )
        try:
            solver = make_solver(cfg, ic_sine)
            result = solver.solve()
            u_num = result.final_state
            u_exact = analytical(x, result.times[-1])
            err = compute_l2_error(u_num, u_exact, dx)
            errors.append(err)
        except ValueError:
            errors.append(np.nan)

    return dt_values, errors


@pytest.mark.slow
def test_temporal_convergence_crank_nicolson() -> None:
    """Crank-Nicolson should show 2nd-order convergence in time."""
    dt_values = [0.01, 0.005, 0.002, 0.001]
    _, errors = run_temporal_convergence(SolverType.CRANK_NICOLSON, dt_values)
    valid = [(dt, e) for dt, e in zip(dt_values, errors) if not np.isnan(e)]
    assert len(valid) >= 3, "Too few valid data points"
    dts, errs = zip(*valid)
    order = estimate_order(list(dts), list(errs))
    assert order >= 1.8, f"CN temporal order {order:.2f} < 1.8"


# ──────────────────────────────────────────────────────────────────────
# Quick smoke test (always runs, not marked slow)
# ──────────────────────────────────────────────────────────────────────

def test_convergence_smoke() -> None:
    """Quick sanity check: finer grid → smaller error for CN."""
    _, err_coarse = run_convergence(SolverType.CRANK_NICOLSON, [20])
    _, err_fine = run_convergence(SolverType.CRANK_NICOLSON, [60])
    assert err_fine[0] < err_coarse[0], (
        f"Error did not decrease with refinement: "
        f"coarse={err_coarse[0]:.4e}  fine={err_fine[0]:.4e}"
    )
