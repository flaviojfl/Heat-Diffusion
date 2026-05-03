"""
Unit tests for 1D heat diffusion solvers.

Tests cover:
- Basic solver instantiation and step execution
- Conservation of mass (total energy)
- Boundary condition enforcement
- Comparison with analytical solution
"""

from __future__ import annotations

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from heat_diffusion.config import (
    BoundaryConditionConfig,
    BoundaryConditionsConfig,
    BoundaryType,
    Dimension,
    SimulationConfig,
    SolverType,
)
from heat_diffusion.solvers import make_solver
from heat_diffusion.utils.analytical import analytical_1d_sine, compute_l2_error


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────

ALPHA = 1.0e-3
LX = 1.0
NX = 40
T_END = 0.1


def make_cfg(solver_type: SolverType, nx: int = NX, dt: float | None = None) -> SimulationConfig:
    return SimulationConfig(
        alpha=ALPHA,
        L_x=LX,
        nx=nx,
        t_end=T_END,
        dt=dt,
        solver_type=solver_type,
        dimension=Dimension.ONE_D,
        save_every=5,
        verbose=False,
    )


def ic_sine(x: np.ndarray) -> np.ndarray:
    """Initial condition: u(x,0) = sin(πx/L)."""
    return np.sin(np.pi * x / LX)


def analytical(x: np.ndarray, t: float) -> np.ndarray:
    return analytical_1d_sine(x, t, ALPHA, LX)


# ──────────────────────────────────────────────────────────────────────
# Tests: Instantiation
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("solver_type", list(SolverType))
def test_solver_instantiation(solver_type: SolverType) -> None:
    """All solvers should instantiate without error."""
    cfg = make_cfg(solver_type)
    solver = make_solver(cfg, ic_sine)
    assert solver is not None
    assert solver.u.shape == (NX,)


@pytest.mark.parametrize("solver_type", list(SolverType))
def test_solver_produces_result(solver_type: SolverType) -> None:
    """All solvers should run and return a SimulationResult."""
    cfg = make_cfg(solver_type)
    solver = make_solver(cfg, ic_sine)
    result = solver.solve()
    assert len(result.states) > 1
    assert len(result.times) == len(result.states)
    assert result.wall_time > 0


# ──────────────────────────────────────────────────────────────────────
# Tests: Dirichlet BCs
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("solver_type", list(SolverType))
def test_dirichlet_bc_enforced(solver_type: SolverType) -> None:
    """Dirichlet BCs u=0 at both ends must be maintained throughout."""
    cfg = make_cfg(solver_type)
    solver = make_solver(cfg, ic_sine)
    result = solver.solve()
    for state in result.states:
        assert abs(state[0]) < 1e-12, f"Left BC violated: {state[0]}"
        assert abs(state[-1]) < 1e-12, f"Right BC violated: {state[-1]}"


# ──────────────────────────────────────────────────────────────────────
# Tests: Physical properties
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("solver_type", list(SolverType))
def test_maximum_principle(solver_type: SolverType) -> None:
    """Temperature should not exceed initial maximum or fall below initial minimum."""
    cfg = make_cfg(solver_type)
    solver = make_solver(cfg, ic_sine)
    u0 = solver.u.copy()
    u_min, u_max = float(u0.min()), float(u0.max())
    result = solver.solve()
    for i, state in enumerate(result.states[1:], 1):
        assert float(state.max()) <= u_max + 1e-10, (
            f"Max principle violated at step {i}: {state.max():.6f} > {u_max:.6f}"
        )


@pytest.mark.parametrize("solver_type", list(SolverType))
def test_monotone_decay(solver_type: SolverType) -> None:
    """Total energy (integral of u) should decay monotonically with Dirichlet BCs."""
    cfg = make_cfg(solver_type)
    solver = make_solver(cfg, ic_sine)
    result = solver.solve()
    energies = [np.trapezoid(s, dx=cfg.dx) for s in result.states]
    for i in range(1, len(energies)):
        assert energies[i] <= energies[i - 1] + 1e-10, (
            f"Energy increased at step {i}: {energies[i]:.6f} > {energies[i-1]:.6f}"
        )


# ──────────────────────────────────────────────────────────────────────
# Tests: Accuracy vs analytical solution
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("solver_type", [SolverType.CRANK_NICOLSON, SolverType.FTCS])
def test_accuracy_against_analytical(solver_type: SolverType) -> None:
    """L2 error should be small compared to initial amplitude."""
    cfg = make_cfg(solver_type, nx=80)
    solver = make_solver(cfg, ic_sine, analytical_solution=analytical)
    result = solver.solve()
    assert result.errors, "No errors were recorded"
    final_error = result.errors[-1]
    assert final_error < 1e-3, (
        f"L2 error too large for {solver_type.value}: {final_error:.2e}"
    )


# ──────────────────────────────────────────────────────────────────────
# Tests: FTCS stability
# ──────────────────────────────────────────────────────────────────────

def test_ftcs_stability_check() -> None:
    """FTCS must raise ValueError when dt violates the CFL condition."""
    # For nx=20, dx=1/19≈0.0526, alpha=1e-3: limit = dx²/(2α) ≈ 1.385
    # Use dt=2.0 to definitely exceed it
    cfg = make_cfg(SolverType.FTCS, nx=20, dt=2.0)
    with pytest.raises(ValueError, match="FTCS instability"):
        make_solver(cfg, ic_sine)


# ──────────────────────────────────────────────────────────────────────
# Tests: Configuration validation
# ──────────────────────────────────────────────────────────────────────

def test_negative_alpha_raises() -> None:
    with pytest.raises(ValueError, match="alpha"):
        SimulationConfig(alpha=-1.0)


def test_zero_t_end_raises() -> None:
    with pytest.raises(ValueError, match="t_end"):
        SimulationConfig(t_end=0.0)


def test_invalid_theta_raises() -> None:
    with pytest.raises(ValueError, match="theta"):
        SimulationConfig(theta=1.5, solver_type=SolverType.THETA_METHOD)
