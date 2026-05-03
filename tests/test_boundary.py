"""
Unit tests for boundary condition implementations.
"""

from __future__ import annotations

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from heat_diffusion.boundary import (
    BoundaryApplicator,
    DirichletBC,
    NeumannBC,
    RobinBC,
    make_bc,
)
from heat_diffusion.config import (
    BoundaryConditionConfig,
    BoundaryConditionsConfig,
    BoundaryType,
    Dimension,
)


# ──────────────────────────────────────────────────────────────────────
# DirichletBC
# ──────────────────────────────────────────────────────────────────────

def test_dirichlet_sets_value() -> None:
    cfg = BoundaryConditionConfig(BoundaryType.DIRICHLET, value=42.0)
    bc = DirichletBC(cfg)
    arr = np.array([5.0])
    bc.apply(arr)
    assert arr[0] == pytest.approx(42.0)


def test_dirichlet_zero() -> None:
    cfg = BoundaryConditionConfig(BoundaryType.DIRICHLET, value=0.0)
    bc = DirichletBC(cfg)
    arr = np.array([99.0])
    bc.apply(arr)
    assert arr[0] == pytest.approx(0.0)


# ──────────────────────────────────────────────────────────────────────
# NeumannBC
# ──────────────────────────────────────────────────────────────────────

def test_neumann_zero_flux_flat_profile() -> None:
    """Zero Neumann BC on a flat profile should not change the boundary."""
    cfg = BoundaryConditionConfig(BoundaryType.NEUMANN, value=0.0)
    bc = NeumannBC(cfg)
    # Flat profile: all equal values
    u = np.full(10, 5.0)
    u_copy = u.copy()
    bc.apply(u, dx=0.1, side="left")
    assert u[0] == pytest.approx(u_copy[0], abs=1e-10)


def test_neumann_applied_left_and_right() -> None:
    """Neumann BC should modify only the targeted boundary node."""
    cfg = BoundaryConditionConfig(BoundaryType.NEUMANN, value=0.0)
    bc = NeumannBC(cfg)
    u = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    original_interior = u[1:-1].copy()
    bc.apply(u, dx=0.25, side="left")
    # Interior should be untouched
    np.testing.assert_array_equal(u[1:-1], original_interior)


# ──────────────────────────────────────────────────────────────────────
# RobinBC
# ──────────────────────────────────────────────────────────────────────

def test_robin_degenerates_to_dirichlet() -> None:
    """Robin with beta=0 reduces to Dirichlet: alpha*u = g → u = g/alpha."""
    g = 10.0
    a = 2.0
    cfg = BoundaryConditionConfig(BoundaryType.ROBIN, value=g, alpha=a, beta=0.0)
    bc = RobinBC(cfg)
    u = np.array([5.0, 4.0, 3.0])
    bc.apply(u, dx=0.5, side="left")
    assert u[0] == pytest.approx(g / a, rel=1e-6)


def test_robin_singular_raises() -> None:
    """Robin BC with degenerate coefficients should raise ValueError."""
    cfg = BoundaryConditionConfig(BoundaryType.ROBIN, value=1.0, alpha=1.0, beta=0.5)
    bc = RobinBC(cfg)
    u = np.array([1.0, 2.0, 3.0])
    # Make denominator zero: alpha*dx - beta = 0 → dx = beta/alpha = 0.5
    with pytest.raises(ValueError, match="denominator"):
        bc.apply(u, dx=0.5, side="left")


# ──────────────────────────────────────────────────────────────────────
# make_bc factory
# ──────────────────────────────────────────────────────────────────────

def test_make_bc_factory_dirichlet() -> None:
    cfg = BoundaryConditionConfig(BoundaryType.DIRICHLET, value=5.0)
    bc = make_bc(cfg)
    assert isinstance(bc, DirichletBC)


def test_make_bc_factory_neumann() -> None:
    cfg = BoundaryConditionConfig(BoundaryType.NEUMANN, value=0.0)
    bc = make_bc(cfg)
    assert isinstance(bc, NeumannBC)


# ──────────────────────────────────────────────────────────────────────
# BoundaryApplicator 1D
# ──────────────────────────────────────────────────────────────────────

def test_boundary_applicator_1d_dirichlet() -> None:
    """BoundaryApplicator should enforce Dirichlet BCs on both ends."""
    bc_cfg = BoundaryConditionsConfig(
        left=BoundaryConditionConfig(BoundaryType.DIRICHLET, value=100.0),
        right=BoundaryConditionConfig(BoundaryType.DIRICHLET, value=0.0),
    )
    applicator = BoundaryApplicator(bc_cfg, Dimension.ONE_D)
    u = np.linspace(50.0, 50.0, 10)
    applicator.apply_1d(u, dx=0.1)
    assert u[0] == pytest.approx(100.0)
    assert u[-1] == pytest.approx(0.0)


# ──────────────────────────────────────────────────────────────────────
# BoundaryApplicator 2D
# ──────────────────────────────────────────────────────────────────────

def test_boundary_applicator_2d_dirichlet() -> None:
    """2D BoundaryApplicator should enforce Dirichlet BCs on all four sides."""
    bc_cfg = BoundaryConditionsConfig(
        left=BoundaryConditionConfig(BoundaryType.DIRICHLET, value=0.0),
        right=BoundaryConditionConfig(BoundaryType.DIRICHLET, value=0.0),
        bottom=BoundaryConditionConfig(BoundaryType.DIRICHLET, value=0.0),
        top=BoundaryConditionConfig(BoundaryType.DIRICHLET, value=0.0),
    )
    applicator = BoundaryApplicator(bc_cfg, Dimension.TWO_D)
    u = np.ones((8, 8), dtype=np.float64)
    applicator.apply_2d(u, dx=0.1, dy=0.1)
    np.testing.assert_array_almost_equal(u[0, :], 0.0)
    np.testing.assert_array_almost_equal(u[-1, :], 0.0)
    np.testing.assert_array_almost_equal(u[:, 0], 0.0)
    np.testing.assert_array_almost_equal(u[:, -1], 0.0)
