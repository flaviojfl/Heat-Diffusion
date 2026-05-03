"""
Crank-Nicolson implicit solver (θ = 0.5).

The Crank-Nicolson scheme is second-order accurate in both space and time:
    O(Δt²) + O(Δx²)

It is unconditionally stable for any r = α Δt / Δx².

1D scheme:
    -r/2 · u^{n+1}_{i-1} + (1+r) · u^{n+1}_i - r/2 · u^{n+1}_{i+1}
        = r/2 · u^n_{i-1} + (1-r) · u^n_i + r/2 · u^n_{i+1}

Leading to a tridiagonal system A · u^{n+1} = b, solved with
scipy.sparse.linalg.spsolve for efficiency.

2D scheme (ADI - Alternating Direction Implicit):
    Peaceman-Rachford splitting for the 2D case, which decouples the
    2D problem into two sequential 1D solves per time step, maintaining
    second-order accuracy while keeping the cost to O(N²).
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.typing import NDArray

from ..boundary import BoundaryApplicator
from ..config import Dimension
from .base import BaseSolver

logger = logging.getLogger(__name__)

ArrayFloat = NDArray[np.float64]


class CrankNicolsonSolver(BaseSolver):
    """
    Crank-Nicolson solver using sparse tridiagonal (1D) or ADI (2D) methods.

    Attributes
    ----------
    A_lhs : scipy.sparse.csr_matrix
        Left-hand-side sparse matrix (implicit part).
    A_rhs : scipy.sparse.csr_matrix
        Right-hand-side sparse matrix (explicit part, for forming b).
    """

    @property
    def name(self) -> str:
        return "Crank-Nicolson (Implicit, O(Δt²))"

    def _build_system(self) -> None:
        """Construct LHS and RHS sparse matrices."""
        cfg = self.config
        self.r_x = cfg.alpha * cfg.dt / cfg.dx ** 2
        self._bc = BoundaryApplicator(cfg.boundary_conditions, cfg.dimension)

        if cfg.dimension == Dimension.ONE_D:
            self.A_lhs, self.A_rhs = self._build_tridiagonal_1d(self.r_x, cfg.nx)
            self._factored = spla.factorized(self.A_lhs.tocsc())
        else:
            self.r_y = cfg.alpha * cfg.dt / cfg.dy ** 2
            # Build separate x and y tridiagonal operators for ADI
            self.Ax_lhs, self.Ax_rhs = self._build_tridiagonal_1d(self.r_x, cfg.nx)
            self.Ay_lhs, self.Ay_rhs = self._build_tridiagonal_1d(self.r_y, cfg.ny)
            self._factor_x = spla.factorized(self.Ax_lhs.tocsc())
            self._factor_y = spla.factorized(self.Ay_lhs.tocsc())
        logger.debug("Sparse system built for %s.", self.name)

    @staticmethod
    def _build_tridiagonal_1d(
        r: float, n: int
    ) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
        """
        Build 1D Crank-Nicolson tridiagonal matrices.

        LHS:  A = tridiag(-r/2,  1+r, -r/2)
        RHS:  B = tridiag( r/2, 1-r,  r/2)

        Boundary rows are identity (Dirichlet assumed; BCs applied afterward).

        Parameters
        ----------
        r : float
            Diffusion number α Δt / Δx².
        n : int
            Number of grid points.

        Returns
        -------
        Tuple[sp.csr_matrix, sp.csr_matrix]
            (A_lhs, A_rhs) sparse matrices.
        """
        diag_lhs = np.full(n, 1.0 + r)
        diag_rhs = np.full(n, 1.0 - r)
        off_lhs = np.full(n - 1, -r / 2.0)
        off_rhs = np.full(n - 1,  r / 2.0)

        # Boundary rows = identity
        diag_lhs[0] = diag_lhs[-1] = 1.0
        diag_rhs[0] = diag_rhs[-1] = 1.0
        off_lhs[0] = off_lhs[-1] = 0.0
        off_rhs[0] = off_rhs[-1] = 0.0

        A_lhs = sp.diags([off_lhs, diag_lhs, off_lhs], [-1, 0, 1], format="csr")
        A_rhs = sp.diags([off_rhs, diag_rhs, off_rhs], [-1, 0, 1], format="csr")
        return A_lhs, A_rhs

    def step(self) -> None:
        """Advance one Crank-Nicolson time step."""
        if self.config.dimension == Dimension.ONE_D:
            self._step_1d()
        else:
            self._step_2d_adi()

    def _step_1d(self) -> None:
        """Solve the tridiagonal system for the 1D case."""
        b = self.A_rhs.dot(self.u)
        # Enforce Dirichlet BCs in the RHS vector directly
        b[0] = self.config.boundary_conditions.left.value
        b[-1] = self.config.boundary_conditions.right.value
        self.u[:] = self._factored(b)
        self._bc.apply_1d(self.u, self.config.dx)

    def _step_2d_adi(self) -> None:
        """
        Peaceman-Rachford ADI step for 2D Crank-Nicolson.

        Step 1 (x-sweep): implicit in x, explicit in y  → u*
        Step 2 (y-sweep): implicit in y, explicit in x  → u^{n+1}
        """
        u = self.u
        ny, nx = self.config.ny, self.config.nx
        ry = self.r_y
        u_star = np.empty_like(u)

        # --- X-sweep: rows (fix j, solve in i) ---
        for j in range(ny):
            # Build RHS: explicit in y at row j
            rhs = self.Ax_rhs.dot(u[j, :])
            if j > 0:
                rhs[1:-1] += (ry / 2.0) * u[j - 1, 1:-1]
            if j < ny - 1:
                rhs[1:-1] += (ry / 2.0) * u[j + 1, 1:-1]
            rhs[1:-1] -= ry * u[j, 1:-1]
            # Apply boundary values
            rhs[0] = self.config.boundary_conditions.left.value
            rhs[-1] = self.config.boundary_conditions.right.value
            u_star[j, :] = self._factor_x(rhs)

        # --- Y-sweep: columns (fix i, solve in j) ---
        rx = self.r_x
        for i in range(nx):
            rhs = self.Ay_rhs.dot(u_star[:, i])
            if i > 0:
                rhs[1:-1] += (rx / 2.0) * u_star[1:-1, i - 1]
            if i < nx - 1:
                rhs[1:-1] += (rx / 2.0) * u_star[1:-1, i + 1]
            rhs[1:-1] -= rx * u_star[1:-1, i]
            rhs[0] = self.config.boundary_conditions.bottom.value
            rhs[-1] = self.config.boundary_conditions.top.value
            u[:, i] = self._factor_y(rhs)

        self._bc.apply_2d(self.u, self.config.dx, self.config.dy)
