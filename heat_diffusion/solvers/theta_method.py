"""
General θ-method solver.

The θ-method is a one-parameter family of time integration schemes:

    (I - θ Δt L) u^{n+1} = (I + (1-θ) Δt L) u^n

where L is the discrete Laplacian operator.

Special cases:
    θ = 0    → FTCS (fully explicit, 1st order in time)
    θ = 0.5  → Crank-Nicolson (2nd order in time)
    θ = 1    → Fully implicit backward Euler (1st order, maximally diffusive)

All values 0 ≤ θ ≤ 1 are unconditionally stable for θ ≥ 0.5.
For 0 ≤ θ < 0.5, stability requires r ≤ 1/(2(1-2θ)).
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.typing import NDArray

from ..boundary import BoundaryApplicator
from ..config import Dimension, SolverType
from .base import BaseSolver

logger = logging.getLogger(__name__)

ArrayFloat = NDArray[np.float64]


class ThetaMethodSolver(BaseSolver):
    """
    General θ-method solver for 1D and 2D heat equations.

    Parameters
    ----------
    theta : float
        Time-weighting parameter ∈ [0, 1].
        0 → explicit FTCS, 0.5 → Crank-Nicolson, 1 → implicit Euler.
    """

    @property
    def name(self) -> str:
        theta = self.config.theta
        if abs(theta) < 1e-10:
            return "θ-method (θ=0, FTCS)"
        elif abs(theta - 0.5) < 1e-10:
            return "θ-method (θ=0.5, Crank-Nicolson)"
        elif abs(theta - 1.0) < 1e-10:
            return "θ-method (θ=1, Implicit Euler)"
        return f"θ-method (θ={theta:.2f})"

    def _build_system(self) -> None:
        """Build sparse LHS and RHS matrices for the chosen θ."""
        cfg = self.config
        theta = cfg.theta
        self.r_x = cfg.alpha * cfg.dt / cfg.dx ** 2
        self._bc = BoundaryApplicator(cfg.boundary_conditions, cfg.dimension)

        if cfg.dimension == Dimension.ONE_D:
            self.A_lhs, self.A_rhs = self._build_matrices_1d(
                self.r_x, cfg.nx, theta
            )
            self._factored = spla.factorized(self.A_lhs.tocsc())
        else:
            self.r_y = cfg.alpha * cfg.dt / cfg.dy ** 2
            self.Ax_lhs, self.Ax_rhs = self._build_matrices_1d(
                self.r_x, cfg.nx, theta
            )
            self.Ay_lhs, self.Ay_rhs = self._build_matrices_1d(
                self.r_y, cfg.ny, theta
            )
            self._factor_x = spla.factorized(self.Ax_lhs.tocsc())
            self._factor_y = spla.factorized(self.Ay_lhs.tocsc())

        self._check_theta_stability()
        logger.debug("θ-method system built: θ=%.2f  r=%.4f", theta, self.r_x)

    def _check_theta_stability(self) -> None:
        """Check conditional stability for θ < 0.5."""
        theta = self.config.theta
        r = self.r_x
        if theta < 0.5:
            limit = 1.0 / (2.0 * (1.0 - 2.0 * theta)) if theta < 0.5 else np.inf
            if r > limit:
                logger.warning(
                    "θ-method with θ=%.2f may be unstable! "
                    "r=%.4f > stability_limit=%.4f",
                    theta, r, limit,
                )

    @staticmethod
    def _build_matrices_1d(
        r: float, n: int, theta: float
    ) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
        """
        Build 1D tridiagonal matrices for the θ-method.

        LHS: A = I + θ r · L  =  tridiag(-θr, 1+2θr, -θr)
        RHS: B = I - (1-θ) r · L  =  tridiag((1-θ)r, 1-2(1-θ)r, (1-θ)r)

        where L is the discrete Laplacian.

        Parameters
        ----------
        r : float
            Diffusion number α Δt / Δx².
        n : int
            Number of grid points.
        theta : float
            Weighting parameter.

        Returns
        -------
        Tuple[sp.csr_matrix, sp.csr_matrix]
            (A_lhs, A_rhs) sparse matrices.
        """
        # LHS: implicit contribution
        diag_lhs = np.full(n, 1.0 + 2.0 * theta * r)
        off_lhs = np.full(n - 1, -theta * r)

        # RHS: explicit contribution
        diag_rhs = np.full(n, 1.0 - 2.0 * (1.0 - theta) * r)
        off_rhs = np.full(n - 1, (1.0 - theta) * r)

        # Boundary rows = identity (BCs applied separately)
        diag_lhs[0] = diag_lhs[-1] = 1.0
        diag_rhs[0] = diag_rhs[-1] = 1.0
        off_lhs[0] = off_lhs[-1] = 0.0
        off_rhs[0] = off_rhs[-1] = 0.0

        A_lhs = sp.diags([off_lhs, diag_lhs, off_lhs], [-1, 0, 1], format="csr")
        A_rhs = sp.diags([off_rhs, diag_rhs, off_rhs], [-1, 0, 1], format="csr")
        return A_lhs, A_rhs

    def step(self) -> None:
        """Advance one θ-method time step."""
        if self.config.dimension == Dimension.ONE_D:
            self._step_1d()
        else:
            self._step_2d()

    def _step_1d(self) -> None:
        """θ-method step for 1D."""
        theta = self.config.theta
        if abs(theta) < 1e-10:
            # Pure explicit — use FTCS vectorised stencil directly
            r = self.r_x
            self.u[1:-1] += r * (
                self.u[:-2] - 2.0 * self.u[1:-1] + self.u[2:]
            )
        else:
            b = self.A_rhs.dot(self.u)
            b[0] = self.config.boundary_conditions.left.value
            b[-1] = self.config.boundary_conditions.right.value
            self.u[:] = self._factored(b)
        self._bc.apply_1d(self.u, self.config.dx)

    def _step_2d(self) -> None:
        """
        θ-method ADI sweep for 2D (Peaceman-Rachford style).

        This is an operator-splitting approach:
            Half-step: implicit in x, explicit in y
            Half-step: implicit in y, explicit in x
        """
        theta = self.config.theta
        u = self.u
        ny, nx = self.config.ny, self.config.nx
        ry = self.r_y

        u_star = np.empty_like(u)

        # --- X-sweep ---
        for j in range(ny):
            rhs = self.Ax_rhs.dot(u[j, :])
            if j > 0:
                rhs[1:-1] += (1.0 - theta) * ry * u[j - 1, 1:-1]
            if j < ny - 1:
                rhs[1:-1] += (1.0 - theta) * ry * u[j + 1, 1:-1]
            rhs[1:-1] -= 2.0 * (1.0 - theta) * ry * u[j, 1:-1]
            rhs[0] = self.config.boundary_conditions.left.value
            rhs[-1] = self.config.boundary_conditions.right.value
            u_star[j, :] = self._factor_x(rhs)

        # --- Y-sweep ---
        rx = self.r_x
        for i in range(nx):
            rhs = self.Ay_rhs.dot(u_star[:, i])
            if i > 0:
                rhs[1:-1] += (1.0 - theta) * rx * u_star[1:-1, i - 1]
            if i < nx - 1:
                rhs[1:-1] += (1.0 - theta) * rx * u_star[1:-1, i + 1]
            rhs[1:-1] -= 2.0 * (1.0 - theta) * rx * u_star[1:-1, i]
            rhs[0] = self.config.boundary_conditions.bottom.value
            rhs[-1] = self.config.boundary_conditions.top.value
            u[:, i] = self._factor_y(rhs)

        self._bc.apply_2d(self.u, self.config.dx, self.config.dy)
