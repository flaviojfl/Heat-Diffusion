"""
FTCS (Forward-Time Centered-Space) explicit solver.

Implements the explicit Euler scheme for the heat equation:

    u^{n+1}_i = u^n_i + r (u^n_{i-1} - 2u^n_i + u^n_{i+1})

where r = α Δt / Δx².

Stability condition (Von Neumann): r ≤ 1/2 (1D),  r_x + r_y ≤ 1/2 (2D).
"""

from __future__ import annotations

import numpy as np

from ..boundary import BoundaryApplicator
from ..config import Dimension
from .base import BaseSolver


class FTCSSolver(BaseSolver):
    """
    Explicit FTCS solver for the 1D and 2D heat equation.

    Time complexity per step: O(N) in 1D, O(N²) in 2D.
    Stability: Conditionally stable. Requires r ≤ 0.5 in 1D.

    Scheme (1D):
        u^{n+1}_i = u^n_i + r(u^n_{i-1} - 2u^n_i + u^n_{i+1})

    Scheme (2D):
        u^{n+1}_{i,j} = u^n_{i,j}
            + r_x (u^n_{i,j-1} - 2u^n_{i,j} + u^n_{i,j+1})
            + r_y (u^n_{i-1,j} - 2u^n_{i,j} + u^n_{i+1,j})
    """

    @property
    def name(self) -> str:
        return "FTCS (Explicit Euler)"

    def _build_system(self) -> None:
        """Pre-compute diffusion numbers and set up boundary applicator."""
        cfg = self.config
        self.r_x = cfg.alpha * cfg.dt / cfg.dx ** 2
        if cfg.dimension == Dimension.TWO_D:
            self.r_y = cfg.alpha * cfg.dt / cfg.dy ** 2
        self._bc = BoundaryApplicator(cfg.boundary_conditions, cfg.dimension)

    def step(self) -> None:
        """Advance one FTCS time step."""
        if self.config.dimension == Dimension.ONE_D:
            self._step_1d()
        else:
            self._step_2d()

    def _step_1d(self) -> None:
        """FTCS update for the 1D heat equation (vectorised)."""
        u = self.u
        r = self.r_x
        # Interior nodes only — boundaries handled separately
        u[1:-1] = u[1:-1] + r * (u[:-2] - 2.0 * u[1:-1] + u[2:])
        # Apply boundary conditions
        self._bc.apply_1d(u, self.config.dx)

    def _step_2d(self) -> None:
        """FTCS update for the 2D heat equation (vectorised with slicing)."""
        u = self.u
        rx, ry = self.r_x, self.r_y
        # Interior nodes
        u[1:-1, 1:-1] = (
            u[1:-1, 1:-1]
            + rx * (u[1:-1, :-2] - 2.0 * u[1:-1, 1:-1] + u[1:-1, 2:])
            + ry * (u[:-2, 1:-1] - 2.0 * u[1:-1, 1:-1] + u[2:, 1:-1])
        )
        self._bc.apply_2d(u, self.config.dx, self.config.dy)
