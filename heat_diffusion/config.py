"""
Central configuration for heat diffusion simulations.

All simulation parameters are defined here using Python dataclasses,
providing type safety, validation, and a single source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class SolverType(str, Enum):
    """Available numerical solver schemes."""

    FTCS = "ftcs"
    CRANK_NICOLSON = "crank_nicolson"
    THETA_METHOD = "theta_method"


class BoundaryType(str, Enum):
    """Supported boundary condition types."""

    DIRICHLET = "dirichlet"   # Fixed value:  u = g
    NEUMANN = "neumann"       # Fixed flux:   du/dn = g
    ROBIN = "robin"           # Mixed:        a*u + b*du/dn = g


class Dimension(int, Enum):
    """Supported spatial dimensions."""

    ONE_D = 1
    TWO_D = 2


@dataclass
class BoundaryConditionConfig:
    """
    Configuration for a single boundary condition.

    Parameters
    ----------
    bc_type : BoundaryType
        Type of boundary condition (Dirichlet, Neumann, Robin).
    value : float
        Prescribed value (g in the boundary equations).
    alpha : float
        Robin coefficient for u term (a in a*u + b*du/dn = g).
        Only used when bc_type is ROBIN.
    beta : float
        Robin coefficient for du/dn term (b in a*u + b*du/dn = g).
        Only used when bc_type is ROBIN.
    """

    bc_type: BoundaryType = BoundaryType.DIRICHLET
    value: float = 0.0
    alpha: float = 1.0
    beta: float = 1.0


@dataclass
class BoundaryConditionsConfig:
    """
    Full boundary conditions configuration for 1D or 2D domains.

    For 1D: left and right boundaries.
    For 2D: left, right, bottom, and top boundaries.
    """

    left: BoundaryConditionConfig = field(
        default_factory=lambda: BoundaryConditionConfig(BoundaryType.DIRICHLET, 0.0)
    )
    right: BoundaryConditionConfig = field(
        default_factory=lambda: BoundaryConditionConfig(BoundaryType.DIRICHLET, 0.0)
    )
    bottom: BoundaryConditionConfig = field(
        default_factory=lambda: BoundaryConditionConfig(BoundaryType.DIRICHLET, 0.0)
    )
    top: BoundaryConditionConfig = field(
        default_factory=lambda: BoundaryConditionConfig(BoundaryType.DIRICHLET, 0.0)
    )


@dataclass
class SimulationConfig:
    """
    Master configuration for a heat diffusion simulation.

    Parameters
    ----------
    alpha : float
        Thermal diffusivity [m²/s]. Must be positive.
    L_x : float
        Domain length in x-direction [m]. Must be positive.
    L_y : float
        Domain length in y-direction [m]. Required for 2D simulations.
    nx : int
        Number of spatial grid points in x-direction (including boundaries).
    ny : int
        Number of spatial grid points in y-direction (including boundaries).
        Required for 2D simulations.
    t_end : float
        Total simulation time [s]. Must be positive.
    dt : float
        Time step size [s]. If None, it is chosen automatically to satisfy
        the Von Neumann stability criterion for the selected solver.
    solver_type : SolverType
        Numerical scheme to use for time integration.
    theta : float
        Weighting parameter for the θ-method (0 ≤ θ ≤ 1).
        θ=0 → FTCS (explicit), θ=0.5 → Crank-Nicolson, θ=1 → Fully implicit.
        Only used when solver_type is THETA_METHOD.
    dimension : Dimension
        Spatial dimension of the simulation (1D or 2D).
    boundary_conditions : BoundaryConditionsConfig
        Boundary condition configuration for all domain boundaries.
    save_every : int
        Save simulation state every N time steps.
    verbose : bool
        If True, print progress and diagnostic information.

    Examples
    --------
    >>> cfg = SimulationConfig(alpha=1e-4, L_x=1.0, nx=100, t_end=0.1)
    >>> cfg.dx
    0.010101010101010102
    """

    alpha: float = 1.0e-4
    L_x: float = 1.0
    L_y: float = 1.0
    nx: int = 50
    ny: int = 50
    t_end: float = 0.5
    dt: Optional[float] = None
    solver_type: SolverType = SolverType.CRANK_NICOLSON
    theta: float = 0.5
    dimension: Dimension = Dimension.ONE_D
    boundary_conditions: BoundaryConditionsConfig = field(
        default_factory=BoundaryConditionsConfig
    )
    save_every: int = 10
    verbose: bool = True

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate configuration parameters."""
        if self.alpha <= 0:
            raise ValueError(f"Thermal diffusivity alpha must be positive, got {self.alpha}")
        if self.L_x <= 0:
            raise ValueError(f"Domain length L_x must be positive, got {self.L_x}")
        if self.L_y <= 0:
            raise ValueError(f"Domain length L_y must be positive, got {self.L_y}")
        if self.nx < 3:
            raise ValueError(f"nx must be at least 3, got {self.nx}")
        if self.ny < 3:
            raise ValueError(f"ny must be at least 3, got {self.ny}")
        if self.t_end <= 0:
            raise ValueError(f"t_end must be positive, got {self.t_end}")
        if self.dt is not None and self.dt <= 0:
            raise ValueError(f"dt must be positive when specified, got {self.dt}")
        if not (0.0 <= self.theta <= 1.0):
            raise ValueError(f"theta must be in [0, 1], got {self.theta}")

    @property
    def dx(self) -> float:
        """Spatial step size in x-direction [m]."""
        return self.L_x / (self.nx - 1)

    @property
    def dy(self) -> float:
        """Spatial step size in y-direction [m]."""
        return self.L_y / (self.ny - 1)

    @property
    def r_x(self) -> float:
        """Diffusion number in x: alpha * dt / dx²."""
        if self.dt is None:
            raise RuntimeError("dt must be set before computing r_x")
        return self.alpha * self.dt / self.dx**2

    @property
    def r_y(self) -> float:
        """Diffusion number in y: alpha * dt / dy²."""
        if self.dt is None:
            raise RuntimeError("dt must be set before computing r_y")
        return self.alpha * self.dt / self.dy**2

    @property
    def stability_limit_ftcs(self) -> float:
        """Maximum stable dt for FTCS in 1D: dx² / (2 * alpha)."""
        if self.dimension == Dimension.ONE_D:
            return self.dx**2 / (2.0 * self.alpha)
        else:
            return 1.0 / (2.0 * self.alpha * (1.0 / self.dx**2 + 1.0 / self.dy**2))

    def choose_stable_dt(self, safety_factor: float = 0.4) -> float:
        """
        Automatically choose a stable time step for FTCS.

        Parameters
        ----------
        safety_factor : float
            Fraction of the stability limit to use (< 1 for safety).

        Returns
        -------
        float
            Stable time step size.
        """
        return safety_factor * self.stability_limit_ftcs
