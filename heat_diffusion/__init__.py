"""Heat Diffusion simulation package."""

from .config import (
    BoundaryConditionConfig,
    BoundaryConditionsConfig,
    BoundaryType,
    Dimension,
    SimulationConfig,
    SolverType,
)
from .solvers import (
    CrankNicolsonSolver,
    FTCSSolver,
    SimulationResult,
    ThetaMethodSolver,
    make_solver,
)
from .utils.analytical import (
    analytical_1d_sine,
    analytical_2d_sine,
    compute_l2_error,
    compute_linf_error,
)

__version__ = "1.0.0"
__all__ = [
    # Config
    "SimulationConfig",
    "SolverType",
    "BoundaryType",
    "BoundaryConditionConfig",
    "BoundaryConditionsConfig",
    "Dimension",
    # Solvers
    "make_solver",
    "FTCSSolver",
    "CrankNicolsonSolver",
    "ThetaMethodSolver",
    "SimulationResult",
    # Analytical
    "analytical_1d_sine",
    "analytical_2d_sine",
    "compute_l2_error",
    "compute_linf_error",
]
