"""
Solver module for heat diffusion simulations.

Provides a factory function to instantiate the correct solver based on
the SimulationConfig.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..config import SimulationConfig, SolverType
from .base import BaseSolver, SimulationResult
from .crank_nicolson import CrankNicolsonSolver
from .ftcs import FTCSSolver
from .theta_method import ThetaMethodSolver

__all__ = [
    "BaseSolver",
    "SimulationResult",
    "FTCSSolver",
    "CrankNicolsonSolver",
    "ThetaMethodSolver",
    "make_solver",
]


def make_solver(
    config: SimulationConfig,
    initial_condition: Callable | np.ndarray,
    analytical_solution: Callable | None = None,
) -> BaseSolver:
    """
    Factory function: instantiate the solver specified in the config.

    Parameters
    ----------
    config : SimulationConfig
        Simulation configuration (includes solver_type).
    initial_condition : Callable or np.ndarray
        Initial temperature distribution.
    analytical_solution : Callable | None
        Optional analytical solution for error tracking.

    Returns
    -------
    BaseSolver
        Concrete solver instance ready to call `.solve()`.

    Raises
    ------
    ValueError
        If the solver type is not recognised.

    Examples
    --------
    >>> from heat_diffusion.config import SimulationConfig, SolverType
    >>> cfg = SimulationConfig(solver_type=SolverType.CRANK_NICOLSON)
    >>> solver = make_solver(cfg, lambda x: np.sin(np.pi * x))
    """
    dispatch = {
        SolverType.FTCS: FTCSSolver,
        SolverType.CRANK_NICOLSON: CrankNicolsonSolver,
        SolverType.THETA_METHOD: ThetaMethodSolver,
    }
    cls = dispatch.get(config.solver_type)
    if cls is None:
        raise ValueError(
            f"Unknown solver type: {config.solver_type!r}. "
            f"Available: {list(dispatch.keys())}"
        )
    return cls(config, initial_condition, analytical_solution)
