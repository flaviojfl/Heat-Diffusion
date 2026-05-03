"""
Abstract base class for all heat diffusion solvers.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray

from ..config import Dimension, SimulationConfig

logger = logging.getLogger(__name__)

ArrayFloat = NDArray[np.float64]


@dataclass
class SimulationResult:
    """
    Container for simulation results.

    Attributes
    ----------
    times : list[float]
        Time stamps at which states were saved.
    states : list[ArrayFloat]
        Saved solution states. Each element has the same shape as the
        initial condition.
    config : SimulationConfig
        The configuration used for this simulation.
    solver_name : str
        Human-readable name of the solver used.
    wall_time : float
        Wall-clock time taken by the solver in seconds.
    errors : list[float]
        L2 errors against analytical solution (empty if not available).
    """

    times: list[float] = field(default_factory=list)
    states: list[ArrayFloat] = field(default_factory=list)
    config: Optional[SimulationConfig] = None
    solver_name: str = ""
    wall_time: float = 0.0
    errors: list[float] = field(default_factory=list)

    @property
    def final_state(self) -> ArrayFloat:
        """The last saved state."""
        if not self.states:
            raise RuntimeError("No states have been saved yet.")
        return self.states[-1]

    @property
    def n_steps_saved(self) -> int:
        """Number of saved time steps."""
        return len(self.times)


class BaseSolver(ABC):
    """
    Abstract base class for all heat diffusion solvers.

    All concrete solvers (FTCS, Crank-Nicolson, θ-method) inherit from
    this class and implement the `step` and `build_system` methods.

    Parameters
    ----------
    config : SimulationConfig
        Simulation configuration.
    initial_condition : Callable[[ArrayFloat], ArrayFloat] | ArrayFloat
        Either a callable f(x) → u0 (1D) or f(x, y) → u0 (2D),
        or a pre-computed numpy array of the correct shape.
    analytical_solution : Callable | None
        Optional reference solution for error tracking.
    """

    def __init__(
        self,
        config: SimulationConfig,
        initial_condition: Callable | ArrayFloat,
        analytical_solution: Callable | None = None,
    ) -> None:
        self.config = config
        self.analytical_solution = analytical_solution
        self._setup_grid()
        self._resolve_dt()
        self._check_stability()
        self.u = self._init_solution(initial_condition)
        self._build_system()
        logger.info("Solver %s initialised. dt=%.2e, r=%.4f",
                    self.name, self.config.dt, self._diffusion_number())

    # ------------------------------------------------------------------
    # Properties to override
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the solver."""
        ...

    # ------------------------------------------------------------------
    # Abstract methods to implement
    # ------------------------------------------------------------------

    @abstractmethod
    def step(self) -> None:
        """Advance the solution by one time step in-place."""
        ...

    @abstractmethod
    def _build_system(self) -> None:
        """Pre-compute matrices or any solver-specific structures."""
        ...

    # ------------------------------------------------------------------
    # Common setup helpers
    # ------------------------------------------------------------------

    def _setup_grid(self) -> None:
        """Create spatial grids."""
        cfg = self.config
        self.x = np.linspace(0.0, cfg.L_x, cfg.nx)
        if cfg.dimension == Dimension.TWO_D:
            self.y = np.linspace(0.0, cfg.L_y, cfg.ny)
            self.X, self.Y = np.meshgrid(self.x, self.y)
        self.t = 0.0

    def _resolve_dt(self) -> None:
        """If dt is not set, choose it automatically."""
        if self.config.dt is None:
            self.config.dt = self.config.choose_stable_dt(safety_factor=0.4)
            logger.info("dt not specified; auto-selected dt=%.4e", self.config.dt)

    def _check_stability(self) -> None:
        """
        Verify the Von Neumann stability criterion for FTCS.
        Warns if violated; implicit methods are unconditionally stable.
        """
        from ..config import SolverType
        cfg = self.config
        if cfg.solver_type == SolverType.FTCS:
            limit = cfg.stability_limit_ftcs
            if cfg.dt > limit:
                raise ValueError(
                    f"FTCS instability detected! dt={cfg.dt:.4e} > limit={limit:.4e}. "
                    f"Reduce dt or use an implicit solver (Crank-Nicolson / θ-method)."
                )
            logger.info(
                "Von Neumann CFL check passed: dt=%.4e (limit=%.4e, ratio=%.3f)",
                cfg.dt, limit, cfg.dt / limit,
            )
        else:
            logger.info("Implicit solver – unconditionally stable (CFL check skipped).")

    def _diffusion_number(self) -> float:
        """Return the diffusion number r = alpha * dt / dx²."""
        return self.config.alpha * self.config.dt / self.config.dx ** 2

    def _init_solution(
        self,
        initial_condition: Callable | ArrayFloat,
    ) -> ArrayFloat:
        """Evaluate or copy the initial condition."""
        cfg = self.config
        if isinstance(initial_condition, np.ndarray):
            u = initial_condition.astype(np.float64)
        elif cfg.dimension == Dimension.ONE_D:
            u = initial_condition(self.x).astype(np.float64)
        else:
            u = initial_condition(self.X, self.Y).astype(np.float64)

        expected = (cfg.nx,) if cfg.dimension == Dimension.ONE_D else (cfg.ny, cfg.nx)
        if u.shape != expected:
            raise ValueError(
                f"Initial condition shape {u.shape} does not match grid {expected}."
            )
        return u

    # ------------------------------------------------------------------
    # Main solve loop
    # ------------------------------------------------------------------

    def solve(self) -> SimulationResult:
        """
        Run the full simulation from t=0 to t_end.

        Returns
        -------
        SimulationResult
            Object containing saved states, times, errors, and metadata.
        """
        cfg = self.config
        result = SimulationResult(config=cfg, solver_name=self.name)

        n_steps = int(np.ceil(cfg.t_end / cfg.dt))
        actual_dt = cfg.t_end / n_steps  # adjust dt to hit t_end exactly
        if abs(actual_dt - cfg.dt) / cfg.dt > 0.01:
            logger.debug(
                "dt adjusted from %.4e to %.4e to hit t_end exactly.",
                cfg.dt, actual_dt,
            )
        cfg.dt = actual_dt
        self._build_system()  # rebuild if dt changed

        t_start_wall = time.perf_counter()

        # Save initial state
        result.times.append(self.t)
        result.states.append(self.u.copy())
        self._record_error(result, self.t)

        for step_idx in range(1, n_steps + 1):
            self.step()
            self.t += cfg.dt

            if step_idx % cfg.save_every == 0 or step_idx == n_steps:
                result.times.append(self.t)
                result.states.append(self.u.copy())
                self._record_error(result, self.t)

            if cfg.verbose and step_idx % max(1, n_steps // 10) == 0:
                logger.info(
                    "  [%s] Step %d/%d  t=%.4f  max(u)=%.4f",
                    self.name, step_idx, n_steps, self.t, float(np.max(self.u)),
                )

        result.wall_time = time.perf_counter() - t_start_wall
        logger.info(
            "Simulation finished. Solver=%s  Wall time=%.3fs  Steps=%d",
            self.name, result.wall_time, n_steps,
        )
        return result

    def _record_error(self, result: SimulationResult, t: float) -> None:
        """Compute and record L2 error if analytical solution is available."""
        if self.analytical_solution is None:
            return
        from ..utils.analytical import compute_l2_error
        cfg = self.config
        if cfg.dimension == Dimension.ONE_D:
            u_exact = self.analytical_solution(self.x, t)
            err = compute_l2_error(self.u, u_exact, cfg.dx)
        else:
            u_exact = self.analytical_solution(self.X, self.Y, t)
            err = compute_l2_error(self.u, u_exact, cfg.dx, cfg.dy)
        result.errors.append(err)
