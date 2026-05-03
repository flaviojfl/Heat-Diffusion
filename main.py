"""
Heat Diffusion Simulation — Main Entry Point

Demonstrates all three numerical schemes in 1D and 2D with:
- Automatic Von Neumann stability checking
- Comparison plots between solvers
- Error tracking against analytical solution
- Animation generation
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Use non-interactive backend when running headlessly
matplotlib.use("Agg")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("heat_diffusion.main")

from heat_diffusion.config import (
    BoundaryConditionConfig,
    BoundaryConditionsConfig,
    BoundaryType,
    Dimension,
    SimulationConfig,
    SolverType,
)
from heat_diffusion.solvers import make_solver
from heat_diffusion.utils.analytical import analytical_1d_sine, analytical_2d_sine
from heat_diffusion.visualization.animator import animate_1d, animate_2d
from heat_diffusion.visualization.plotter import (
    plot_1d_comparison,
    plot_1d_snapshots,
    plot_2d_heatmap,
    plot_2d_surface,
    plot_convergence,
    plot_error_evolution,
)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

ALPHA = 1.0e-3
LX = 1.0


# ──────────────────────────────────────────────────────────────────────
# Initial conditions
# ──────────────────────────────────────────────────────────────────────

def ic_1d_sine(x: np.ndarray) -> np.ndarray:
    """Single-mode sine: u(x,0) = sin(πx/L)"""
    return np.sin(np.pi * x / LX)


def ic_2d_sine(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """2D sine IC: u(x,y,0) = sin(πx/Lx) * sin(πy/Ly)"""
    return np.sin(np.pi * x / LX) * np.sin(np.pi * y / LX)


def ic_2d_hotspot(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Gaussian hot-spot at the centre of the domain."""
    cx, cy = LX / 2, LX / 2
    sigma = 0.08
    return 100.0 * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))


# ──────────────────────────────────────────────────────────────────────
# Demo 1: 1D comparison of all three solvers
# ──────────────────────────────────────────────────────────────────────

def demo_1d_comparison() -> None:
    logger.info("=" * 60)
    logger.info("DEMO 1: 1D Solver Comparison")
    logger.info("=" * 60)

    results = []
    for solver_type in [SolverType.FTCS, SolverType.CRANK_NICOLSON, SolverType.THETA_METHOD]:
        theta = 0.5 if solver_type == SolverType.THETA_METHOD else 0.5
        cfg = SimulationConfig(
            alpha=ALPHA,
            L_x=LX,
            nx=80,
            t_end=0.2,
            solver_type=solver_type,
            theta=theta,
            dimension=Dimension.ONE_D,
            save_every=10,
            verbose=True,
        )
        solver = make_solver(
            cfg,
            ic_1d_sine,
            analytical_solution=lambda x, t: analytical_1d_sine(x, t, ALPHA, LX),
        )
        result = solver.solve()
        results.append(result)
        logger.info("  %s  wall_time=%.3fs", result.solver_name, result.wall_time)

    # Comparison plot
    plot_1d_comparison(
        results,
        t_target=0.1,
        analytical_fn=lambda x, t: analytical_1d_sine(x, t, ALPHA, LX),
        save_path=OUTPUT_DIR / "1d_solver_comparison.png",
    )

    # Error evolution
    plot_error_evolution(results, save_path=OUTPUT_DIR / "1d_error_evolution.png")

    # Snapshots for CN
    plot_1d_snapshots(
        results[1],  # Crank-Nicolson
        n_snapshots=6,
        analytical_fn=lambda x, t: analytical_1d_sine(x, t, ALPHA, LX),
        save_path=OUTPUT_DIR / "1d_cn_snapshots.png",
    )
    logger.info("Demo 1 complete. Plots saved to %s/", OUTPUT_DIR)


# ──────────────────────────────────────────────────────────────────────
# Demo 2: θ-method parameter sweep
# ──────────────────────────────────────────────────────────────────────

def demo_theta_sweep() -> None:
    logger.info("=" * 60)
    logger.info("DEMO 2: θ-method parameter sweep")
    logger.info("=" * 60)

    theta_values = [0.0, 0.5, 1.0]
    results = []
    for theta in theta_values:
        cfg = SimulationConfig(
            alpha=ALPHA,
            L_x=LX,
            nx=60,
            t_end=0.15,
            solver_type=SolverType.THETA_METHOD,
            theta=theta,
            dimension=Dimension.ONE_D,
            save_every=5,
            verbose=False,
        )
        solver = make_solver(
            cfg,
            ic_1d_sine,
            analytical_solution=lambda x, t: analytical_1d_sine(x, t, ALPHA, LX),
        )
        results.append(solver.solve())

    plot_1d_comparison(
        results,
        t_target=0.1,
        analytical_fn=lambda x, t: analytical_1d_sine(x, t, ALPHA, LX),
        save_path=OUTPUT_DIR / "theta_sweep_comparison.png",
    )
    plot_error_evolution(results, save_path=OUTPUT_DIR / "theta_sweep_errors.png")
    logger.info("Demo 2 complete.")


# ──────────────────────────────────────────────────────────────────────
# Demo 3: 2D simulation (heatmap + surface)
# ──────────────────────────────────────────────────────────────────────

def demo_2d_simulation() -> None:
    logger.info("=" * 60)
    logger.info("DEMO 3: 2D Heat Diffusion")
    logger.info("=" * 60)

    # ── 2D with Gaussian hot-spot (Crank-Nicolson) ──
    cfg = SimulationConfig(
        alpha=ALPHA,
        L_x=LX,
        L_y=LX,
        nx=50,
        ny=50,
        t_end=0.3,
        solver_type=SolverType.CRANK_NICOLSON,
        dimension=Dimension.TWO_D,
        save_every=5,
        verbose=True,
    )
    solver = make_solver(cfg, ic_2d_hotspot)
    result = solver.solve()

    plot_2d_heatmap(result, time_index=-1,
                    save_path=OUTPUT_DIR / "2d_heatmap_final.png")
    plot_2d_surface(result, time_index=-1,
                    save_path=OUTPUT_DIR / "2d_surface_final.png")

    # Intermediate time steps
    for idx in [0, len(result.states) // 3, len(result.states) // 2, -1]:
        plot_2d_heatmap(result, time_index=idx,
                        save_path=OUTPUT_DIR / f"2d_heatmap_step{idx}.png")

    logger.info("Demo 3 complete.")


# ──────────────────────────────────────────────────────────────────────
# Demo 4: Convergence analysis
# ──────────────────────────────────────────────────────────────────────

def demo_convergence() -> None:
    logger.info("=" * 60)
    logger.info("DEMO 4: Spatial Convergence Analysis")
    logger.info("=" * 60)

    nx_values = [20, 30, 50, 80, 120, 200]
    results_by_solver: dict[str, tuple[list, list]] = {}

    for solver_type in [SolverType.FTCS, SolverType.CRANK_NICOLSON]:
        step_sizes, errors = [], []
        for nx in nx_values:
            dx = LX / (nx - 1)
            cfg = SimulationConfig(
                alpha=ALPHA,
                L_x=LX,
                nx=nx,
                t_end=0.05,
                solver_type=solver_type,
                dimension=Dimension.ONE_D,
                save_every=100000,
                verbose=False,
            )
            solver = make_solver(
                cfg,
                ic_1d_sine,
                analytical_solution=lambda x, t: analytical_1d_sine(x, t, ALPHA, LX),
            )
            result = solver.solve()
            if result.errors:
                step_sizes.append(dx)
                errors.append(result.errors[-1])

        results_by_solver[solver_type.value] = (step_sizes, errors)

    # Plot convergence for CN
    hs, errs = results_by_solver[SolverType.CRANK_NICOLSON.value]
    plot_convergence(
        hs, errs,
        expected_order=2.0,
        label="Crank-Nicolson",
        save_path=OUTPUT_DIR / "convergence_cn.png",
    )

    # FTCS
    hs, errs = results_by_solver[SolverType.FTCS.value]
    plot_convergence(
        hs, errs,
        expected_order=2.0,
        label="FTCS",
        save_path=OUTPUT_DIR / "convergence_ftcs.png",
    )

    logger.info("Demo 4 complete.")


# ──────────────────────────────────────────────────────────────────────
# Demo 5: Boundary conditions showcase
# ──────────────────────────────────────────────────────────────────────

def demo_boundary_conditions() -> None:
    logger.info("=" * 60)
    logger.info("DEMO 5: Mixed Boundary Conditions")
    logger.info("=" * 60)

    bc_cfg = BoundaryConditionsConfig(
        left=BoundaryConditionConfig(BoundaryType.DIRICHLET, value=100.0),
        right=BoundaryConditionConfig(BoundaryType.NEUMANN, value=0.0),
    )
    cfg = SimulationConfig(
        alpha=ALPHA,
        L_x=LX,
        nx=80,
        t_end=1.0,
        solver_type=SolverType.CRANK_NICOLSON,
        dimension=Dimension.ONE_D,
        boundary_conditions=bc_cfg,
        save_every=10,
        verbose=False,
    )

    def ic_mixed(x: np.ndarray) -> np.ndarray:
        return 100.0 * (1.0 - x / LX)

    solver = make_solver(cfg, ic_mixed)
    result = solver.solve()
    plot_1d_snapshots(
        result, n_snapshots=6,
        save_path=OUTPUT_DIR / "1d_mixed_bc.png",
    )
    logger.info("Demo 5 complete.")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Starting Heat Diffusion Simulation Suite")
    logger.info("Output directory: %s", OUTPUT_DIR.resolve())

    demo_1d_comparison()
    demo_theta_sweep()
    demo_2d_simulation()
    demo_convergence()
    demo_boundary_conditions()

    logger.info("All demos complete. Check %s/ for output files.", OUTPUT_DIR)
