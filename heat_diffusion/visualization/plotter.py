"""
Static plotting utilities for heat diffusion simulation results.

Provides publication-quality figures for:
- 1D solution snapshots
- 2D heatmaps and 3D surface plots
- Comparative plots between multiple solvers
- Error evolution over time
- Convergence analysis plots
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection

from ..config import Dimension
from ..solvers.base import SimulationResult

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Style defaults
# ──────────────────────────────────────────────────────────────────────
CMAP_HEAT = "inferno"
CMAP_DIFF = "RdBu_r"
SOLVER_COLORS = ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A"]
SOLVER_STYLES = ["-", "--", "-.", ":"]


def _save_or_show(fig: Figure, save_path: Optional[Path]) -> None:
    """Save figure to disk or display it interactively."""
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Figure saved to %s", save_path)
    else:
        plt.show()


# ──────────────────────────────────────────────────────────────────────
# 1-D Plotting
# ──────────────────────────────────────────────────────────────────────

def plot_1d_snapshots(
    result: SimulationResult,
    n_snapshots: int = 5,
    analytical_fn: Optional[object] = None,
    save_path: Optional[Path] = None,
) -> Figure:
    """
    Plot multiple time snapshots of a 1D simulation.

    Parameters
    ----------
    result : SimulationResult
        Simulation output from a 1D solver.
    n_snapshots : int
        Number of evenly-spaced snapshots to display.
    analytical_fn : Callable | None
        If provided, overlays the exact solution at each snapshot time.
    save_path : Path | None
        If given, saves the figure to this path instead of showing it.

    Returns
    -------
    Figure
        Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0f0f12")
    ax.set_facecolor("#161620")

    x = np.linspace(
        0.0,
        result.config.L_x,
        result.config.nx,
    )

    # Choose evenly-spaced indices
    indices = np.linspace(0, len(result.states) - 1, n_snapshots, dtype=int)
    cmap = plt.cm.plasma
    colors = [cmap(i / (n_snapshots - 1)) for i in range(n_snapshots)]

    for k, idx in enumerate(indices):
        t = result.times[idx]
        u = result.states[idx]
        ax.plot(x, u, color=colors[k], linewidth=2, label=f"t = {t:.3f} s")
        if analytical_fn is not None:
            u_exact = analytical_fn(x, t)
            ax.plot(
                x, u_exact,
                color=colors[k], linewidth=1, linestyle="--", alpha=0.6,
            )

    if analytical_fn is not None:
        ax.plot([], [], "w--", linewidth=1, label="Analytical (dashed)")

    ax.set_xlabel("x [m]", color="white")
    ax.set_ylabel("Temperature [K]", color="white")
    ax.set_title(
        f"Heat Diffusion — {result.solver_name}\n"
        f"α={result.config.alpha:.2e}  Δx={result.config.dx:.4f}  "
        f"Δt={result.config.dt:.2e}",
        color="white", fontsize=11,
    )
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.legend(framealpha=0.3, labelcolor="white", fontsize=9)
    ax.grid(True, color="#333", linewidth=0.5, alpha=0.7)
    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


def plot_1d_comparison(
    results: List[SimulationResult],
    t_target: float,
    analytical_fn: Optional[object] = None,
    save_path: Optional[Path] = None,
) -> Figure:
    """
    Compare multiple solver results at a given time.

    Parameters
    ----------
    results : list[SimulationResult]
        Simulation outputs from multiple solvers.
    t_target : float
        Time at which to compare (nearest saved state is used).
    analytical_fn : Callable | None
        Analytical solution for reference.
    save_path : Path | None
        Output path for saving.

    Returns
    -------
    Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#0f0f12")

    ax_sol, ax_err = axes
    for ax in axes:
        ax.set_facecolor("#161620")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        ax.grid(True, color="#333", linewidth=0.5)

    x = np.linspace(0, results[0].config.L_x, results[0].config.nx)

    for i, res in enumerate(results):
        # Find nearest time index
        idx = int(np.argmin(np.abs(np.array(res.times) - t_target)))
        t_actual = res.times[idx]
        u = res.states[idx]
        color = SOLVER_COLORS[i % len(SOLVER_COLORS)]
        style = SOLVER_STYLES[i % len(SOLVER_STYLES)]

        ax_sol.plot(x, u, color=color, linestyle=style, linewidth=2,
                    label=res.solver_name)
        if analytical_fn is not None:
            u_exact = analytical_fn(x, t_actual)
            ax_err.semilogy(
                x, np.abs(u - u_exact) + 1e-16,
                color=color, linestyle=style, linewidth=1.5,
                label=res.solver_name,
            )

    if analytical_fn is not None:
        u_exact = analytical_fn(x, t_target)
        ax_sol.plot(x, u_exact, "w--", linewidth=2, alpha=0.7, label="Analytical")

    ax_sol.set_xlabel("x [m]", color="white")
    ax_sol.set_ylabel("Temperature [K]", color="white")
    ax_sol.set_title(f"Solution at t ≈ {t_target:.3f} s", color="white")
    ax_sol.legend(framealpha=0.3, labelcolor="white", fontsize=9)

    ax_err.set_xlabel("x [m]", color="white")
    ax_err.set_ylabel("|u_num − u_exact|", color="white")
    ax_err.set_title("Pointwise Error", color="white")
    ax_err.legend(framealpha=0.3, labelcolor="white", fontsize=9)

    fig.suptitle("Solver Comparison — Heat Diffusion 1D", color="white",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


def plot_error_evolution(
    results: List[SimulationResult],
    save_path: Optional[Path] = None,
) -> Figure:
    """
    Plot the L2 error over time for one or more solvers.

    Parameters
    ----------
    results : list[SimulationResult]
        Simulation results with non-empty `.errors` lists.
    save_path : Path | None
        Output path for saving.

    Returns
    -------
    Figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0f0f12")
    ax.set_facecolor("#161620")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    for i, res in enumerate(results):
        if not res.errors:
            logger.warning("No errors recorded for solver '%s'.", res.solver_name)
            continue
        color = SOLVER_COLORS[i % len(SOLVER_COLORS)]
        ax.semilogy(res.times, res.errors,
                    color=color, linewidth=2, label=res.solver_name)

    ax.set_xlabel("Time [s]", color="white")
    ax.set_ylabel("L2 Error ||u_num − u_exact||₂", color="white")
    ax.set_title("Error Evolution Over Time", color="white", fontsize=12)
    ax.legend(framealpha=0.3, labelcolor="white")
    ax.grid(True, color="#333", linewidth=0.5)
    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


# ──────────────────────────────────────────────────────────────────────
# 2-D Plotting
# ──────────────────────────────────────────────────────────────────────

def plot_2d_heatmap(
    result: SimulationResult,
    time_index: int = -1,
    save_path: Optional[Path] = None,
) -> Figure:
    """
    Plot a 2D heatmap at a given saved time step.

    Parameters
    ----------
    result : SimulationResult
        Simulation output from a 2D solver.
    time_index : int
        Index into result.states to plot (-1 = final).
    save_path : Path | None
        Output path for saving.

    Returns
    -------
    Figure
    """
    cfg = result.config
    u = result.states[time_index]
    t = result.times[time_index]

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#0f0f12")
    ax.set_facecolor("#161620")

    x = np.linspace(0, cfg.L_x, cfg.nx)
    y = np.linspace(0, cfg.L_y, cfg.ny)
    X, Y = np.meshgrid(x, y)

    im = ax.contourf(X, Y, u, levels=50, cmap=CMAP_HEAT)
    cb: Colorbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Temperature [K]", color="white")
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="white")

    ax.contour(X, Y, u, levels=10, colors="white", linewidths=0.5, alpha=0.4)
    ax.set_xlabel("x [m]", color="white")
    ax.set_ylabel("y [m]", color="white")
    ax.set_title(
        f"2D Heat Diffusion — {result.solver_name}\nt = {t:.4f} s",
        color="white", fontsize=11,
    )
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


def plot_2d_surface(
    result: SimulationResult,
    time_index: int = -1,
    save_path: Optional[Path] = None,
) -> Figure:
    """
    Render a 3D surface plot of the 2D temperature field.

    Parameters
    ----------
    result : SimulationResult
        Simulation output from a 2D solver.
    time_index : int
        Index into result.states to plot (-1 = final).
    save_path : Path | None
        Output path.

    Returns
    -------
    Figure
    """
    cfg = result.config
    u = result.states[time_index]
    t = result.times[time_index]

    fig = plt.figure(figsize=(10, 7))
    fig.patch.set_facecolor("#0f0f12")
    ax: Axes3D = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#0f0f12")

    x = np.linspace(0, cfg.L_x, cfg.nx)
    y = np.linspace(0, cfg.L_y, cfg.ny)
    X, Y = np.meshgrid(x, y)

    surf = ax.plot_surface(X, Y, u, cmap=CMAP_HEAT, edgecolor="none", alpha=0.92)
    cb = fig.colorbar(surf, ax=ax, fraction=0.03, pad=0.1)
    cb.set_label("T [K]", color="white")
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="white")

    ax.set_xlabel("x [m]", color="white", labelpad=8)
    ax.set_ylabel("y [m]", color="white", labelpad=8)
    ax.set_zlabel("T [K]", color="white", labelpad=8)
    ax.set_title(
        f"3D Surface — {result.solver_name}  (t = {t:.4f} s)",
        color="white", pad=12,
    )
    ax.tick_params(colors="white")
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
    ax.grid(True, color="#333")
    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


# ──────────────────────────────────────────────────────────────────────
# Convergence Analysis
# ──────────────────────────────────────────────────────────────────────

def plot_convergence(
    step_sizes: Sequence[float],
    errors: Sequence[float],
    expected_order: float = 2.0,
    label: str = "Numerical Error",
    save_path: Optional[Path] = None,
) -> Figure:
    """
    Plot convergence rate on a log-log scale.

    Parameters
    ----------
    step_sizes : Sequence[float]
        Grid or time step sizes (x-axis).
    errors : Sequence[float]
        Corresponding L2 errors (y-axis).
    expected_order : float
        Theoretical convergence order for the reference slope line.
    label : str
        Legend label for the numerical errors.
    save_path : Path | None
        Output path.

    Returns
    -------
    Figure
    """
    hs = np.array(step_sizes)
    errs = np.array(errors)

    # Fit observed convergence order
    coeffs = np.polyfit(np.log(hs), np.log(errs), 1)
    observed_order = coeffs[0]

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("#0f0f12")
    ax.set_facecolor("#161620")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    ax.loglog(hs, errs, "o-", color=SOLVER_COLORS[0], linewidth=2,
              markersize=7, label=f"{label} (order ≈ {observed_order:.2f})")

    # Reference slope
    ref_y = errs[0] * (hs / hs[0]) ** expected_order
    ax.loglog(hs, ref_y, "--", color="white", linewidth=1.5, alpha=0.6,
              label=f"O(h^{expected_order}) reference")

    ax.set_xlabel("Step size h", color="white")
    ax.set_ylabel("L2 Error", color="white")
    ax.set_title(
        f"Convergence Analysis  (observed order: {observed_order:.2f})",
        color="white",
    )
    ax.legend(framealpha=0.3, labelcolor="white")
    ax.grid(True, which="both", color="#333", linewidth=0.5)
    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig
