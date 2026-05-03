"""
Animation utilities for heat diffusion simulation results.

Generates smooth animations of the temperature field evolution using
matplotlib.animation for both 1D and 2D simulations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ..solvers.base import SimulationResult

logger = logging.getLogger(__name__)

CMAP_HEAT = "inferno"


def animate_1d(
    result: SimulationResult,
    interval: int = 50,
    save_path: Optional[Path] = None,
    fps: int = 20,
    analytical_fn: Optional[object] = None,
) -> animation.FuncAnimation:
    """
    Animate the 1D temperature field over time.

    Parameters
    ----------
    result : SimulationResult
        Simulation result with saved states.
    interval : int
        Delay between frames in milliseconds.
    save_path : Path | None
        If given, save the animation as MP4 or GIF.
    fps : int
        Frames per second for saved video.
    analytical_fn : Callable | None
        Optional analytical solution to overlay.

    Returns
    -------
    animation.FuncAnimation
        Animation object (call .save() or display in notebook).
    """
    cfg = result.config
    x = np.linspace(0.0, cfg.L_x, cfg.nx)
    all_u = np.array(result.states)

    u_min = float(all_u.min())
    u_max = float(all_u.max()) + 1e-10

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("#0f0f12")
    ax.set_facecolor("#161620")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.set_xlim(0, cfg.L_x)
    ax.set_ylim(u_min - 0.05 * abs(u_max - u_min),
                u_max + 0.05 * abs(u_max - u_min))
    ax.set_xlabel("x [m]", color="white")
    ax.set_ylabel("Temperature [K]", color="white")
    ax.grid(True, color="#333", linewidth=0.5)

    (line,) = ax.plot([], [], color="#E63946", linewidth=2.5, label="Numerical")
    time_text = ax.text(
        0.02, 0.94, "", transform=ax.transAxes,
        color="white", fontsize=10, va="top",
    )

    exact_line = None
    if analytical_fn is not None:
        (exact_line,) = ax.plot(
            [], [], "--", color="#FFD166", linewidth=1.5, alpha=0.8, label="Analytical"
        )
        ax.legend(framealpha=0.3, labelcolor="white")

    title = ax.set_title(
        f"{result.solver_name} — 1D Heat Diffusion",
        color="white", fontsize=11,
    )

    def init():
        line.set_data([], [])
        time_text.set_text("")
        if exact_line is not None:
            exact_line.set_data([], [])
        return (line, time_text) + ((exact_line,) if exact_line else ())

    def update(frame: int):
        u = result.states[frame]
        t = result.times[frame]
        line.set_data(x, u)
        time_text.set_text(f"t = {t:.4f} s")
        artists = [line, time_text]
        if exact_line is not None and analytical_fn is not None:
            u_exact = analytical_fn(x, t)
            exact_line.set_data(x, u_exact)
            artists.append(exact_line)
        return artists

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(result.states),
        init_func=init,
        interval=interval,
        blit=True,
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = save_path.suffix.lower()
        if suffix == ".gif":
            writer = animation.PillowWriter(fps=fps)
        else:
            writer = animation.FFMpegWriter(fps=fps, bitrate=1200)
        anim.save(str(save_path), writer=writer)
        logger.info("1D animation saved to %s", save_path)

    return anim


def animate_2d(
    result: SimulationResult,
    interval: int = 60,
    save_path: Optional[Path] = None,
    fps: int = 15,
) -> animation.FuncAnimation:
    """
    Animate the 2D temperature heatmap over time.

    Parameters
    ----------
    result : SimulationResult
        Simulation result from a 2D solver.
    interval : int
        Delay between frames in milliseconds.
    save_path : Path | None
        If given, save the animation as MP4 or GIF.
    fps : int
        Frames per second for saved video.

    Returns
    -------
    animation.FuncAnimation
    """
    cfg = result.config
    x = np.linspace(0, cfg.L_x, cfg.nx)
    y = np.linspace(0, cfg.L_y, cfg.ny)
    X, Y = np.meshgrid(x, y)

    all_u = np.array(result.states)
    vmin = float(all_u.min())
    vmax = float(all_u.max()) + 1e-10

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#0f0f12")
    ax.set_facecolor("#161620")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    # Initial contourf
    cf = ax.contourf(X, Y, result.states[0], levels=50,
                     cmap=CMAP_HEAT, vmin=vmin, vmax=vmax)
    cb = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Temperature [K]", color="white")
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="white")

    ax.set_xlabel("x [m]", color="white")
    ax.set_ylabel("y [m]", color="white")
    title = ax.set_title(
        f"2D Heat Diffusion — {result.solver_name}\nt = 0.0000 s",
        color="white", fontsize=11,
    )

    def init():
        return []

    def update(frame: int):
        ax.clear()
        ax.set_facecolor("#161620")
        t = result.times[frame]
        u = result.states[frame]
        cf_ = ax.contourf(X, Y, u, levels=50,
                          cmap=CMAP_HEAT, vmin=vmin, vmax=vmax)
        ax.contour(X, Y, u, levels=8, colors="white", linewidths=0.4, alpha=0.3)
        ax.set_xlabel("x [m]", color="white")
        ax.set_ylabel("y [m]", color="white")
        ax.set_title(
            f"2D Heat Diffusion — {result.solver_name}\nt = {t:.4f} s",
            color="white", fontsize=11,
        )
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        return []

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(result.states),
        init_func=init,
        interval=interval,
        blit=False,
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = save_path.suffix.lower()
        if suffix == ".gif":
            writer = animation.PillowWriter(fps=fps)
        else:
            writer = animation.FFMpegWriter(fps=fps, bitrate=1200)
        anim.save(str(save_path), writer=writer)
        logger.info("2D animation saved to %s", save_path)

    return anim
