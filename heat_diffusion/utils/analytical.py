"""
Analytical solutions for the heat equation.

Provides exact solutions for standard benchmark problems used in
convergence testing and validation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

ArrayFloat = NDArray[np.float64]


def analytical_1d_sine(
    x: ArrayFloat,
    t: float,
    alpha: float,
    L: float = 1.0,
    n_modes: int = 20,
) -> ArrayFloat:
    """
    Analytical solution for the 1D heat equation with homogeneous Dirichlet BCs.

    Solves:
        ∂u/∂t = α ∂²u/∂x²,   x ∈ [0, L],   t > 0
        u(0, t) = u(L, t) = 0
        u(x, 0) = sin(π x / L)

    The exact solution is:
        u(x, t) = sin(π x / L) * exp(-α (π/L)² t)

    For general initial conditions expanded in a Fourier sine series:
        u(x, t) = Σ_n B_n sin(n π x / L) exp(-α (n π/L)² t)

    Parameters
    ----------
    x : ArrayFloat
        Spatial coordinates in [0, L].
    t : float
        Current time.
    alpha : float
        Thermal diffusivity.
    L : float
        Domain length.
    n_modes : int
        Number of Fourier modes to include (for generality).

    Returns
    -------
    ArrayFloat
        Temperature distribution at time t.

    Notes
    -----
    For the specific IC u(x,0) = sin(πx/L), only n=1 contributes:
        u(x, t) = exp(-α(π/L)² t) * sin(πx/L)
    """
    u = np.exp(-alpha * (np.pi / L) ** 2 * t) * np.sin(np.pi * x / L)
    return u


def analytical_1d_multi_mode(
    x: ArrayFloat,
    t: float,
    alpha: float,
    L: float = 1.0,
    n_modes: int = 50,
) -> ArrayFloat:
    """
    Fourier series solution for 1D heat equation with a hot-spot initial condition.

    Initial condition: u(x,0) = 1 on [L/4, 3L/4], 0 elsewhere.
    Boundary conditions: u(0,t) = u(L,t) = 0 (Dirichlet).

    The Fourier sine coefficients are:
        B_n = (2/L) ∫₀ᴸ u(x,0) sin(nπx/L) dx

    Parameters
    ----------
    x : ArrayFloat
        Spatial coordinates.
    t : float
        Current time.
    alpha : float
        Thermal diffusivity.
    L : float
        Domain length.
    n_modes : int
        Number of Fourier modes for the series truncation.

    Returns
    -------
    ArrayFloat
        Temperature field at time t.
    """
    u = np.zeros_like(x, dtype=np.float64)
    for n in range(1, n_modes + 1):
        # Integral of sin(nπx/L) over [L/4, 3L/4]
        k = n * np.pi / L
        Bn = (2.0 / L) * (
            -np.cos(k * 3.0 * L / 4.0) / k + np.cos(k * L / 4.0) / k
        )
        lam = alpha * (n * np.pi / L) ** 2
        u += Bn * np.sin(k * x) * np.exp(-lam * t)
    return u


def analytical_2d_sine(
    x: ArrayFloat,
    y: ArrayFloat,
    t: float,
    alpha: float,
    Lx: float = 1.0,
    Ly: float = 1.0,
) -> ArrayFloat:
    """
    Analytical solution for the 2D heat equation.

    Solves:
        ∂u/∂t = α (∂²u/∂x² + ∂²u/∂y²)
        u = 0 on all boundaries
        u(x,y,0) = sin(πx/Lx) * sin(πy/Ly)

    Exact solution:
        u(x,y,t) = sin(πx/Lx) sin(πy/Ly) exp(-α π²(1/Lx² + 1/Ly²) t)

    Parameters
    ----------
    x : ArrayFloat
        2D array of x-coordinates (meshgrid format).
    y : ArrayFloat
        2D array of y-coordinates (meshgrid format).
    t : float
        Current time.
    alpha : float
        Thermal diffusivity.
    Lx : float
        Domain length in x.
    Ly : float
        Domain length in y.

    Returns
    -------
    ArrayFloat
        2D temperature field at time t.
    """
    decay = np.exp(-alpha * np.pi**2 * (1.0 / Lx**2 + 1.0 / Ly**2) * t)
    return np.sin(np.pi * x / Lx) * np.sin(np.pi * y / Ly) * decay


def compute_l2_error(
    numerical: ArrayFloat,
    analytical: ArrayFloat,
    dx: float,
    dy: float | None = None,
) -> float:
    """
    Compute the L2 (RMS) error between numerical and analytical solutions.

    Parameters
    ----------
    numerical : ArrayFloat
        Numerical solution array.
    analytical : ArrayFloat
        Analytical solution array (same shape).
    dx : float
        Spatial step in x.
    dy : float | None
        Spatial step in y. If None, 1D formula is used.

    Returns
    -------
    float
        L2 norm of the error: sqrt(Σ(u_num - u_exact)² * dx [* dy]).
    """
    diff = numerical - analytical
    if dy is None:
        return float(np.sqrt(np.sum(diff**2) * dx))
    return float(np.sqrt(np.sum(diff**2) * dx * dy))


def compute_linf_error(
    numerical: ArrayFloat,
    analytical: ArrayFloat,
) -> float:
    """
    Compute the L∞ (max absolute) error.

    Parameters
    ----------
    numerical : ArrayFloat
        Numerical solution array.
    analytical : ArrayFloat
        Analytical solution array (same shape).

    Returns
    -------
    float
        Maximum absolute error: max|u_num - u_exact|.
    """
    return float(np.max(np.abs(numerical - analytical)))
