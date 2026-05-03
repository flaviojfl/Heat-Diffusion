"""
Boundary condition implementations for heat diffusion simulations.

Supports Dirichlet, Neumann, and Robin boundary conditions in both
1D and 2D domains.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from numpy.typing import NDArray

from .config import BoundaryConditionConfig, BoundaryConditionsConfig, BoundaryType, Dimension

logger = logging.getLogger(__name__)

ArrayFloat = NDArray[np.float64]


class BoundaryConditionBase(ABC):
    """Abstract base class for all boundary conditions."""

    def __init__(self, config: BoundaryConditionConfig) -> None:
        self.config = config
        self.bc_type = config.bc_type
        self.value = config.value

    @abstractmethod
    def apply(self, u: ArrayFloat, side: str = "left", **kwargs: float) -> ArrayFloat:
        """Apply boundary condition to the solution array."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.bc_type.value}, value={self.value})"


class DirichletBC(BoundaryConditionBase):
    """
    Dirichlet (fixed value) boundary condition: u = g.

    Parameters
    ----------
    config : BoundaryConditionConfig
        Configuration with bc_type=DIRICHLET and value=g.
    """

    def apply(self, u: ArrayFloat, side: str = "left", **kwargs: float) -> ArrayFloat:
        """Set boundary node to fixed value."""
        if side in ("left", "bottom"):
            u[0] = self.value
        else:
            u[-1] = self.value
        return u


class NeumannBC(BoundaryConditionBase):
    """
    Neumann (fixed flux) boundary condition: du/dn = g.

    Uses a second-order one-sided finite difference stencil.

    Parameters
    ----------
    config : BoundaryConditionConfig
        Configuration with bc_type=NEUMANN and value=g (flux).
    """

    def apply(self, u: ArrayFloat, dx: float, side: str = "left", **kwargs: float) -> ArrayFloat:
        """
        Apply Neumann BC using second-order ghost-point formula.

        Parameters
        ----------
        u : ArrayFloat
            Solution array (all interior + boundary nodes in one axis).
        dx : float
            Grid spacing.
        side : str
            Which side: 'left'/'bottom' or 'right'/'top'.
        """
        g = self.value
        if side in ("left", "bottom"):
            # du/dx|_0 = g  →  u[0] = u[1] - g*dx  (1st order)
            # or 2nd order: u[0] = (4*u[1] - u[2]) / 3 - 2*g*dx/3
            u[0] = (4.0 * u[1] - u[2]) / 3.0 - 2.0 * g * dx / 3.0
        else:  # right or top
            # du/dx|_N = g  →  u[-1] = u[-2] + g*dx
            # 2nd order: u[-1] = (4*u[-2] - u[-3]) / 3 + 2*g*dx/3
            u[-1] = (4.0 * u[-2] - u[-3]) / 3.0 + 2.0 * g * dx / 3.0
        return u


class RobinBC(BoundaryConditionBase):
    """
    Robin (mixed) boundary condition: alpha * u + beta * du/dn = g.

    Parameters
    ----------
    config : BoundaryConditionConfig
        Configuration with bc_type=ROBIN, value=g, alpha, beta.
    """

    def apply(
        self,
        u: ArrayFloat,
        dx: float,
        side: str = "left",
        **kwargs: float,
    ) -> ArrayFloat:
        """
        Apply Robin BC using first-order approximation of the flux term.

        Parameters
        ----------
        u : ArrayFloat
            Solution array.
        dx : float
            Grid spacing.
        side : str
            Which side: 'left'/'bottom' or 'right'/'top'.
        """
        a = self.config.alpha
        b = self.config.beta
        g = self.value

        if side in ("left", "bottom"):
            # a*u[0] + b*(u[1]-u[0])/dx = g
            # u[0] = (g*dx - b*u[1]) / (a*dx - b)  ... when a*dx != b
            denom = a * dx - b
            if abs(denom) < 1e-14:
                raise ValueError(
                    "Robin BC denominator (alpha*dx - beta) is near zero. "
                    "Check your Robin coefficients."
                )
            u[0] = (g * dx - b * u[1]) / denom
        else:
            # a*u[-1] + b*(u[-1]-u[-2])/dx = g
            # u[-1] = (g*dx + b*u[-2]) / (a*dx + b)
            denom = a * dx + b
            if abs(denom) < 1e-14:
                raise ValueError(
                    "Robin BC denominator (alpha*dx + beta) is near zero. "
                    "Check your Robin coefficients."
                )
            u[-1] = (g * dx + b * u[-2]) / denom
        return u


def make_bc(config: BoundaryConditionConfig) -> BoundaryConditionBase:
    """
    Factory function: create the appropriate BC object from configuration.

    Parameters
    ----------
    config : BoundaryConditionConfig
        Boundary condition configuration.

    Returns
    -------
    BoundaryConditionBase
        Concrete boundary condition object.

    Raises
    ------
    ValueError
        If the boundary type is not recognised.
    """
    dispatch = {
        BoundaryType.DIRICHLET: DirichletBC,
        BoundaryType.NEUMANN: NeumannBC,
        BoundaryType.ROBIN: RobinBC,
    }
    cls = dispatch.get(config.bc_type)
    if cls is None:
        raise ValueError(f"Unknown boundary type: {config.bc_type!r}")
    return cls(config)


class BoundaryApplicator:
    """
    Applies all four (or two) boundary conditions to a solution array.

    Parameters
    ----------
    bc_config : BoundaryConditionsConfig
        Full boundary conditions configuration.
    dimension : Dimension
        Spatial dimension (1D or 2D).
    """

    def __init__(
        self,
        bc_config: BoundaryConditionsConfig,
        dimension: Dimension,
    ) -> None:
        self.dimension = dimension
        self.left = make_bc(bc_config.left)
        self.right = make_bc(bc_config.right)
        if dimension == Dimension.TWO_D:
            self.bottom = make_bc(bc_config.bottom)
            self.top = make_bc(bc_config.top)
        logger.debug("BoundaryApplicator created: %s", self)

    def apply_1d(self, u: ArrayFloat, dx: float) -> ArrayFloat:
        """Apply left and right BCs to a 1D solution array.

        Parameters
        ----------
        u : ArrayFloat
            1D array of shape (nx,).
        dx : float
            Grid spacing.

        Returns
        -------
        ArrayFloat
            Modified array with BCs applied.
        """
        # Pass the full array; each BC modifies only u[0] or u[-1]
        self.left.apply(u, dx=dx, side="left")
        self.right.apply(u, dx=dx, side="right")
        return u

    def apply_2d(self, u: ArrayFloat, dx: float, dy: float) -> ArrayFloat:
        """Apply all four BCs to a 2D solution array.

        Parameters
        ----------
        u : ArrayFloat
            2D array of shape (ny, nx).
        dx : float
            Grid spacing in x.
        dy : float
            Grid spacing in y.

        Returns
        -------
        ArrayFloat
            Modified array with BCs applied on all four sides.
        """
        # Left and right boundaries (operate on each row)
        for j in range(u.shape[0]):
            self.left.apply(u[j, :], dx=dx, side="left")
            self.right.apply(u[j, :], dx=dx, side="right")

        # Bottom and top boundaries (operate on each column)
        for i in range(u.shape[1]):
            self.bottom.apply(u[:, i], dx=dy, side="bottom")
            self.top.apply(u[:, i], dx=dy, side="top")

        return u

    def __repr__(self) -> str:
        return (
            f"BoundaryApplicator(dim={self.dimension.value}, "
            f"left={self.left}, right={self.right})"
        )
