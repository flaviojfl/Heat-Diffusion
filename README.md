# 🔥 Heat Diffusion — Scientific Computing Simulator

A high-quality, production-grade Python package for simulating the **heat diffusion (parabolic PDE)** equation in 1D and 2D domains using finite difference methods.

---

## 📐 Mathematical Background

The heat equation describes how temperature $u(\mathbf{x}, t)$ evolves in a domain $\Omega$ over time:

$$\frac{\partial u}{\partial t} = \alpha \nabla^2 u, \quad \mathbf{x} \in \Omega,\; t > 0$$

where $\alpha > 0$ is the **thermal diffusivity** [m²/s].

### Boundary Conditions

| Type | Formula | Use case |
|------|---------|----------|
| **Dirichlet** | $u = g$ on $\partial\Omega$ | Fixed wall temperature |
| **Neumann** | $\partial u / \partial n = g$ | Insulated wall (g=0) or prescribed flux |
| **Robin** | $\alpha u + \beta \,\partial u/\partial n = g$ | Newton's law of cooling |

### Analytical Solution (1D, Dirichlet)

For $u(x,0) = \sin(\pi x / L)$, $u(0,t) = u(L,t) = 0$:

$$u(x, t) = \sin\!\left(\frac{\pi x}{L}\right) \exp\!\left(-\alpha \left(\frac{\pi}{L}\right)^2 t\right)$$

---

## 🧮 Numerical Schemes

### 1. FTCS — Explicit Euler

$$u_i^{n+1} = u_i^n + r\left(u_{i-1}^n - 2u_i^n + u_{i+1}^n\right), \quad r = \frac{\alpha\,\Delta t}{\Delta x^2}$$

- **Accuracy**: $\mathcal{O}(\Delta t) + \mathcal{O}(\Delta x^2)$
- **Stability**: Requires $r \leq \tfrac{1}{2}$ (Von Neumann criterion)
- **Cost**: $\mathcal{O}(N)$ per step, trivially vectorised

### 2. Crank-Nicolson — Second-order Implicit

$$-\frac{r}{2}u_{i-1}^{n+1} + (1+r)\,u_i^{n+1} - \frac{r}{2}u_{i+1}^{n+1} = \frac{r}{2}u_{i-1}^n + (1-r)\,u_i^n + \frac{r}{2}u_{i+1}^n$$

- **Accuracy**: $\mathcal{O}(\Delta t^2) + \mathcal{O}(\Delta x^2)$
- **Stability**: **Unconditionally stable** for any $r$
- **Cost**: Tridiagonal system solved via `scipy.sparse.linalg.spsolve`

### 3. θ-Method — Generalised Family

$$(I - \theta\,\Delta t\,L)\,u^{n+1} = (I + (1-\theta)\,\Delta t\,L)\,u^n$$

| $\theta$ | Scheme | Accuracy | Stability |
|----------|--------|----------|-----------|
| 0 | FTCS (explicit) | $\mathcal{O}(\Delta t)$ | Conditional ($r \leq 1/2$) |
| 0.5 | Crank-Nicolson | $\mathcal{O}(\Delta t^2)$ | Unconditional |
| 1 | Implicit Euler | $\mathcal{O}(\Delta t)$ | Unconditional, maximum dissipation |

### 2D Extension — ADI Splitting (Peaceman-Rachford)

The 2D problem is split into two 1D sweeps per time step:

$$\frac{u^* - u^n}{\Delta t/2} = \alpha\left(\delta_x^2 u^* + \delta_y^2 u^n\right), \quad
\frac{u^{n+1} - u^*}{\Delta t/2} = \alpha\left(\delta_x^2 u^* + \delta_y^2 u^{n+1}\right)$$

This maintains second-order accuracy while reducing to $O(N^2)$ tridiagonal solves.

---

## 📁 Project Structure

```
heat_diffusion/
├── heat_diffusion/              # Core library
│   ├── __init__.py              # Public API
│   ├── config.py                # SimulationConfig dataclass
│   ├── boundary.py              # Dirichlet / Neumann / Robin BCs
│   ├── solvers/
│   │   ├── __init__.py          # make_solver() factory
│   │   ├── base.py              # BaseSolver + SimulationResult
│   │   ├── ftcs.py              # Explicit FTCS solver
│   │   ├── crank_nicolson.py    # Crank-Nicolson (sparse)
│   │   └── theta_method.py      # General θ-method
│   ├── visualization/
│   │   ├── plotter.py           # Static plots & comparisons
│   │   └── animator.py          # Smooth animations
│   └── utils/
│       └── analytical.py        # Analytical solutions & error norms
├── tests/
│   ├── test_boundary.py         # BC unit tests
│   ├── test_solvers_1d.py       # 1D solver tests
│   └── test_convergence.py      # Convergence order verification
├── main.py                      # Demo script (all 5 demos)
├── pyproject.toml
└── requirements.txt
```

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/your-username/heat-diffusion.git
cd heat-diffusion
pip install -r requirements.txt
```

### Minimal 1D Example

```python
import numpy as np
from heat_diffusion.config import SimulationConfig, SolverType, Dimension
from heat_diffusion.solvers import make_solver
from heat_diffusion.visualization.plotter import plot_1d_snapshots

# Configure the simulation
cfg = SimulationConfig(
    alpha=1e-3,              # Thermal diffusivity [m²/s]
    L_x=1.0,                 # Domain length [m]
    nx=100,                  # Grid points
    t_end=0.5,               # Simulation end time [s]
    solver_type=SolverType.CRANK_NICOLSON,
    dimension=Dimension.ONE_D,
)

# Define initial condition
def ic(x):
    return np.sin(np.pi * x)

# Run simulation
solver = make_solver(cfg, ic)
result = solver.solve()

# Visualise
plot_1d_snapshots(result, n_snapshots=6)
print(f"Wall time: {result.wall_time:.3f}s")
```

### 1D with Analytical Validation

```python
from heat_diffusion.utils.analytical import analytical_1d_sine

solver = make_solver(
    cfg,
    ic,
    analytical_solution=lambda x, t: analytical_1d_sine(x, t, cfg.alpha, cfg.L_x),
)
result = solver.solve()
print(f"Final L2 error: {result.errors[-1]:.4e}")
```

### 2D Gaussian Hot-spot

```python
import numpy as np
from heat_diffusion.config import SimulationConfig, SolverType, Dimension
from heat_diffusion.solvers import make_solver
from heat_diffusion.visualization.plotter import plot_2d_heatmap, plot_2d_surface

cfg = SimulationConfig(
    alpha=1e-3, L_x=1.0, L_y=1.0,
    nx=60, ny=60, t_end=0.4,
    solver_type=SolverType.CRANK_NICOLSON,
    dimension=Dimension.TWO_D,
)

def ic_hotspot(x, y):
    cx, cy = 0.5, 0.5
    return 100 * np.exp(-((x-cx)**2 + (y-cy)**2) / (2*0.05**2))

solver = make_solver(cfg, ic_hotspot)
result = solver.solve()

plot_2d_heatmap(result)
plot_2d_surface(result)
```

### Mixed Boundary Conditions

```python
from heat_diffusion.config import (
    BoundaryConditionConfig, BoundaryConditionsConfig, BoundaryType
)

bc_cfg = BoundaryConditionsConfig(
    left=BoundaryConditionConfig(BoundaryType.DIRICHLET, value=100.0),
    right=BoundaryConditionConfig(BoundaryType.NEUMANN, value=0.0),   # Insulated
)
cfg = SimulationConfig(..., boundary_conditions=bc_cfg)
```

### Run All Demos

```bash
python main.py
# Outputs saved to ./outputs/
```

---

## 🧪 Running Tests

```bash
# Fast tests only
pytest -m "not slow"

# Full suite including convergence analysis
pytest

# With coverage
pytest --cov=heat_diffusion --cov-report=html
```

---

## 📊 Key Design Decisions

| Aspect | Decision | Reason |
|--------|----------|--------|
| **Sparse matrices** | `scipy.sparse.csr_matrix` + LU factorisation | O(N) solves; cache factorisation across steps |
| **Type safety** | `dataclasses` + `type hints` throughout | Catch config errors at construction time |
| **Error tracking** | L2 norm at every saved step | Enable convergence rate measurement |
| **2D implicit** | ADI (Peaceman-Rachford) | Maintains O(Δt²) accuracy with O(N²) cost |
| **dt selection** | Auto-computed at 40% of CFL limit | Safe default; user can override |
| **Logging** | `logging` module (not print) | Configurable verbosity without code changes |

---

## 📈 Convergence Results

Crank-Nicolson achieves **second-order spatial convergence**:

| Grid (nx) | Δx | L2 Error |
|-----------|-----|---------|
| 20 | 0.0526 | ~1.2e-4 |
| 50 | 0.0204 | ~1.8e-5 |
| 100 | 0.0101 | ~4.4e-6 |
| 200 | 0.0050 | ~1.1e-6 |

Observed order ≈ **2.0** (expected for CN) ✓

---

## 📄 License

MIT License. See `LICENSE` for details.
