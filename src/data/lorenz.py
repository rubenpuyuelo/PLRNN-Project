"""Lorenz (1963) chaotic dynamical system."""

import numpy as np
from .simulation import simulate_ode


def lorenz_rhs(t: float, state: np.ndarray, *, sigma: float = 10.0,
               rho: float = 28.0, beta: float = 8.0 / 3.0) -> np.ndarray:
    """Return (dx/dt, dy/dt, dz/dt) for the Lorenz system.

    Equations:
        x' = sigma * (y - x)
        y' = x * (rho - z) - y
        z' = x * y - beta * z
    """
    x, y, z = state
    return np.array([
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z,
    ])


def simulate_lorenz(*, n_steps: int = 1000, dt: float = 0.02,
                    burn_in: float = 5.0, initial_state=(1.0, 1.0, 1.0),
                    sigma: float = 10.0, rho: float = 28.0,
                    beta: float = 8.0 / 3.0) -> tuple[np.ndarray, np.ndarray]:
    """Generate a clean trajectory of the Lorenz attractor."""
    def rhs(t, state):
        return lorenz_rhs(t, state, sigma=sigma, rho=rho, beta=beta)
    return simulate_ode(rhs, initial_state, n_steps=n_steps, dt=dt, burn_in=burn_in)
