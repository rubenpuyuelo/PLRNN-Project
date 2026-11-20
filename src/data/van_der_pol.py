"""Van der Pol nonlinear oscillator; formulation in Koppe et al. (2019), Fig. 4E."""

import numpy as np
from .simulation import simulate_ode


def van_der_pol_rhs(t: float, state: np.ndarray, *, mu: float = 2.0,
                     omega: float = 1.0) -> np.ndarray:
    """Return (dx/dt, dy/dt) for the Van der Pol oscillator.

    Equations:
        x' = y
        y' = -mu * (x**2 - 1) * y - omega**2 * x
    """
    x, y = state
    return np.array([
        y,
        -mu * (x * x - 1.0) * y - (omega ** 2) * x,
    ])


def simulate_van_der_pol(*, n_steps: int = 1000, dt: float = 0.05,
                         burn_in: float = 15.0,
                         initial_state=(2.0, 0.0),
                         mu: float = 2.0, omega: float = 1.0
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Generate a clean trajectory of the Van der Pol limit cycle."""
    def rhs(t, state):
        return van_der_pol_rhs(t, state, mu=mu, omega=omega)
    return simulate_ode(rhs, initial_state, n_steps=n_steps, dt=dt, burn_in=burn_in)
