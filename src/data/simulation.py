"""Numerical integration helpers for synthetic dynamical systems."""
from collections.abc import Callable
from typing import Sequence

import numpy as np
from scipy.integrate import solve_ivp


def simulate_ode(
    rhs: Callable,
    initial_state: Sequence[float],
    *,
    n_steps: int = 1000,
    dt: float = 0.02,
    burn_in: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate an ODE and discard an initial transient.

    Parameters
    ----------
    rhs: callable
        SciPy ODE function with signature rhs(t, state).
    initial_state: sequence
        Starting point in state space.
    n_steps: int
        Number of *retained* observations.
    dt: float
        Time between observations (not the adaptive internal solver step).
    burn_in: float
        Transient duration, which must be an integer multiple of dt.

    Returns
    -------
    (time, states)
        Time coordinates (shape: [n_steps]) and clean state trajectory
        (shape: [n_steps, n_dimensions]).
    """
    if n_steps < 2:
        raise ValueError("n_steps must be at least 2")
    if dt <= 0 or burn_in < 0:
        raise ValueError("dt must be positive and burn_in must be non-negative")
    transient_steps = int(round(burn_in / dt))
    if not np.isclose(burn_in, transient_steps * dt, rtol=0, atol=1e-9):
        raise ValueError("burn_in must be an integer multiple of dt")

    initial_state = np.asarray(initial_state, dtype=float)
    if initial_state.ndim != 1 or not np.isfinite(initial_state).all():
        raise ValueError("initial_state must be a finite one-dimensional vector")

    t_eval = dt * np.arange(transient_steps + n_steps, dtype=float)
    solution = solve_ivp(
        rhs,
        t_span=(0.0, float(t_eval[-1])),
        y0=initial_state,
        t_eval=t_eval,
        method="DOP853",
        rtol=1e-9,
        atol=1e-11,
    )
    if not solution.success:
        raise RuntimeError(f"Integration failed: {solution.message}")
    time = t_eval[transient_steps:] - t_eval[transient_steps]
    states = solution.y.T[transient_steps:]
    if not np.isfinite(states).all():
        raise RuntimeError("Integration produced non-finite values")
    return time, states
