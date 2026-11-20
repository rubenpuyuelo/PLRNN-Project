"""Phase 4: strictly held-out, long-run tests of reconstructed dynamics.

This module does *not* modify or train any model. Independent clean benchmark
reference trajectories and free-running model trajectories are compared in the
same training-standardized observation coordinates.

WARNING: empirical histogram KL/JSD depends on bins, sample sizes and
observational coordinates. None is the paper's normalized KL_x; Lyapunov
estimators are finite-time numerical diagnostics, not chaos certificates.
"""
from __future__ import annotations

from typing import Callable
import numpy as np
from scipy.signal import welch

from src.evaluation.dynamics import occupancy_metrics


def finite_difference_spectrum(signal: np.ndarray, dt: float, *, nperseg: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD; input is a single observed scalar timeseries."""
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1 or len(x) < 32 or dt <= 0 or not np.isfinite(x).all():
        raise ValueError('signal must be >=32 finite samples and dt must be positive')
    f, psd = welch(x, fs=1 / dt, nperseg=min(nperseg, len(x)), detrend='constant')
    return f, psd


def spectrum_stats(reference: np.ndarray, generated: np.ndarray, dt: float) -> dict[str, float | None]:
    """Frequency characteristics of the first observed dimension (van der Pol).

    Frequency resolution is limited; near-constant generated signals are
    reported as no oscillation instead of a fictitious peak at the first bin.
    """
    ref = np.asarray(reference)
    gen = np.asarray(generated)
    if ref.ndim != 2 or gen.ndim != 2 or ref.shape[1] != gen.shape[1]:
        raise ValueError('reference and generated require same observed dimension')
    f, p = finite_difference_spectrum(ref[:, 0], dt)
    fg, q = finite_difference_spectrum(gen[:, 0], dt)
    if len(f) != len(fg) or not np.allclose(f, fg):
        raise ValueError('samples must have same length for comparable Welch bins')
    k0 = int(np.argmax(p[1:]) + 1)
    status = bool(np.std(gen[:, 0]) > 1e-3 * max(np.std(ref[:, 0]), 1e-12)
                  and np.sum(q[1:]) > 1e-10)
    peak = float(f[np.argmax(q[1:]) + 1]) if status else None
    pn, qn = p[1:] / max(np.sum(p[1:]), 1e-20), q[1:] / max(np.sum(q[1:]), 1e-20)
    overlap = float(np.sum(np.minimum(pn, qn))) if status else 0.0
    return {
        'reference_peak_frequency': float(f[k0]),
        'generated_peak_frequency': peak,
        'relative_peak_error': float(abs(peak - f[k0]) / f[k0]) if peak is not None and f[k0] > 0 else None,
        'normalized_spectrum_overlap': overlap,
        'sustained_oscillation': status,
    }


def lyapunov_plrnn(model, states: np.ndarray, dt: float, *, seed: int = 1984) -> float:
    """Finite-time max tangent growth of the deterministic PLRNN map per physical time.

    J(z) = diag(A) + W @ diag(1_{z>0}).  Uses states BEFORE each transition.
    No process noise, no numerical extrapolation or trajectory perturbation.
    """
    z = np.asarray(states, dtype=np.float64)
    if z.ndim != 2 or z.shape[1] != model.n_latent or len(z) < 10 or dt <= 0:
        raise ValueError('invalid trajectory/dt')
    rng = np.random.default_rng(seed)
    v = rng.normal(size=model.n_latent)
    v /= np.linalg.norm(v)
    total = 0.0
    for row in z:
        # W @ (diag(gate) @ v) == W @ (gate*v)
        v = model.A * v + model.W @ ((row > 0) * v)
        s = np.linalg.norm(v)
        if s == 0.0 or not np.isfinite(s):
            return float('-inf')
        total += np.log(s)
        v /= s
    return float(total / (len(z) * dt))


def lyapunov_lorenz_approx(states_unstandardized: np.ndarray, dt: float, *, seed: int = 1984) -> float:
    """Finite-time tangent-growth approximation for Lorenz ODE, for orientation.

    Midpoint-like truncated matrix exponential applied along sampled ODE states;
    only an estimate, not an exact reference exponent.
    """
    x = np.asarray(states_unstandardized, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 3 or len(x) < 10 or dt <= 0:
        raise ValueError('invalid physical Lorenz states')
    rng = np.random.default_rng(seed)
    v = rng.normal(size=3)
    v /= np.linalg.norm(v)
    total = 0.0
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    for a, b, c in x:
        J = np.array([[-sigma, sigma, 0.0], [rho-c, -1.0, -a], [b, a, -beta]])
        # Second-order expansion for small sampling step (dt = 0.02).
        u = J @ v
        v = v + dt*u + 0.5*dt**2*(J@u)
        s = np.linalg.norm(v)
        if s == 0 or not np.isfinite(s):
            return float('-inf')
        total += np.log(s)
        v /= s
    return float(total/(len(x)*dt))


def evaluate_trajectory(reference: np.ndarray, generated: np.ndarray,
                        *, dt: float, system: str) -> dict:
    """Distribution and physical-dynamics metrics for same-length [T,N] runs."""
    ref = np.asarray(reference, dtype=np.float64)
    gen = np.asarray(generated, dtype=np.float64)
    if ref.shape != gen.shape or ref.ndim != 2 or len(ref) < 32:
        raise ValueError('reference and generation must be same shape [T>=32,N]')
    if not np.isfinite(ref).all() or not np.isfinite(gen).all():
        raise ValueError('nonfinite trajectory')
    widths = (0.5, 1.0, 1.5)
    metrics = {}
    for w in widths:
        m = occupancy_metrics(ref, gen, width=w)
        key = str(w)
        metrics[f'js_width_{key}'] = m['histogram_js']
        metrics[f'kl_raw_width_{key}'] = m['histogram_kl_raw']
    std_ref = np.std(ref, axis=0)
    std_gen = np.std(gen, axis=0)
    ratio = std_gen / np.maximum(std_ref, 1e-10)
    metrics['std_ratio_by_dimension'] = ratio.tolist()
    metrics['mean_std_ratio'] = float(np.mean(ratio))
    metrics['near_fixed_point'] = bool(np.max(std_gen) < 0.03 * np.max(std_ref))
    metrics['finite_trajectory'] = True
    if system == 'van_der_pol':
        metrics.update(spectrum_stats(ref, gen, dt))
    return metrics


def summarise(values: list[float]) -> dict[str, float | int | None]:
    """Descriptive summary only; no confidence interval claim for tiny n."""
    arr = np.asarray(values, dtype=float)
    good = arr[np.isfinite(arr)]
    return {'n': int(len(good)), 'median': float(np.median(good)) if len(good) else None,
            'min': float(np.min(good)) if len(good) else None,
            'max': float(np.max(good)) if len(good) else None}
