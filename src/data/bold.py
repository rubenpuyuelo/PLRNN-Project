"""Synthetic BOLD observation simulator for Phase 6 (NOT actual human fMRI).

The Van der Pol and Lorenz phase-1 standardized clean trajectories are used only
as illustrative neural-activity *proxies*. They are not sampled neural signals.
Their ODE time step is not a physiological TR: here one proxy sample corresponds
to one **illustrative** BOLD observation index (default TR = 1.0 s).

A causal canonical double-gamma hemodynamic response function (HRF) convolves
with the proxy and a linear spatial mixing matrix; Gaussian measurement noise is
added. The BOLD normalization is learned strictly from the training segment.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy.stats import gamma
from scipy.signal import lfilter


def canonical_hrf(tr: float = 1.0, duration: float = 32.0,
                  peak: float = 6.0, undershoot: float = 16.0,
                  undershoot_scale: float = 1.0 / 6.0) -> np.ndarray:
    """Discretized causal double-gamma response, unit DC gain.

    A pedagogical HRF parameterization using gamma(shape=peak, scale=1), etc.
    It is not an exact SPM implementation (which uses additional details).
    """
    if not (np.isfinite(tr) and np.isfinite(duration) and
            np.isfinite(peak) and np.isfinite(undershoot) and
            np.isfinite(undershoot_scale)):
        raise ValueError('HRF parameters must be finite')
    if tr <= 0 or duration <= tr or peak <= 0 or undershoot <= peak or undershoot_scale < 0:
        raise ValueError('invalid HRF parameters')
    t = np.arange(0.0, duration, tr, dtype=np.float64)
    if len(t) < 2:
        raise ValueError('HRF must contain at least two samples')
    h = gamma.pdf(t, a=peak) - undershoot_scale * gamma.pdf(t, a=undershoot)
    gain = h.sum()
    if gain <= 1e-10:
        raise ValueError('HRF has zero or negative area')
    return h / gain


def causal_convolution(states: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Causal convolution per channel, [T,D]. No access to future samples."""
    states = np.asarray(states, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    if (states.ndim != 2 or states.shape[0] < 1 or states.shape[1] < 1
            or kernel.ndim != 1 or kernel.size < 1
            or not np.isfinite(states).all() or not np.isfinite(kernel).all()):
        raise ValueError('states [T,D] and kernel [L] must be nonempty and finite')
    return lfilter(kernel, [1.0], states, axis=0)


def build_synthetic_bold(neural_proxy: np.ndarray, train_end: int, *,
                         tr: float = 1.0, noise_std: float = 0.05,
                         seed: int = 42, kernel: np.ndarray | None = None) -> dict:
    """Synthesize standardized BOLD from observed-channel neural proxies.

    Noise scale is relative to per-channel *training* clean-BOLD standard dev.
    True spatial mapping is fixed for reproducibility and invertible for D<=3.
    Normalization of y uses only noisy training values.
    """
    z = np.asarray(neural_proxy, dtype=np.float64)
    if (z.ndim != 2 or len(z) < 100 or z.shape[1] < 1 or z.shape[1] > 3
            or not np.isfinite(z).all()):
        raise ValueError('neural proxy must be finite [T>=100, 1<=D<=3]')
    if not (50 <= train_end <= len(z) - 30):
        raise ValueError('invalid train/test split')
    if np.any(z[:train_end].std(axis=0) < 1e-8):
        raise ValueError('neural proxy has almost constant training channel')
    if not np.isfinite(noise_std) or noise_std < 0:
        raise ValueError('invalid noise_std')
    h = canonical_hrf(tr=tr) if kernel is None else np.asarray(kernel, dtype=float)
    # Fixed, nontrivial mixing to prevent the experiment becoming elementwise scaling.
    mixing = {1: np.array([[1.0]]),
              2: np.array([[1.0, 0.25], [-0.15, 0.95]]),
              3: np.array([[1.0, 0.20, 0.00], [-0.15, 0.95, 0.10],
                           [0.08, -0.12, 1.05]])}[z.shape[1]]
    raw_clean = causal_convolution(z, h) @ mixing.T
    scale = raw_clean[:train_end].std(axis=0)
    if (scale < 1e-7).any():
        raise ValueError('negligible variance of training BOLD signal')
    rng = np.random.default_rng(seed)
    raw_obs = raw_clean + rng.normal(size=raw_clean.shape) * (noise_std * scale)
    mean = raw_obs[:train_end].mean(axis=0)
    std = raw_obs[:train_end].std(axis=0)
    if (std < 1e-10).any():
        raise ValueError('negligible variance of observed BOLD signal')
    return {
        'neural_proxy': z.astype('float32'),
        'bold_clean': ((raw_clean - mean) / std).astype('float32'),
        'bold_observed': ((raw_obs - mean) / std).astype('float32'),
        'hrf': h.astype('float64'),
        'mixing_true': mixing.astype('float64'),
        'bold_train_mean': mean,
        'bold_train_std': std,
        'train_end': np.int64(train_end),
        'tr_illustrative': np.float64(tr),
        'noise_std': np.float64(noise_std),
    }


def save_bold(dataset: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **dataset)
