"""Simple empirical occupancy metrics for free-running dynamical-system samples.

NOTE: We report raw KL and Jensen-Shannon divergence using smoothed histograms.
They are not the *normalized* KL_x metric from Koppe et al. (2019), Eq. 9.
Both samples must be in the SAME observation coordinates (phase-1 scaling).
"""
from __future__ import annotations
import numpy as np


def occupancy_metrics(reference: np.ndarray, generated: np.ndarray,
                      *, lower: float = -4., upper: float = 4., width: float = 1.,
                      alpha: float = 1e-3) -> dict[str, float]:
    """Shared-bin histogram KL(ref||gen) and JSD, with explicit overflow bins.

    Outside [-4,4] gets an overflow bin rather than being silently discarded.
    Histograms include additive smoothing, alpha per multidimensional bin.
    """
    a = np.asarray(reference, dtype=float)
    b = np.asarray(generated, dtype=float)
    if a.ndim != 2 or b.shape[1:] != a.shape[1:] or len(a) < 1 or len(b) < 1:
        raise ValueError('reference and generated must be nonempty [T,N] with matching N')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('samples must be finite')
    if alpha <= 0 or width <= 0 or upper <= lower:
        raise ValueError('bad histogram settings')
    ticks = np.arange(lower, upper + width*0.5, width)
    edges = np.concatenate(([-np.inf], ticks, [np.inf]))
    p = np.histogramdd(a, bins=[edges]*a.shape[1])[0].ravel() + alpha
    q = np.histogramdd(b, bins=[edges]*b.shape[1])[0].ravel() + alpha
    p /= p.sum(); q /= q.sum()
    m = 0.5 * (p+q)
    kl = float(np.sum(p * np.log(p/q)))
    js = float(0.5 * (np.sum(p * np.log(p/m)) + np.sum(q*np.log(q/m))))
    return {'histogram_kl_raw': kl, 'histogram_js': js,
            'n_reference': int(len(a)), 'n_generated': int(len(b)),
            'histogram_width': float(width), 'histogram_alpha': float(alpha)}
