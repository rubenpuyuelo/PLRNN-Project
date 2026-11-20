"""Linear-Gaussian state-space model (LDS-SSM), with Kalman EM learning.

    z[t] = F @ z[t-1] + h + eps[t],  eps ~ N(0, diag(Q))
    x[t] = B @ z[t] + eta[t],         eta ~ N(0, diag(R))

This is a fully latent linear-Gaussian baseline in the spirit of the LDS-SSM
comparator in Koppe et al. (2019). It is a deliberately independent Python
implementation; it is NOT their annealed EM procedure or a port of MATLAB code.

Rows are time and columns are observed/latent dimensions throughout.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy.linalg import cho_factor, cho_solve


def _symmetric(a: np.ndarray) -> np.ndarray:
    return 0.5 * (a + a.T)


def _spd_cholesky(a: np.ndarray):
    """Stable Cholesky with very small adaptive numerical jitter, if needed."""
    a = _symmetric(a)
    eye = np.eye(a.shape[0])
    for jitter in (0., 1e-10, 1e-8, 1e-6):
        try:
            return cho_factor(a + jitter * eye, lower=True, check_finite=False)
        except np.linalg.LinAlgError:
            continue
    raise ValueError("Covariance is not positive definite")


def _stabilize(F: np.ndarray, max_radius: float = 0.995) -> np.ndarray:
    """Project by a scalar multiplier if any eigenvalue exceeds max_radius.

    This explicit constraint can break exact unconstrained-EM monotonicity.
    """
    radius = float(np.max(np.abs(np.linalg.eigvals(F))))
    if radius > max_radius:
        return F * (max_radius / radius)
    return F


@dataclass
class KalmanResult:
    filtered_mean: np.ndarray                  # [T, M]
    filtered_cov: np.ndarray                   # [T, M, M]
    predicted_mean: np.ndarray                 # [T, M]
    predicted_cov: np.ndarray                  # [T, M, M]
    log_likelihood: float
    smoothed_mean: np.ndarray | None = None     # [T, M]
    smoothed_cov: np.ndarray | None = None      # [T, M, M]
    cross_cov: np.ndarray | None = None         # [T-1, M, M]; Cov(z[t+1], z[t] | X)


@dataclass
class LinearGaussianSSM:
    n_latent: int
    seed: int = 42
    noise_floor: float = 1e-4
    max_radius: float = 0.995
    F: np.ndarray | None = field(init=False, default=None)
    h: np.ndarray | None = field(init=False, default=None)
    B: np.ndarray | None = field(init=False, default=None)
    Q: np.ndarray | None = field(init=False, default=None)  # diagonal as 1D
    R: np.ndarray | None = field(init=False, default=None)  # diagonal as 1D
    mu0: np.ndarray | None = field(init=False, default=None)
    P0: np.ndarray | None = field(init=False, default=None)
    history_: list[float] = field(init=False, default_factory=list)

    def _check_observations(self, observations: np.ndarray) -> np.ndarray:
        x = np.asarray(observations, dtype=np.float64)
        if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
            raise ValueError("observations must be [T>=2, n_observed>=1]")
        if not np.isfinite(x).all():
            raise ValueError("observations must be finite")
        if self.n_latent < 1 or self.noise_floor <= 0:
            raise ValueError("Invalid n_latent or noise_floor")
        if not (0.0 < self.max_radius < 1.0):
            raise ValueError("max_radius must be in (0,1)")
        if self.B is not None and x.shape[1] != self.B.shape[0]:
            raise ValueError("Input observation dimension differs from fitted model")
        return x

    def initialize(self, observations: np.ndarray) -> "LinearGaussianSSM":
        """Deterministic PCA initialization plus a linear regression for dynamics."""
        x = self._check_observations(observations)
        t, d = x.shape
        m = self.n_latent
        rng = np.random.default_rng(self.seed)
        u, _, vt = np.linalg.svd(x - x.mean(axis=0), full_matrices=False)
        n_pca = min(m, d)
        B = np.zeros((d, m))
        B[:, :n_pca] = vt[:n_pca].T
        # Extra latent axes can start with tiny, independent signal.
        if m > d:
            B[:, d:] = 0.01 * rng.normal(size=(d, m - d))
        z = x @ B
        G = np.linalg.lstsq(
            np.column_stack([z[:-1], np.ones(t-1)]), z[1:], rcond=1e-8
        )[0]
        self.F = _stabilize(G[:m].T, self.max_radius)
        self.h = G[m]
        self.B = B
        resid_lat = z[1:] - z[:-1] @ self.F.T - self.h
        self.Q = np.maximum(resid_lat.var(axis=0), 0.02)
        resid_obs = x - z @ self.B.T
        self.R = np.maximum(resid_obs.var(axis=0), 0.05)
        self.mu0 = z[0].copy()
        self.P0 = np.eye(m)
        self.history_ = []
        return self

    def filter(self, observations: np.ndarray) -> KalmanResult:
        """Causal Kalman filter: each filtered state uses x[0:t+1], never future x."""
        if self.F is None:
            raise RuntimeError("Call initialize() or fit() first")
        x = self._check_observations(observations)
        T = x.shape[0]
        M = self.n_latent
        means_f = np.empty((T, M))
        cov_f = np.empty((T, M, M))
        means_p = np.empty((T, M))
        cov_p = np.empty((T, M, M))
        Q = np.diag(self.Q)
        R = np.diag(self.R)
        eye = np.eye(M)
        ll = 0.
        for i in range(T):
            if i == 0:
                mu = self.mu0.copy()
                P = self.P0.copy()
            else:
                mu = self.F @ means_f[i-1] + self.h
                P = _symmetric(self.F @ cov_f[i-1] @ self.F.T + Q)
            means_p[i] = mu
            cov_p[i] = P
            diff = x[i] - self.B @ mu
            S = _symmetric(self.B @ P @ self.B.T + R)
            cf = _spd_cholesky(S)
            # K = P B^T S^-1, avoiding matrix inverses.
            K = cho_solve(cf, self.B @ P, check_finite=False).T
            means_f[i] = mu + K @ diff
            remainder = eye - K @ self.B
            cov_f[i] = _symmetric(remainder @ P @ remainder.T + K @ R @ K.T)
            ll -= 0.5 * (len(diff) * np.log(2 * np.pi)
                         + 2 * np.sum(np.log(np.diag(cf[0])))
                         + diff @ cho_solve(cf, diff, check_finite=False))
        return KalmanResult(means_f, cov_f, means_p, cov_p, float(ll))

    def smooth(self, observations: np.ndarray) -> KalmanResult:
        """Rauch–Tung–Striebel smoothing, with exact lag-one covariances."""
        out = self.filter(observations)
        T, M = out.filtered_mean.shape
        sm = out.filtered_mean.copy()
        sc = out.filtered_cov.copy()
        cross = np.empty((T-1, M, M))
        gain = np.empty((T-1, M, M))
        for t in range(T-2, -1, -1):
            # J_t = P_f[t] F.T (P_pred[t+1])^{-1}
            cf = _spd_cholesky(out.predicted_cov[t+1])
            J = cho_solve(cf, self.F @ out.filtered_cov[t], check_finite=False).T
            gain[t] = J
            sm[t] += J @ (sm[t+1] - out.predicted_mean[t+1])
            sc[t] = _symmetric(sc[t] + J @ (sc[t+1] - out.predicted_cov[t+1]) @ J.T)
        for t in range(1, T):
            # For a linear-Gaussian Markov chain, Cov(z_t,z_{t-1}|X)
            # equals Cov(z_t|X) J_{t-1}^T.
            cross[t-1] = sc[t] @ gain[t-1].T
        out.smoothed_mean = sm
        out.smoothed_cov = sc
        out.cross_cov = cross
        return out

    def _m_step(self, x: np.ndarray, out: KalmanResult) -> None:
        """Closed-form ML update of F, h, B and diagonal Q/R, using EM moments."""
        z = out.smoothed_mean
        cov = out.smoothed_cov
        cross = out.cross_cov
        T, M = z.shape
        # E[zt zt^T] for each t
        zz = cov + np.einsum('ti,tj->tij', z, z)
        # Weighted regression: z_t = F z_{t-1} + h + eps_t
        uu = np.zeros((M+1, M+1))
        uu[:M, :M] = zz[:-1].sum(axis=0)
        uu[:M, M] = z[:-1].sum(axis=0)
        uu[M, :M] = z[:-1].sum(axis=0)
        uu[M, M] = T - 1
        yz = cross + np.einsum('ti,tj->tij', z[1:], z[:-1])
        yu = np.column_stack([yz.sum(axis=0), z[1:].sum(axis=0)])
        G = yu @ np.linalg.pinv(uu, rcond=1e-10)
        new_F = _stabilize(G[:, :M], self.max_radius)
        new_h = G[:, M]
        G[:, :M] = new_F  # use projected coefficients for Q residual
        yy = zz[1:].sum(axis=0)
        Eres = yy - G @ yu.T - yu @ G.T + G @ uu @ G.T
        new_Q = np.maximum(np.diag(_symmetric(Eres)) / (T-1), self.noise_floor)
        # Weighted regression: x_t = B z_t + eta_t (no observation bias)
        xxz = x.T @ z
        sum_zz = zz.sum(axis=0)
        new_B = xxz @ np.linalg.pinv(sum_zz, rcond=1e-10)
        obs_Eres = x.T @ x - new_B @ xxz.T - xxz @ new_B.T + new_B @ sum_zz @ new_B.T
        new_R = np.maximum(np.diag(_symmetric(obs_Eres)) / T, self.noise_floor)
        self.F, self.h, self.B = new_F, new_h, new_B
        self.Q, self.R = new_Q, new_R
        # Fix the initial prior to avoid a degenerate ML covariance from one sample.
        # The first state's posterior mean can be used for more stable initialization.
        self.mu0 = z[0].copy()

    def fit(self, observations: np.ndarray, *, n_iter: int = 30,
            tol: float = 1e-5, verbose: bool = False) -> "LinearGaussianSSM":
        """Fit using the observed TRAINING interval only. Validation data not accepted here.

        As parameter constraints and noise floors are present, the procedure is
        generalized/constrained EM, not guaranteed to increase likelihood every step.
        """
        x = self._check_observations(observations)
        if n_iter < 1:
            raise ValueError("n_iter must be >=1")
        self.initialize(x)
        self.history_ = [self.filter(x).log_likelihood]
        for step in range(n_iter):
            sm = self.smooth(x)
            self._m_step(x, sm)
            ll = self.filter(x).log_likelihood
            self.history_.append(ll)
            if verbose and (step == 0 or (step + 1) % 10 == 0 or step == n_iter - 1):
                print(f"EM step {step+1:2d}: log likelihood={ll: .2f}, spectral radius={self.spectral_radius:.4f}")
            if abs(self.history_[-1] - self.history_[-2]) <= tol * (1 + abs(self.history_[-2])):
                break
        return self

    @property
    def spectral_radius(self) -> float:
        if self.F is None:
            raise RuntimeError("Model not initialized")
        return float(np.max(np.abs(np.linalg.eigvals(self.F))))

    def rollout(self, initial_state: np.ndarray, n_steps: int, *,
                sample_process_noise: bool = False, seed: int = 123) -> tuple[np.ndarray, np.ndarray]:
        """Generate unseen states/observations without conditioning on future data.

        Returns z_gen and x_gen with shape [n_steps, ...]. Output index 0 is
        the *next* state following initial_state.
        """
        if self.F is None:
            raise RuntimeError("Fit or initialize model first")
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        z0 = np.asarray(initial_state, dtype=float)
        if z0.shape != (self.n_latent,):
            raise ValueError("initial_state has incorrect shape")
        rng = np.random.default_rng(seed)
        zs = np.empty((n_steps, self.n_latent))
        for t in range(n_steps):
            z0 = self.F @ z0 + self.h
            if sample_process_noise:
                z0 = z0 + rng.normal(0, np.sqrt(self.Q))
            zs[t] = z0
        return zs, zs @ self.B.T

    def forecast_mse(self, observations: np.ndarray, train_end: int,
                     horizons: tuple[int, ...] = (1, 5, 10, 20)) -> dict[int, float]:
        """Held-out multi-step prediction: causal filtered anchors in test period.

        Uses NO future observations to construct each prediction. The test
        observations up to anchor t are permitted; training ends at train_end.
        """
        x = self._check_observations(observations)
        if train_end < 2 or train_end >= len(x):
            raise ValueError("Invalid train_end")
        filt = self.filter(x)
        result = {}
        for horizon in horizons:
            if horizon < 1 or train_end + horizon > len(x):
                raise ValueError("Invalid horizon")
            # anchors t in [train_end-1, len(x)-horizon-1]; target in TEST
            anchors = filt.filtered_mean[train_end-1:len(x)-horizon]
            future = anchors.copy()
            for _ in range(horizon):
                future = future @ self.F.T + self.h
            pred = future @ self.B.T
            target = x[train_end-1+horizon:]
            result[int(horizon)] = float(np.mean((target-pred)**2))
        return result
