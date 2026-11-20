"""Piecewise-linear recurrent neural-network state-space model (PLRNN-SSM).

Koppe et al. (2019), Eqs. 1-2 (without external stimuli):
    z[t] = A z[t-1] + W relu(z[t-1]) + h + eps[t]
    x[t] = B relu(z[t]) + eta[t]
with diagonal A, zero-diagonal W and Gaussian noise.

This is an independent pedagogical *MAP approximation*: alternation between
optimization of the full latent trajectory Z (a joint MAP E-like step) and
ridge-regularized conditional least-squares parameter updates (M-like steps).
It is NOT the original paper's full Laplace-EM, posterior covariance estimates,
or exact Algorithm 1 annealing procedure. Do not describe it as exact EM.

Shape convention: x[T,N], z[T,M], B[N,M], W[M,M].
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _check_finite(x: np.ndarray, name: str) -> None:
    if not np.isfinite(x).all():
        raise ValueError(f"{name} contains nonfinite values")


@dataclass
class PiecewiseLinearSSM:
    n_latent: int
    seed: int = 42
    obs_var: float = 0.10          # fixed measurement variance in MAP objective
    ridge_dyn: float = 1.0        # stabilizes transition regression
    ridge_obs: float = 1e-2
    A: np.ndarray | None = field(default=None, init=False)   # diag entries [M]
    W: np.ndarray | None = field(default=None, init=False)   # [M,M], diag 0
    h: np.ndarray | None = field(default=None, init=False)   # [M]
    B: np.ndarray | None = field(default=None, init=False)   # [N,M]
    process_var_: float = field(default=1.0, init=False)
    history_: list[dict] = field(default_factory=list, init=False)
    latent_train_: np.ndarray | None = field(default=None, init=False)
    encoder_: np.ndarray | None = field(default=None, init=False)   # [3N+1,M]

    def _validate_x(self, observations: np.ndarray) -> np.ndarray:
        x = np.asarray(observations, dtype=np.float64)
        if x.ndim != 2 or x.shape[0] < 3 or x.shape[1] < 1:
            raise ValueError('observations must have shape [T>=3, N>=1]')
        if self.n_latent < 2 * x.shape[1]:
            raise ValueError('n_latent >= 2*n_observed required by paired lift initialization')
        if self.obs_var <= 0 or self.ridge_dyn < 0 or self.ridge_obs <= 0:
            raise ValueError('variances/ridge penalties must be positive')
        _check_finite(x, 'observations')
        if self.B is not None and self.B.shape[0] != x.shape[1]:
            raise ValueError('inconsistent observed dimensions')
        return x

    def _initialize(self, x: np.ndarray) -> np.ndarray:
        """Lift observations to signed pairs plus randomly mixed latent features.

        This only initializes inference: the fitted states and decoder are learned.
        """
        rng = np.random.default_rng(self.seed)
        T, N = x.shape
        M = self.n_latent
        z = np.zeros((T, M), dtype=np.float64)
        z[:, :N] = x
        z[:, N:2*N] = -x
        if M > 2*N:
            mix = rng.normal(size=(N, M-2*N)) / np.sqrt(N)
            z[:, 2*N:] = 0.6 * np.tanh(x @ mix + rng.normal(scale=0.2, size=M-2*N))
        self.A = np.full(M, 0.9)
        self.W = np.zeros((M, M))
        self.h = np.zeros(M)
        self.B = np.zeros((N, M))
        self.B[:, :N] = np.eye(N)
        self.B[:, N:2*N] = -np.eye(N)
        self.process_var_ = 1.0
        self.history_ = []
        return z

    def step(self, z: np.ndarray) -> np.ndarray:
        """One deterministic PLRNN transition, vectorized over batch shape."""
        if self.A is None:
            raise RuntimeError('Fit the model first')
        z = np.asarray(z, dtype=np.float64)
        if z.shape[-1] != self.n_latent:
            raise ValueError('incorrect latent shape')
        return z * self.A + relu(z) @ self.W.T + self.h

    def observe(self, z: np.ndarray) -> np.ndarray:
        """Nonlinear observation/decoder map (paper Eq. 2)."""
        if self.B is None:
            raise RuntimeError('Fit the model first')
        return relu(np.asarray(z, dtype=np.float64)) @ self.B.T

    def _m_step(self, x: np.ndarray, z: np.ndarray, *, fit_decoder: bool) -> None:
        """Conditional least squares on MAP states, with parameter constraints.

        Not an EM M-step integrating posterior covariance; regressions treat z as fixed.
        """
        T, M = z.shape
        if fit_decoder:
            H = relu(z)
            ridge = self.ridge_obs * np.eye(M)
            self.B = np.linalg.solve(H.T @ H + ridge, H.T @ x).T
        prev, nxt = z[:-1], z[1:]
        phi = relu(prev)
        W = np.zeros((M, M))
        A = np.empty(M)
        h = np.empty(M)
        for i in range(M):
            cols = np.arange(M) != i
            # A_i z_i(t-1) + sum_{j!=i} W_ij relu(z_j(t-1)) + h_i
            X = np.column_stack((prev[:, i], phi[:, cols], np.ones(T-1)))
            pen = np.full(M + 1, self.ridge_dyn)
            pen[0] *= 0.2                # don't over-penalize autoregression
            pen[-1] *= 0.02              # almost unpenalized bias
            beta = np.linalg.solve(X.T @ X + np.diag(pen), X.T @ nxt[:, i])
            A[i] = np.clip(beta[0], -1.05, 1.05)
            W[i, cols] = np.clip(beta[1:-1], -2.0, 2.0)
            h[i] = beta[-1]
        self.A, self.W, self.h = A, W, h

    def _update_latents(self, x: np.ndarray, z: np.ndarray, *, process_var: float,
                        steps: int, lr: float) -> np.ndarray:
        """Optimize full MAP latent path with fixed model params using PyTorch Adam."""
        try:
            import torch
        except ImportError as e:
            raise ImportError('PyTorch required: pip install -r requirements-phase3.txt') from e
        # CPU is fast on these small T x M matrices; torch.float64 matches regression.
        torch.set_num_threads(min(torch.get_num_threads(), 4))
        X = torch.as_tensor(x, dtype=torch.float64)
        Z = torch.nn.Parameter(torch.as_tensor(z, dtype=torch.float64).clone())
        A = torch.as_tensor(self.A, dtype=torch.float64)
        W = torch.as_tensor(self.W, dtype=torch.float64)
        h = torch.as_tensor(self.h, dtype=torch.float64)
        B = torch.as_tensor(self.B, dtype=torch.float64)
        start = torch.as_tensor(z[0].copy(), dtype=torch.float64)
        opt = torch.optim.Adam([Z], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            transformed = torch.relu(Z)
            xhat = transformed @ B.T
            zhat = Z[:-1] * A + transformed[:-1] @ W.T + h
            # MAP negative log posterior per time, up to constants.
            obs_loss = ((X - xhat)**2).sum() / (len(x) * self.obs_var)
            dyn_loss = ((Z[1:] - zhat)**2).sum() / ((len(x)-1) * process_var)
            anchor = 0.05 * ((Z[0] - start)**2).sum()
            loss = obs_loss + dyn_loss + anchor
            loss.backward()
            torch.nn.utils.clip_grad_norm_([Z], max_norm=10.0)
            opt.step()
        result = Z.detach().numpy().copy()
        _check_finite(result, 'MAP latent trajectory')
        return result

    def objective(self, x: np.ndarray, z: np.ndarray, process_var: float) -> dict:
        obs_mse = float(np.mean((x-self.observe(z))**2))
        dyn_mse = float(np.mean((z[1:]-self.step(z[:-1]))**2))
        return {'observation_mse': obs_mse, 'dynamics_mse': dyn_mse,
                'map_data_objective': float(x.shape[1]*obs_mse/self.obs_var
                    + self.n_latent*dyn_mse/process_var)}

    def fit(self, observations: np.ndarray, *,
            schedule: tuple[float, ...] = (1.0, 0.3, 0.10, 0.03),
            outer_iters: int = 5, latent_steps: int = 60,
            lr: float = 0.04, verbose: bool = False) -> 'PiecewiseLinearSSM':
        """Train on observations provided; caller must pass TRAINING subset only.

        Annealing is an *analogue* of increasing latent temporal consistency.
        Avoid training on test observations to keep evaluation honest.
        """
        x = self._validate_x(observations)
        if len(schedule) < 1 or any(q <= 0 for q in schedule):
            raise ValueError('schedule must contain positive process variances')
        if outer_iters < 1 or latent_steps < 1 or lr <= 0:
            raise ValueError('invalid training hyperparameters')
        z = self._initialize(x)
        # Stabilize decoder before optimizing latent states.
        self._m_step(x, z, fit_decoder=True)
        for stage, q in enumerate(schedule):
            for iteration in range(outer_iters):
                z = self._update_latents(x, z, process_var=q, steps=latent_steps, lr=lr)
                # Fix decoder after first two stages to avoid a decoder-only fit.
                self._m_step(x, z, fit_decoder=(stage < 2))
                score = self.objective(x, z, q)
                record = {'stage': stage+1, 'iteration': iteration+1,
                          'process_var': float(q), **score}
                self.history_.append(record)
                if verbose and (iteration == outer_iters-1):
                    print(f'Stage {stage+1}/{len(schedule)}: '
                          f'Q={q:.4f}, observation MSE={score["observation_mse"]:.5f}, '
                          f'dynamics MSE={score["dynamics_mse"]:.5f}', flush=True)
        self.process_var_ = float(schedule[-1])
        self.latent_train_ = z
        self._fit_encoder(x, z)
        return self

    @staticmethod
    def _encoder_features(x: np.ndarray) -> np.ndarray:
        """Causal instantaneous signed-ReLU features used by auxiliary encoder."""
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        return np.column_stack((x, relu(x), relu(-x), np.ones(len(x))))

    def _fit_encoder(self, x: np.ndarray, z: np.ndarray) -> None:
        """Train *auxiliary* recognition map on training observations/latents only.

        This is not part of the 2019 paper's exact Laplace-EM procedure.
        It provides a stable initial estimate for causal held-out forecasting.
        """
        u = self._encoder_features(x)
        reg = 0.1 * np.eye(u.shape[1])
        reg[-1, -1] = 0.001
        self.encoder_ = np.linalg.solve(u.T @ u + reg, u.T @ z)

    def encode(self, x: np.ndarray) -> np.ndarray:
        if self.encoder_ is None:
            raise RuntimeError('fit first')
        return self._encoder_features(x) @ self.encoder_

    def filter(self, observations: np.ndarray, initial_state: np.ndarray | None = None,
               *, process_precision: float = 2.0, recognition_precision: float = 20.0,
               iterations: int = 5) -> np.ndarray:
        """Causal approximate MAP inference for a new observation segment.

        Uses learned instantaneous encoder and current observation; no future
        observations are consulted. Initial_state is the state immediately before
        this segment. This is NOT a Kalman/Laplace posterior mean or paper E-step.
        """
        x = np.asarray(observations, dtype=np.float64)
        if x.ndim != 2 or len(x) < 1 or self.B is None or x.shape[1] != self.B.shape[0]:
            raise ValueError('observations must be nonempty with matching dimensions')
        _check_finite(x, 'observations')
        if self.A is None or self.encoder_ is None:
            raise RuntimeError('Fit model first')
        if min(process_precision, recognition_precision) <= 0 or iterations < 1:
            raise ValueError('invalid local-MAP settings')
        if initial_state is None:
            z = self.encode(x[0])[0]
        else:
            z = np.asarray(initial_state, dtype=np.float64).copy()
        if z.shape != (self.n_latent,):
            raise ValueError('initial_state has wrong shape')
        M = self.n_latent
        states = np.empty((len(x), M))
        for t, measurement in enumerate(x):
            obs_guess = self.encode(measurement)[0]
            prior = self.step(z)
            # The auxiliary recognition prior anchors states to a data-supported
            # manifold, preventing spurious explosion of the pure transition.
            zz = obs_guess.copy()
            for _ in range(iterations):
                H = self.B * (zz > 0)[None, :]
                residual = measurement - self.observe(zz) + H @ zz
                lhs = (H.T @ H) / self.obs_var + (process_precision + recognition_precision) * np.eye(M)
                rhs = H.T @ residual / self.obs_var + process_precision * prior + recognition_precision * obs_guess
                candidate = np.linalg.solve(lhs, rhs)
                zz = 0.5 * zz + 0.5 * candidate
            z = zz
            if not np.isfinite(z).all() or np.max(np.abs(z)) > 1e6:
                raise FloatingPointError('recursive MAP inference diverged')
            states[t] = z
        return states

    def rollout(self, initial_state: np.ndarray, n_steps: int, *,
                sample_process_noise: bool = False, seed: int = 42,
                max_abs: float = 1e5) -> tuple[np.ndarray, np.ndarray]:
        """Independent free-running generation, NO conditioning on true observations."""
        if self.A is None:
            raise RuntimeError('Fit first')
        z = np.asarray(initial_state, dtype=np.float64).copy()
        if z.shape != (self.n_latent,) or n_steps < 1:
            raise ValueError('bad rollout initial state / number of steps')
        rng = np.random.default_rng(seed)
        out = np.empty((n_steps, self.n_latent))
        for t in range(n_steps):
            z = self.step(z)
            if sample_process_noise:
                z = z + rng.normal(scale=np.sqrt(self.process_var_), size=self.n_latent)
            if not np.isfinite(z).all() or np.max(np.abs(z)) > max_abs:
                raise FloatingPointError(f'PLRNN rollout diverged at step {t+1}')
            out[t] = z
        return out, self.observe(out)

    def forecast_mse(self, observations: np.ndarray, train_end: int,
                     horizons: tuple[int, ...] = (1, 5, 10, 20)) -> dict[int, float]:
        """Test-only targets; each anchor state uses observations <=anchor."""
        x = self._validate_x(observations)
        if train_end < 3 or train_end >= len(x):
            raise ValueError('invalid train_end')
        if self.latent_train_ is None:
            raise RuntimeError('Training latent states are missing')
        if len(self.latent_train_) != train_end:
            raise ValueError('train_end must match the fitted training split')
        # Initial anchor is inferred from TRAIN data only. All later anchors use
        # only observations up to that test index, never future observations.
        test_states = self.filter(x[train_end:], initial_state=self.latent_train_[-1])
        states = np.vstack([self.latent_train_[-1], test_states])
        scores = {}
        for h in horizons:
            if h < 1 or train_end + h > len(x):
                raise ValueError('invalid horizon')
            zz = states[:-h].copy()
            for _ in range(h):
                zz = self.step(zz)
            pred = self.observe(zz)
            target = x[train_end-1+h:]
            scores[h] = float(np.mean((pred - target)**2))
        return scores

    def save(self, path: str) -> None:
        if self.A is None:
            raise RuntimeError('fit first')
        np.savez_compressed(path, A=self.A, W=self.W, h=self.h, B=self.B,
                            process_var=self.process_var_, obs_var=self.obs_var,
                            latent_train=self.latent_train_, encoder=self.encoder_, seed=self.seed)

    @classmethod
    def load(cls, path: str) -> 'PiecewiseLinearSSM':
        with np.load(path, allow_pickle=False) as data:
            m = cls(n_latent=len(data['A']), seed=int(data['seed']),
                    obs_var=float(data['obs_var']))
            m.A, m.W, m.h, m.B = [data[k].copy() for k in ['A','W','h','B']]
            m.process_var_ = float(data['process_var'])
            m.latent_train_ = data['latent_train'].copy()
            m.encoder_ = data['encoder'].copy()
        return m
