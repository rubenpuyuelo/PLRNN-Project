"""Pedagogical PLRNN-BOLD-SSM approximation for synthetic BOLD observations.

Observed y[t] = B @ (HRF * z)[t] + measurement noise.
Latent z[t+1] = A z[t] + W ReLU(z[t]) + h + process noise.

We adapt the Phase-3 alternating *joint-MAP* optimization by making the
observation objective causal, temporally convolved. This IS NOT the paper's
Laplace-EM, its posterior covariance, nor a faithful reproduction of Koppe's
PLRNN-BOLD-SSM estimator. This model has no nuisance terms or stimuli.

The control model uses kernel [1] and exactly the same code/hyperparameters.
Both use a linear mapping of latent states (the BOLD Eq. 3 form), not the
non-BOLD Eq. 2 ReLU decoder. BOLD-convolved vs delta is the sole distinction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy.signal import lfilter
from src.models.plrnn import PiecewiseLinearSSM, _check_finite
from src.data.bold import causal_convolution


def torch_causal_convolution(Z, h):
    """Differentiable [T,M] causal filtering matching scipy.signal.lfilter."""
    import torch
    import torch.nn.functional as F
    assert Z.ndim == 2
    m = Z.shape[1]
    # Conv1d computes cross-correlation, so invert kernel and left-pad.
    x = Z.T.unsqueeze(0)
    xp = F.pad(x, (h.numel() - 1, 0))
    weights = h.flip(0).reshape(1, 1, -1).repeat(m, 1, 1)
    return F.conv1d(xp, weights, groups=m)[0].T


@dataclass
class PiecewiseLinearBoldSSM(PiecewiseLinearSSM):
    kernel: np.ndarray = field(default_factory=lambda: np.array([1.0]))

    def __post_init__(self):
        self.kernel = np.asarray(self.kernel, dtype=np.float64).copy()
        if (self.kernel.ndim != 1 or len(self.kernel) == 0
                or not np.isfinite(self.kernel).all() or
                abs(self.kernel).sum() < 1e-10):
            raise ValueError('kernel must be finite, 1-D and nonzero')

    def observe(self, z: np.ndarray) -> np.ndarray:
        if self.B is None:
            raise RuntimeError('Fit first')
        z = np.asarray(z, dtype=np.float64)
        if z.ndim != 2 or z.shape[1] != self.n_latent:
            raise ValueError('observed latent path must be [T,M]')
        return causal_convolution(z, self.kernel) @ self.B.T

    def _m_step(self, x: np.ndarray, z: np.ndarray, *, fit_decoder: bool) -> None:
        if fit_decoder:
            v = causal_convolution(z, self.kernel)
            m = z.shape[1]
            self.B = np.linalg.solve(v.T @ v + self.ridge_obs * np.eye(m), v.T @ x).T
        # Original conditional ridge update on PLRNN latent dynamics only.
        super()._m_step(x, z, fit_decoder=False)

    def _update_latents(self, x: np.ndarray, z: np.ndarray, *,
                        process_var: float, steps: int, lr: float) -> np.ndarray:
        import torch
        torch.set_num_threads(min(torch.get_num_threads(), 4))
        X = torch.as_tensor(x, dtype=torch.float64)
        Z = torch.nn.Parameter(torch.as_tensor(z, dtype=torch.float64).clone())
        A = torch.as_tensor(self.A, dtype=torch.float64)
        W = torch.as_tensor(self.W, dtype=torch.float64)
        bias = torch.as_tensor(self.h, dtype=torch.float64)
        B = torch.as_tensor(self.B, dtype=torch.float64)
        k = torch.as_tensor(self.kernel, dtype=torch.float64)
        anchor = torch.as_tensor(z[0].copy(), dtype=torch.float64)
        opt = torch.optim.Adam([Z], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            v = torch_causal_convolution(Z, k)
            yhat = v @ B.T
            next_z = Z[:-1] * A + torch.relu(Z[:-1]) @ W.T + bias
            obs_loss = ((X - yhat)**2).sum() / (len(x) * self.obs_var)
            dyn_loss = ((Z[1:] - next_z)**2).sum() / ((len(x) - 1) * process_var)
            loss = obs_loss + dyn_loss + 0.05 * ((Z[0] - anchor)**2).sum()
            if not torch.isfinite(loss):
                raise FloatingPointError('non-finite BOLD latent-MAP objective')
            loss.backward()
            torch.nn.utils.clip_grad_norm_([Z], 10.0)
            opt.step()
        result = Z.detach().numpy().copy()
        _check_finite(result, 'BOLD latent MAP path')
        return result

    def rollout_bold(self, initial_state: np.ndarray, n_steps: int, *,
                     prefix: np.ndarray | None = None,
                     max_abs: float = 1e5) -> tuple[np.ndarray, np.ndarray]:
        """Generate new latent states and BOLD with past *training* context only.

        prefix should contain only training latent states preceding generation.
        This avoids wrongly resetting HRF convolution to zero at the split.
        """
        z_gen, _ = super().rollout(initial_state, n_steps, max_abs=max_abs)
        length = len(self.kernel) - 1
        if length:
            if prefix is None:
                prev = np.zeros((length, self.n_latent))
            else:
                prev = np.asarray(prefix, dtype=np.float64)
                if prev.ndim != 2 or prev.shape[1] != self.n_latent:
                    raise ValueError('prefix must be [S,M]')
                prev = prev[-length:]
            full = np.concatenate([prev, z_gen], axis=0)
            gen = self.observe(full)[len(prev):]
        else:
            gen = self.observe(z_gen)
        return z_gen, gen

    def save(self, path: str) -> None:
        if self.A is None:
            raise RuntimeError('fit first')
        np.savez_compressed(path, A=self.A, W=self.W, h=self.h, B=self.B,
                            kernel=self.kernel, process_var=self.process_var_,
                            obs_var=self.obs_var, latent_train=self.latent_train_,
                            seed=self.seed)

    @classmethod
    def load(cls, path: str):
        with np.load(path, allow_pickle=False) as data:
            model = cls(n_latent=len(data['A']), seed=int(data['seed']),
                        obs_var=float(data['obs_var']),kernel=data['kernel'])
            for key in ['A', 'W', 'h', 'B']:
                setattr(model, key, data[key].copy())
            model.process_var_ = float(data['process_var'])
            model.latent_train_ = data['latent_train'].copy()
        return model
