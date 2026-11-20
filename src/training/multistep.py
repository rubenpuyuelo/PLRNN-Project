"""Train-only multistep transition refinement for the existing PLRNN-SSM.

The same latent transition as Koppe et al. (2019) is kept:
  z[t+1] = diag(A) z[t] + W relu(z[t]) + h  (diag(W)=0)
  x[t]   = B relu(z[t])

This optional *post-MAP* refinement uses truncated unrolling from *training*
MAP latent states to fit transition parameters against k-step training
observations. The observation decoder and training recognition map remain fixed.
It is NOT Laplace-EM, nor a guarantee of accurate autonomous chaos.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass, asdict
import numpy as np
from src.models.plrnn import PiecewiseLinearSSM


@dataclass(frozen=True)
class RefinementConfig:
    horizon: int = 10
    iterations: int = 320
    batch_size: int = 96
    learning_rate: float = 0.006
    anchor_strength: float = 0.05
    seed: int = 9105
    gradient_clip: float = 3.0


def multistep_loss(model: PiecewiseLinearSSM, observations: np.ndarray,
                   *, horizon: int, stride: int = 1) -> float:
    """Mean observed k-step MSE across training anchors; no model training.

    Targets are strictly contained in `observations`, which must be TRAIN only.
    Sampled latent anchors are the saved training MAP path.
    """
    x = np.asarray(observations, dtype=float)
    if horizon < 1 or stride < 1 or model.latent_train_ is None:
        raise ValueError('invalid horizon/stride or missing training latents')
    if x.ndim != 2 or len(x) != len(model.latent_train_) or len(x) <= horizon:
        raise ValueError('training observations must match latent_train_')
    initial_idx = np.arange(0, len(x) - horizon, stride)
    z = model.latent_train_[initial_idx].copy()
    terms = []
    for k in range(1, horizon+1):
        z = model.step(z)
        yhat = model.observe(z)
        terms.append(np.mean((yhat - x[initial_idx + k]) ** 2))
        if not np.isfinite(terms[-1]):
            return float('inf')
    return float(np.mean(terms))


def refine_transitions(model: PiecewiseLinearSSM, train_obs: np.ndarray,
                       config: RefinementConfig = RefinementConfig()
                       ) -> tuple[PiecewiseLinearSSM, list[dict]]:
    """Return a DEEP COPY with optimized A, W, h. Input remains unchanged.

    All anchors and all target observations must be from the training subset.
    No generation on the test set is used to train, stop or select parameters.
    """
    import torch
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    if not isinstance(model, PiecewiseLinearSSM) or model.A is None or model.latent_train_ is None:
        raise ValueError('fit a PLRNN-SSM before transition refinement')
    x = np.asarray(train_obs, dtype=np.float64)
    T, N = x.shape
    H = config.horizon
    if (x.ndim != 2 or T != len(model.latent_train_) or N != model.B.shape[0]
            or H < 1 or H >= T or config.iterations < 1 or config.batch_size < 1
            or config.learning_rate <= 0 or config.anchor_strength < 0
            or config.gradient_clip <= 0 or not np.isfinite(x).all()):
        raise ValueError('invalid train-only data or refinement settings')
    device='cpu'
    def ten(arr): return torch.as_tensor(np.asarray(arr),dtype=torch.float64,device=device)
    original = (ten(model.A),ten(model.W),ten(model.h))
    A=torch.nn.Parameter(original[0].clone())
    W=torch.nn.Parameter(original[1].clone())
    h=torch.nn.Parameter(original[2].clone())
    B=ten(model.B)
    Z=ten(model.latent_train_)
    X=ten(x)
    M=model.n_latent
    mask=1-torch.eye(M,dtype=torch.float64,device=device)
    opt=torch.optim.Adam([A,W,h], lr=config.learning_rate)
    rng=np.random.default_rng(config.seed)
    logs=[]
    for step in range(config.iterations):
        indices=rng.integers(0,T-H,size=config.batch_size)
        z=Z[indices].clone()
        obs_loss=torch.zeros((),dtype=torch.float64,device=device)
        for k in range(1,H+1):
            z=z*A + torch.relu(z) @ (W*mask).T + h
            y=torch.relu(z) @ B.T
            obs_loss=obs_loss+torch.mean((y-X[indices+k])**2) / H
        # Anchor the transition to the original MAP estimate to avoid fitting
        # arbitrary high-amplitude long-horizon trajectories.
        reg=(torch.mean((A-original[0])**2)
             +torch.mean(((W-original[1])*mask)**2)
             +torch.mean((h-original[2])**2))
        loss=obs_loss+config.anchor_strength*reg
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([A,W,h],config.gradient_clip)
        opt.step()
        with torch.no_grad():
            # Prevent the numerically explosive coefficient regimes that arise
            # from unconstrained MAP regression, without enforcing contraction.
            A.clamp_(-1.05,1.05)
            W.clamp_(-2.0,2.0)
            W.mul_(mask)
            h.clamp_(-3.0,3.0)
        if not np.isfinite(float(loss.detach())):
            raise FloatingPointError('nonfinite fine-tuning loss')
        if step==0 or (step+1) % max(1,config.iterations//25)==0 or step==config.iterations-1:
            logs.append({'iteration':step+1,'batch_observation_mse':float(obs_loss.detach()),
                         'anchor_regularizer':float(reg.detach())})
    result=deepcopy(model)
    result.A=A.detach().numpy().copy()
    result.W=(W*mask).detach().numpy().copy()
    result.h=h.detach().numpy().copy()
    # The decoder and recognition model are deliberately fixed.
    return result,logs
