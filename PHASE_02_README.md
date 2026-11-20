# Phase 2 — Linear Dynamical System (LDS-SSM)

**Dynamical Systems in Machine Learning — Team 15**

## Why this phase exists

We need a mathematically meaningful *linear state-space baseline* before
implementing the nonlinear PLRNN-SSM. A linear dynamical model can fit short-term
trajectories but cannot, in its autonomous deterministic stable version, generate
a nonlinear limit cycle or chaotic attractor. We will quantify this limitation
using the **same Phase-1 time series** in Phase 3.

## What has been implemented

The model is linear and Gaussian, with an actual **latent state**:

\[
  z_t = F z_{t-1} + h + \varepsilon_t,
  \qquad \varepsilon_t \sim \mathcal{N}(0,Q)
\]

\[
  x_t = B z_t + \eta_t,
  \qquad \eta_t \sim \mathcal{N}(0,R).
\]

- `F` (`M × M`): full linear transition matrix, equivalent to `A+W` when the
  nonlinear function in the paper is replaced by the identity (`A` diagonal,
  `W` with zero diagonal).
- `h` (`M`): transition bias.
- `B` (`D × M`): observation matrix.
- `Q` (`M` diagonal variances): process noise.
- `R` (`D` diagonal variances): observation noise.

**Kalman filter:** computes p(z_t | x_{1:t}) in forward time.

**RTS smoother:** computes p(z_t | x_{1:T}) and the lag-one covariances required
for full EM sufficient statistics.

**EM:** alternates smoothing latent states and estimating F/h/B/Q/R in closed
form. The initialization uses PCA, and the initial prior covariance is fixed
to avoid degeneracy from a single initial observation. An optional stability
projection `rho(F) <= 0.995`, plus small covariance floors, is used. Hence this
is **constrained/generalized EM**; the likelihood need not increase at every
step. In the included runs it increased.

**Important distinction:** this is our independently implemented, fully
linear-Gaussian LDS-SSM reference, **not** a reproduction of the paper's
PLRNN-SSM annealing inference or MATLAB implementation.

## What to copy into your existing repository

Download/extract `PLRNN_Phase2_LDS.zip` and copy its **contents** to the root of
your Phase-1 repository. You do **not** need to recreate Phase 1 or create a
new Git repository.

New files include:

- `src/models/lds.py` — model, filter, smoother, EM, forecasts, generation
- `src/training/run_lds.py` — experiment runner for both synthetic systems
- `notebooks/02_lds_baseline.ipynb` — explained Colab/Jupyter notebook in English
- `tests/test_lds.py` — mathematical numerical tests and integration checks
- `requirements-phase2.txt` — phase-2 Python environment requirements
- `results/figures/lds_*.png` — example output plots, reproduced by the runner
- `results/metrics/lds_*.json` — example metrics and `.npz` model snapshots
- `PHASE_02_README.md` — this document

**Existing Phase-1 files must remain at their original paths.** The runner
looks for `data/synthetic/van_der_pol_seed42.npz` and
`data/synthetic/lorenz_seed42.npz`.

## Local setup on Windows (VS Code / PowerShell)

From the repository root:

```powershell
# Reactivate your existing environment if necessary
.\.venv\Scripts\Activate.ps1

# Phase 1 packages, plus the notebook kernel if needed
python -m pip install -r requirements-phase1.txt -r requirements-phase2.txt

# Check both previous and new functionality
python -m unittest discover -s tests -v

# Train and save results
python -m src.training.run_lds
```

Then open `notebooks/02_lds_baseline.ipynb` in VS Code, select `.venv`
and execute all cells. The notebook runs the same experiments and displays
figures. The CLI is optional if you prefer the notebook, and vice versa.

If you wish to try fewer EM updates for a quick smoke test:

```powershell
python -m src.training.run_lds --iterations 5
```

## Output files

`results/figures/`:
- `lds_van_der_pol.png`
- `lds_lorenz.png`
- `lds_forecast_mse.png`
- `lds_em_learning_curve.png`

`results/metrics/`:
- `lds_van_der_pol.json` + `lds_van_der_pol_model.npz`
- `lds_lorenz.json` + `lds_lorenz_model.npz`

The `.json` files record evaluated metrics, model parameters, and conditions.
The `.npz` files include matrices, EM objective, and autonomous trajectories
needed for the future PLRNN comparison. These files are small enough to keep
tracked in Git. The `.npz` model files are **not** PyTorch `.pt` checkpoints.

## Experimental protocol and leakage safeguards

Each Phase-1 dataset has 1000 rows with `train_end=700`. Fitting uses only the
**first 700 observations**, including PCA initialization. Phase-1 scaling also
used only the first 700 observations.

For forecast horizon `h`, a causal Kalman filter uses measurements up to an
anchor time `t` (`t >= train_end-1`) to estimate `z_t`; the model then evolves
for `h` steps **without using future observations**. We calculate the MSE
against `x_obs[t+h]` in the held-out segment. This evaluates repeated
*rolling-origin forecasts*, not independent 300-step free generation.

A simple *persistence* benchmark predicts `x[t+h] = x[t]` with identical
anchor times and targets. It provides helpful context: the LDS may perform
worse than persistence, notably on the chaotic Lorenz series.

Separately, **autonomous generation** starts from the final filtered latent
state in the TRAIN segment; then only the fitted `F` and `h` are used. The
generated observations are `B @ z`. No test measurement is ever supplied to
this trajectory. Plots compare the geometry of this generated trajectory with
the true test trajectory using **shared axes**. Pointwise agreement is neither
expected nor a valid metric for chaotic long-run reconstruction.

## Results of verified example run (NumPy/SciPy, seed 42, 35 EM steps)

| System | Horizon 1 | Horizon 5 | Horizon 10 | Horizon 20 |
|---|---:|---:|---:|---:|
| van der Pol / LDS MSE | 0.0236 | 0.1308 | 0.3131 | 0.4560 |
| van der Pol / Persistence MSE | 0.0273 | 0.1966 | 0.5714 | 1.1532 |
| Lorenz / LDS MSE | 0.0590 | 0.9315 | 2.9486 | 5.5368 |
| Lorenz / Persistence MSE | 0.0531 | 0.7411 | 2.0920 | 3.2890 |

These are experimental results from this starter implementation, not numbers
reported in Koppe et al. (2019). Small variations are possible across numerical
library versions, though code and random seeds are fixed.

## How to interpret the results

- A rising forecast MSE does **not** prove by itself that a dynamical system
  was poorly reconstructed; it is important to inspect independent autonomous
  generations and their attractor geometry.
- The LDS `F` has eigenvalues inside the unit circle, so its deterministic
  autonomous behavior converges to a fixed point. It cannot reproduce the
  original van der Pol limit cycle or the Lorenz chaotic attractor.
- A *stochastic* linear Gaussian system with `Q > 0` can keep fluctuating.
  Those fluctuations should not be confused with a self-sustained nonlinear
  periodic orbit or a chaotic attractor.
- We have **not** calculated KL divergence or Lyapunov exponents in Phase 2.
  We will add comparable long-run metrics for the LDS and PLRNN in Phase 3.
- We used observed dimensions equal to latent dimensions (2 and 3) in this
  first baseline. This choice is explicit and will be revisited when we select
  comparable dimensions for both models.

## Source

Koppe, G. et al. (2019). *Identifying nonlinear dynamical systems via
generative recurrent neural networks with applications to fMRI*.
PLOS Computational Biology, 15(8), e1007263.
https://doi.org/10.1371/journal.pcbi.1007263

See equation (1)-(2), figures 4 and 5, and Algorithm 1 on page 26. The LDS-SSM
is obtained by replacing the ReLU nonlinearity in the original model with the
identity mapping (page 10). The current code is an independent Py/NumPy/SciPy
baseline, not the paper's full likelihood and annealing protocol.
