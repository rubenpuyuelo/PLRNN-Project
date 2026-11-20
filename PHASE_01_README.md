# Phase 1 — Synthetic Data Generation

**Team 15 — Dynamical Systems in Machine Learning**

## Aim
Generate and visualize trajectories from the Lorenz system (chaotic attractor)
and Van der Pol oscillator (stable limit cycle). Add reproducible Gaussian
**observation** noise, standardize each channel using the observed *training*
segment only, and save the results for subsequent LDS/PLRNN experiments.

## Setup

Use Python 3.10 or newer.

From the repository root (the folder containing `src/` and `notebooks/`):

```bash
python -m pip install -r requirements-phase1.txt
python -m unittest discover -s tests -v
jupyter notebook notebooks/01_data_generation.ipynb
```

Alternatively, in VS Code open `notebooks/01_data_generation.ipynb` and run the cells
with a Python environment containing the above dependencies.

## Files generated on running the notebook

- `data/synthetic/lorenz_seed42.npz` (+ `.json`)
- `data/synthetic/van_der_pol_seed42.npz` (+ `.json`)
- `results/figures/lorenz_attractor.png`
- `results/figures/van_der_pol_limit_cycle.png`
- `results/figures/time_series.png`

Each NPZ contains `time`, `x_true` (noise-free standardized states), `x_obs`
(noisy standardized observations), `train_mean`, `train_std`, and `train_end`.
All trajectory arrays use the format `[time_steps, state_dimensions]`.

## Paper fidelity and deliberate differences

The mathematical systems and primary parameters match the Lorenz and Van der Pol
benchmarks in Fig. 4 of Koppe et al. (2019):
- Lorenz: sigma=10, rho=28, beta=8/3.
- Van der Pol: mu=2, omega=1.

Numerical integration uses SciPy `solve_ivp` with DOP853, an adaptive Runge–Kutta
method. The **sampling intervals, burn-in durations, initial conditions, and
observation-noise level are explicit project choices**; they should not be
presented as exact settings from the paper.

The original study used **process noise** in its benchmark experiments, while
this first phase adds **observation noise** after generating deterministic
trajectories. A later experiment should distinguish these noise sources and
replicate the paper's stochastic setting more closely.

The generated datasets are suitable to test code and begin model comparison, but
should not be used to claim reproduction of the article's quantitative results.

## Next phase

Use the same NPZ datasets to fit a linear baseline and the PLRNN with matching
latent dimensions, equivalent data splits, and documented random seeds. Evaluate
both short-horizon prediction and independent long-run trajectory generation.
