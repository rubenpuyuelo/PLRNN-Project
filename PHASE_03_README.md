# Phase 3 — PLRNN-SSM (nonlinear model + comparison)

This increment **adds files** to the existing Phase 1 + Phase 2 repository. Copy the contents of this folder into the repository **root**, preserving subdirectories. No branches needed; continue on `main`.

## 1. Goals

1. Implement the PLRNN state-space equations from Koppe et al. (2019), Eqs. 1–2, *without* external experimental input (`C s_t`):

   \[ z_t=A z_{t-1}+W\operatorname{ReLU}(z_{t-1})+h+\varepsilon_t, \quad x_t=B\operatorname{ReLU}(z_t)+\eta_t. \]

   `A` is diagonal, `W` is zero-diagonal, and state/observation noises are assumed Gaussian with fixed scalar variances for the MAP objective. This simplification is **not** the paper's full posterior inference.

2. Fit MAP trajectories and model parameters **only on the first 700 training observations** using an alternating procedure and progressively decreasing assumed process variance `(1, 0.3, 0.1, 0.03)`.
3. Compare held-out causal multi-step MSE for horizons 1, 5, 10 and 20 with the *saved* Phase 2 LDS and naive persistence.
4. Compare independent free-running trajectories and state-space occupancy measures. A good forecasting MSE **does not guarantee** attractor recovery.

## 2. Installation (PowerShell in VS Code)

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-phase1.txt -r requirements-phase2.txt -r requirements-phase3.txt
python -m unittest discover -s tests -v
python -m src.training.run_plrnn
```

PyTorch runs on **CPU**: you do not need a GPU. If `torch` is not available for your Python version, use a Python version supported by the official PyTorch distribution and recreate your virtual environment. The other phase files are unchanged.

For a quicker test (not representative of full results):

```powershell
python -m src.training.run_plrnn --quick
```

Then open `notebooks/03_plrnn_training.ipynb` and select the `.venv` kernel. Use **Run All**. The notebook runs the full comparison and displays the resulting figures/metrics. Run the CLI *or* notebook; you don't need to do both to train models.

## 3. Files in this increment

```text
src/models/plrnn.py               PLRNN-SSM and approximate MAP fitting/inference
src/evaluation/dynamics.py        Raw histogram KL and Jensen–Shannon divergences
src/training/run_plrnn.py         Train/compare/plot/save experiments
notebooks/03_plrnn_training.ipynb Detailed explanatory notebook in English
tests/test_plrnn.py              Mathematical/API/causality checks
requirements-phase3.txt          PyTorch (uses phase 1 and phase 2 too)
PHASE_03_README.md               This guide
results/figures/plrnn_*.png      Figures produced by the included full experiment
results/metrics/plrnn_*.json     Measurements and methodological notes
results/metrics/plrnn_*_model.npz Fitted parameter checkpoints (small NumPy files)
```

Input requirements from earlier phases:

- `data/synthetic/lorenz_seed42.npz`, `van_der_pol_seed42.npz`
- `results/metrics/lds_lorenz.json`, `lds_van_der_pol.json`
- `results/metrics/lds_lorenz_model.npz`, `lds_van_der_pol_model.npz`

If missing, rerun `01_data_generation.ipynb` or `python -m src.training.run_lds` as appropriate.

## 4. How the code works

- **Initialization:** a signed-feature lift of each observation, using positive/negative pairs and random mixtures for extra latent dimensions. The additional units are not extra measured variables.
- **Latent MAP update:** with transition/observation matrices fixed, PyTorch optimizes the entire training latent trajectory to reduce squared observation error + squared transition error weighted by inverse variances.
- **Conditional parameter update:** with MAP trajectories fixed, ridge least squares estimates `A`, off-diagonal `W`, `h` and sometimes `B` (decoder fixed after first two stages). **No posterior covariance correction is included.**
- **Progressive schedule:** assumed transition noise variance is decreased stage-wise, increasing pressure for a self-consistent latent dynamical model. This is **inspired by** Algorithm 1 but **not a replication**.
- **Test-time latent inference:** a signed-ReLU feature encoder trained *only on training data* gives a causal guess of each test state, refined by a local MAP correction with the model dynamics and current observed sample. This is an **auxiliary approximation** not found in the original paper. It uses no future test data.
- **Free running:** after the final training state, recursively run the PLRNN **without test observations**. Compare these generated observations with the held-out clean system.

## 5. Scores and experimental limitations

- Forecast MSE: at test anchor `t`, estimate the latent state using observations up to `t`, then predict `t+h` **without observations between t and t+h**. This is causally comparable to Phase 2 in broad terms but their inference algorithms differ.
- KL: smoothed shared-bin **raw** `KL(p_true || p_generated)` with width=1 in each dimension, bins from -4 to +4 plus explicit overflow bins; alpha=0.001. **NOT** Koppe et al.'s *normalized* KLx metric. We also report bounded Jensen–Shannon divergence (JSD).
- Phase 3 PLRNN uses 8 (van der Pol) or 12 (Lorenz) states, while the Phase 2 LDS uses 2 or 3 states, respectively: **not a parameter-matched comparison**. This must be noted in the report; a fair capacity ablation is a later experiment.
- This is **one training seed and a short held-out trajectory**. Scores must not be represented as replicated statistical findings.
- Distinguish good short-horizon prediction from long-run attractor geometry: free trajectories may have periodicity, fixed points, or fail to reproduce full chaoticity. Examine the saved pictures, raw KL/JSD, and eventually Lyapunov exponents and more seeds.
- The original paper's fMRI **BOLD observation model is not implemented** in Phase 3. That belongs to the later optional phase.

## 6. Typical full-run numbers on the supplied datasets (seed=42)

| System | Test MSE h=5: PLRNN / LDS | h=10: PLRNN / LDS | h=20: PLRNN / LDS |
|---|---|---|---|
| van der Pol | 0.0801 / 0.1308 | 0.1445 / 0.3131 | 0.1885 / 0.4560 |
| Lorenz | 0.2648 / 0.9315 | 0.8683 / 2.9486 | 2.9946 / 5.5368 |

These figures refer to the included, actually executed model using the default settings. They are preliminary, and **do not show faithful recreation of the full original attractors**. The strongest next steps are (1) multi-seed robustness and fair-capacity comparison, (2) qualitative and quantitative attractor reconstruction / improved training, and only afterwards (3) fMRI.

## 7. Git on `main`

```powershell
git status
git add .
git commit -m "Implement Phase 3 PLRNN-SSM and LDS comparison"
git push
```

If you have a `.gitignore` that excludes `*.npz` or `results/`, keep your local generated data/models and decide which compact outputs to version. Never commit the `.venv` directory.
