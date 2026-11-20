# Phase 4 — Independent Attractor Reconstruction, Multiseed Robustness, and Noise

This is an **incremental, additive update** to the Phase 1–3 project. Copy the **contents** of the ZIP to the existing repository root, keeping the same folders. It does not overwrite your earlier data, the LDS, or the PLRNN implementations. Work directly on `main` if you prefer; no feature branches are required.

## 1. Research questions

1. Does a PLRNN that gives better short-horizon forecasts also recover the **unforced long-run system dynamics**?
2. Do free-running results depend on the **training seed** or the **training-only initial latent state**?
3. Is the PLRNN still better than a linear model with **the same latent dimension**?
4. What happens when the **observation-noise level** is increased?

The first three are primary experimental tests; the fourth is an exploratory **one-seed** noise ablation.

## 2. What this phase does (and does not do)

- Starts from **independent, clean, newly generated** Lorenz/van der Pol trajectories, standardized with the *original training* means and standard deviations.
- Reuses the saved Phase 3 PLRNN model for seed 42. Retrains two PLRNN models per system with seeds 7 and 21, using the same MAP schedule/budget and the original 700-sample training slice. There is no test-based selection of the best seed.
- For each PLRNN seed, initiates **three separate deterministic free-running simulations** from states inferred **only from the training data**. Every attempt, including a failure, is recorded.
- Fits a fresh **latent-dimension-matched** LDS with `M=8` for van der Pol and `M=12` for Lorenz, on the same 700 training samples (Kalman EM, 25 iterations). This does **not** match parameter counts or inference methods, so it is an improved but not perfectly controlled comparison.
- Generates 2,400 steps after a 400-step burn-in, for each long-run model experiment. Uses identical observation-space coordinates and shared plotting axes, as well as raw smoothed histogram KL and Jensen–Shannon divergence (JSD) at widths 0.5, 1.0, and 1.5.
- Checks near-fixed-point convergence and finite-time tangent growth (maximal Lyapunov estimate). For van der Pol, additionally evaluates spectral peak and spectral overlap.
- For the exploratory noise test, retrains with noise levels **0, 0.1, and 0.3** times the clean training standard deviation; seed 21, same underlying physical trajectory, same training budget for all levels. **Only one seed per noise level: not a statistical significance claim**.
- Creates tables/figures and a ready-to-run English explanatory notebook.

**Important methodological distinction:** the existing PLRNN is a **joint-MAP / alternating-regression approximation**, *not* the original 2019 paper's posterior Laplace-EM. This phase **evaluates its limitations and robustness**; it does not claim to have fully solved the attractor-reconstruction problem or invented an improved EM estimator.

## 3. Run in VS Code (Windows PowerShell)

At the **repository root**:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-phase1.txt -r requirements-phase2.txt -r requirements-phase3.txt
python -m unittest discover -s tests -v
python -m src.training.run_robustness
```

No additional packages are needed beyond earlier phases.

- **Complete experiment** above: 3 seeds, 3 restarts per seed, long trajectories, matched LDS, noise ablation; outputs under `results/phase4/`.
- **Quick smoke test (does NOT reproduce the included full metrics):** `python -m src.training.run_robustness --quick` (writes `results/phase4_quick/`, leaving full outputs untouched)
- **Skip noise test:** `python -m src.training.run_robustness --skip-noise`
- **Visualize without retraining:** open `notebooks/04_robustness_analysis.ipynb`, select `.venv` kernel and Run All. `RUN_EXPERIMENTS=False` by default, and all full outputs have been included in the ZIP. Change to `True` to rerun the full experiment.

The experiment reads, but **never overwrites**, Phase-1 datasets or Phase-3 checkpoints. It writes its models and all metrics under `results/phase4/`. The completed full run produced **23 passing unit tests** (17 previous + 6 new).

## 4. What is inside

```text
src/evaluation/robustness.py              long-run metrics, spectral and Lyapunov diagnostics
src/training/run_robustness.py             full multiseed, matched-LDS, noise experiments
notebooks/04_robustness_analysis.ipynb    runnable, readable scientific analysis (English)
tests/test_robustness.py                  6 new unit tests
PHASE_04_README.md                       this technical guide
results/phase4/figures/                 four generated figures
results/phase4/metrics/                 per-system run data + summary + noise data
results/phase4/models/                  seed 7/21 fitted MAP checkpoints (seed 42 remains in Phase 3)
```

## 5. Full-run outcomes (for report discussion)

All values below are **descriptive** from this program and this fixed dataset and training design, and **not** exact figures from Koppe et al. (2019).

| System | Model | Stable, bounded runs | Median long-run JSD (bin=1) | Near-fixed-point runs |
|---|---|---:|---:|---:|
| van der Pol | PLRNN, M=8 | 9/9 | 0.4312 | 3/9 |
| van der Pol | LDS, M=8 | 3/3 | 0.4996 | 3/3 |
| Lorenz | PLRNN, M=12 | 9/9 | 0.4218 | 0/9 |
| Lorenz | LDS, M=12 | 3/3 | 0.5891 | 3/3 |

`Stable` here means *bounded and finite*, not *correct dynamics*. Lorenz PLRNN tends to a much simpler loop than the butterfly; it **does not recover the chaotic attractor**. Van der Pol PLRNN can sustain a deformed limit cycle, but is sensitive to initial state and seed. The dimension-matched LDS converges to a fixed point.

The finite-sample reference-versus-reference JSD (calibration, not model performance) is approximately **0.0002** for van der Pol and **0.0654** for Lorenz. Lower JSD is better; comparing these calibrations to the reconstructed model values demonstrates that current reconstruction quality is not yet close to the reference.

### Observation-noise ablation (seed 21, single initialization)

| System | Noise standard deviation / clean train std | Autonomous JSD (bin=1) | 5-step held-out MSE |
|---|---:|---:|---:|
| van der Pol | 0.0 | 0.1611 | 0.0265 |
| van der Pol | 0.1 | 0.2257 | 0.0644 |
| van der Pol | 0.3 | 0.3114 | 0.2327 |
| Lorenz | 0.0 | 0.3740 | 0.2244 |
| Lorenz | 0.1 | 0.4228 | 0.2478 |
| Lorenz | 0.3 | 0.6753 | 0.3362 |

The trend is interesting but **must not be treated as significant without repeating seeds/noise realizations**.

## 6. Limitations and research integrity

- Do not label our raw histogram KL as **normalized KL_x** from Koppe et al. (2019): the latter has a specific normalization. We show different bin widths so the sensitivity is visible.
- Our MAP PLRNN does not estimate full posterior covariances; the test-time recognition encoder is not the paper's exact E-step.
- The 9 PLRNN runs per system are **3 seeds × 3 initial states**, not 9 independent fitted networks. These are small descriptive samples, not a statistical test.
- Latent dimensions are matched between the main Phase-4 PLRNN and LDS, but the **parameterizations and number of parameters** still differ.
- `finite_time_lyapunov` is a finite-orbit diagnostic. The Lorenz reference value depends on how long and how accurately the tangent dynamics is integrated, and should not be promoted as a precise estimated ground-truth exponent.
- A prediction MSE improvement (5 steps) does not prove that the long-run attractor has been reconstructed. That contrast is one of our central research findings.
- Data in this stage use **Gaussian observation noise**, not the paper's added **process noise**. The fMRI observation HRF model is not part of this phase.
- Do **not** select a seed from these reported held-out metrics and then report that same selected held-out result as unbiased performance. All results are provided unfiltered; future tuning should use a separate train/validation split.

## 7. GitHub

Review `git status` and your `.gitignore` first. If you ignore `*.npz` or the `results` folder, the generated models/figures will remain local unless explicitly added. Don't upload `.venv/`.

```powershell
git status
git add .
git commit -m "Complete phase 4: multiseed dynamical robustness evaluation"
git push
```

## 8. What's next

An actual **dynamical-reconstruction improvement** could be a dedicated next experiment: replace the simplified MAP approximation with a more faithful Laplace-EM approximation or introduce a train-only validation procedure that selects transition regularization for free-running stability. Do not claim those improvements have already been implemented.
