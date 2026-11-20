# Phase 5 — Train-Only, Stability-Aware Multi-Step PLRNN Refinement

This is an **additive update** to Phases 1–4. It does **not** replace any earlier checkpoint, overwrite earlier results, implement the original Laplace-EM algorithm, or imply that the Lorenz attractor has been fully recovered.

## Scientific motivation

The Phase-3 PLRNN is fitted using an alternating joint-MAP objective with one-step latent consistency. Its forecasts are better than the LDS in the measured examples, but long-run autonomous dynamics may be geometrically wrong. The purpose here is to distinguish **improved multi-horizon forecast performance** from **improved autonomous dynamical reconstruction**.

The transition and decoder remain:

\[
 z_{t+1} = A z_t + W\max(z_t,0)+h,\qquad x_t = B\max(z_t,0),
\]

where `A` is diagonal and `W` has zero diagonal. The observation decoder `B` is **fixed** during refinement, as is the auxiliary observation-to-latent encoder. Starting from training-only MAP latent anchors \(z_t\), the new optional step minimizes a truncated 10-step observation loss:

\[
 \mathcal{L}_{multi}=\frac{1}{|\mathcal I|H}\sum_{t\in\mathcal I}\sum_{k=1}^{H}\|B\,\mathrm{ReLU}(F_\theta^{k}(z_t))-x_{t+k}\|_2^2/N
 +\lambda\,\mathcal{R}(\theta,\theta_0).
\]

The regularizer anchors `(A,W,h)` to the pre-refinement parameters. No test data enters the objective.

**Stability-aware selection.** The proposed refined model can yield unbounded autonomous trajectories even when its short-horizon MSE improves. We evaluate the fixed interpolation grid `alpha = [0, .1, .2, .3, .4, .5, .6, .75, .9, 1]`, blending original and refined **transition** parameters. For each candidate, **three training-only inferred initial states** are simulated freely. Candidates with **any unstable rollout** are rejected. Among the remaining candidates, we choose the one with the lowest median *training observation* histogram Jensen–Shannon divergence, with bin width 1.0. The validation here is an **internal training-data diagnostic**, not an independent validation set. This is explicitly an exploratory postprocessing strategy, **not** Koppe et al. Algorithm 1 or a guarantee of chaos.

The test-period MSE, independent *clean* ODE reference trajectory, and independent long-run JSD are computed **only after** this selection. However, because the approach and hyperparameters were developed experimentally on this synthetic setting, treat resulting improvements as **exploratory**; further datasets/seeds would be needed for confirmatory performance claims.

## Installation and use (Windows PowerShell)

Copy the **contents** of the phase-5 ZIP into the root of your existing repository. Keep `data/`, `src/`, `notebooks/`, `results/` and `tests/` in place. Work directly on `main`.

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-phase5.txt
python -m unittest discover -s tests -v
```

Expected when all five phases are installed: **30 tests**, `OK`.

To **view saved results without retraining**, open `notebooks/05_multistep_refinement.ipynb`, select the `.venv` kernel and run all. `RUN_EXPERIMENTS=False` by default.

To reproduce the **full** phase-5 run (3 seeds × 2 systems; baseline checkpoints from Phases 3 and 4 are required):

```powershell
python -m src.training.run_phase5
```

It writes **only** within `results/phase5/`. It does not overwrite previous models or datasets.

Quick smoke test (different results, separate folder):

```powershell
python -m src.training.run_phase5 --quick
```

This writes `results/phase5_quick/`. Do not confuse quick outputs with the included full study.

## Contents

```text
src/training/multistep.py         Train-only multistep fine-tuning and loss
src/training/run_phase5.py        Stability-aware selection, independent evaluation, figures
notebooks/05_multistep_refinement.ipynb
requirements-phase5.txt
PHASE_05_README.md
tests/test_multistep.py           Seven additional tests
results/phase5/metrics/          Detailed JSON + short summary CSV
results/phase5/models/           Six *new* refined checkpoints
results/phase5/figures/          Comparison plots and both attractor visualizations
```

## Full experiment — observed results

All `JSD` numbers below are computed on an independent, clean ODE simulation from a different starting state, after 400 burn-in + 2,400 measured generated steps. Three free trajectories are initialized with *training-only* inferred states per checkpoint. Each row is **one fitted PLRNN checkpoint** and three related initial-state rollouts; it is **not** three independent training runs. Reported JSD is the median of those rollouts.

| System | Seed | Chosen alpha (train only) | Independent JSD before → after | Held-out 10-step MSE before → after |
|---|---:|---:|---:|---:|
| van der Pol | 42 | 0.30 | 0.4313 → **0.0593** | 0.1445 → **0.0923** |
| van der Pol | 7 | 0.20 | 0.4996 → **0.1271** | 0.1290 → **0.1102** |
| van der Pol | 21 | 0.40 | 0.2018 → **0.0488** | 0.1257 → **0.0880** |
| Lorenz | 42 | 0.40 | 0.4583 → **0.3940** | 0.8683 → **0.6753** |
| Lorenz | 7 | 0.00 | 0.4197 → 0.4197 | 0.9980 → 0.9980 |
| Lorenz | 21 | 0.00 | 0.4218 → 0.4218 | 0.9515 → 0.9515 |

**Crucial:** van der Pol improves for all three checkpoints in this experiment, but Lorenz improves for only one checkpoint. The Lorenz reconstructed object remains simplified and should **not** be called a faithful chaotic butterfly. The training-only selector appropriately keeps the original parameters when it cannot find a better stable candidate.

**What this is not:** neither the original 2019 PLRNN Laplace-EM nor its annealing algorithm; not normalized KL from that paper; not a statistically powered benchmark; not a real fMRI experiment. The Phase-1 datasets use **observation** noise, unlike the paper's benchmark process-noise setup.

## How to incorporate results in the academic write-up

Suitable claim: *"An exploratory multi-step post-refinement with train-only, stability-aware selection improved independently evaluated limit-cycle reconstruction for van der Pol across three PLRNN fitting seeds. Lorenz improvements were limited and did not recover its full chaotic attractor, emphasizing the gap between short-horizon prediction and autonomous dynamical fidelity."*

Include both selected and rejected candidates (available in the JSON) and report all seeds rather than just seed 42. Point out that training-geometry-based model selection is a heuristic, not full posterior inference.

## GitHub

Check `git status` and `.gitignore`; the `.npz` checkpoints and generated figures might be excluded by custom ignore rules. Do **not** add `.venv` to Git. No branches are necessary.

```powershell
git status
git add .
git commit -m "Add phase 5: stability-aware multistep PLRNN refinement"
git push
```

## Next stage

The project now has an improved, carefully bounded benchmark result. The next logical step is to implement a **synthetic BOLD / HRF observation extension**, comparing correctly modeled hemodynamic filtering with an intentionally naive observation model, before attempting real fMRI recordings. Do not label synthetic BOLD data as real human fMRI.
