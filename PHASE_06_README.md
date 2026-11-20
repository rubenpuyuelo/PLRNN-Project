# Phase 6 — Synthetic BOLD + HRF-Aware PLRNN MAP Extension

**Course:** Dynamical Systems in Machine Learning  
**Project:** Identifying nonlinear dynamical systems using generative recurrent neural networks  
**Reference:** Koppe et al. (2019), *PLOS Computational Biology*, https://doi.org/10.1371/journal.pcbi.1007263

## Purpose and scientific scope

This is a **synthetic observation-model study**, not an analysis of human fMRI scans. It extends the previous benchmarks by constructing **BOLD-like measurements** via a causal hemodynamic response function (HRF) and comparing a matching HRF-aware observation model against an instantaneous-observation ablation.

The clean, standardized **van der Pol** and **Lorenz** ODE trajectories from Phase 1 serve as **neural-activity proxies**. They are not measured neuronal signals. We intentionally map **one simulated ODE sample to one illustrative fMRI observation index (TR = 1 second)**. This is not a physiologically calibrated mapping of the original ODE integration step, nor does it validate brain interpretation. No genuine scans, experimental task events or motion nuisance regressors are used.

Equation for the pedagogical BOLD-SSM observation:

\[
 y_t = B\left(\sum_{k=0}^{L-1} h_k z_{t-k}\right) + \eta_t,
\qquad z_{t+1} = A z_t + W\max(z_t,0) + b + \epsilon_t.
\]

- `h` is a normalized **double-gamma causal HRF** (`src/data/bold.py`), chosen as a simple approximation, **not exact SPM code**.
- `A` is diagonal, `W` has zero diagonal; it is still a ReLU-based PLRNN latent transition.
- We do not estimate HRF coefficients; they are fixed to the same known kernel used to generate the synthetic data.
- For the control, `h=[1]`, producing the instantaneous `y=Bz` map **without** changing PLRNN architecture, optimizer, data split or parameter count.
- This is **our own alternating joint-MAP approximation**, **not** the original PLRNN-BOLD-SSM Laplace-EM inference in Koppe et al. (2019), which additionally infers posterior covariances and handles nuisance regressors / external stimuli.

## Two deliberately different experiments

### Experiment A: oracle observation-model ablation

Here the **true proxy latent path is supplied as an input** to a train-only linear decoder regression. The only difference is using `HRF * proxy_states` vs `proxy_states` as features. Both decoders fit an intercept and the same number of matrix coefficients. No neural or latent-state inference is being tested. We evaluate against the *clean synthetic BOLD* to isolate observation-model mismatch, and also report scores against the noisy measurements.

These results validate that the chosen HRF describes the data-generation mechanism (not a surprising out-of-sample discovery, since data are generated using that exact HRF).

### Experiment B: exploratory BOLD-SSM latent MAP training

No proxy states are passed to the model at training time. Both models receive **only the first 700 noisy BOLD measurements**, with 300 held out. Training alternates between (i) differentiable latent-path MAP optimization with an observation loss (`B*(HRF*Z)` or `B*Z`) and temporal consistency, and (ii) conditional ridge parameter updates for `A, W, b, B`. The process variance is reduced in stages as in the pedagogical Phase-3 MAP implementation.

Free-running generation uses the final **training** latent state; the HRF-aware model carries training latent history into the future to avoid resetting the filter artificially at the train/test boundary. Generated trajectories do **not** condition on held-out BOLD. The **Jensen–Shannon divergence (JSD)** compares generated vs reference **BOLD observation** occupancies in matched train-standardized coordinates, using the same histogram discretization as in previous phases. JSD is **not the paper's normalized KL**. Unstable rollouts are recorded as **unstable**, with a null JSD, not omitted or treated as zero.

Since Van der Pol and Lorenz have long-range dynamics (chaotic for Lorenz), direct long-run time-series MSE is a weaker generative criterion than comparing distributions. We still record time-series MSE as a diagnostic, without claiming phase matching.

## Installation — Windows PowerShell

Extract the ZIP and copy its **contents into the root of your existing GitHub repository**, preserving `src/`, `tests/`, `notebooks/`, `results/`, `data/`. The package **adds** new files and does not replace any of Phases 1–5.

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-phase6.txt
python -m unittest discover -s tests -v
```

**Validated locally:** 42 tests total across Phases 1–6, all passing.

Open `notebooks/06_synthetic_bold.ipynb` in VS Code and choose the `.venv` Python kernel. `RUN_EXPERIMENTS=False` by default: the notebook displays the **supplied full experiment** without retraining. It has already been fully executed and saved with outputs.

To reproduce the whole experiment from the repository root:

```powershell
python -m src.training.run_phase6
```

For a short **smoke test** (not comparable to the full results):

```powershell
python -m src.training.run_phase6 --quick
```

This writes `results/phase6_quick/` and **does not overwrite the full experiment outputs**. To run only oracle decoders:

```powershell
python -m src.training.run_phase6 --skip-map
```

The scripts do not change earlier datasets/checkpoints. They create/refresh only synthetic BOLD data under `data/bold/` and Phase-6 outputs. If you change the seed, data are saved with the new seed suffix and results are overwritten in the selected Phase-6 metrics folder, so use separate saved copies for formal repeated-seed studies.

## Phase-6 deliverables

```text
src/data/bold.py                     Causal double-gamma HRF, synthetic BOLD data
src/models/bold_ssm.py               HRF-aware/instantaneous PLRNN joint-MAP adaptation
src/training/run_phase6.py           Experiments, metrics and figures
notebooks/06_synthetic_bold.ipynb    Executed, English-language report notebook
requirements-phase6.txt             Inherits Phase 1–5 dependencies, adds pandas
PHASE_06_README.md                   This documentation
tests/test_bold.py                   12 tests for causality, correctness and MAP model
data/bold/*.npz                      Synthetic BOLD datasets (never actual scans)
results/phase6/figures/             6 PNGs, HRF, oracle and MAP for both systems
results/phase6/metrics/             Per-system JSON and one summary JSON
results/phase6/models/              4 trained exploratory MAP checkpoints
```

## Results of the supplied full experiment (seed 42)

**Controlled oracle decoder**, MSE vs *clean* held-out BOLD:

| Benchmark | HRF-aware | Instantaneous |
|---|---:|---:|
| van der Pol | **0.000011** | 0.121078 |
| Lorenz | **0.000083** | 0.550267 |

**PLRNN joint-MAP approximation**, *autonomous* held-out BOLD histogram JSD (lower is better; no cherry-picking):

| Benchmark | HRF-aware | Instantaneous | Interpretation |
|---|---:|---:|---|
| van der Pol | **0.12286** | 0.23773 | HRF-aware improves JSD in this one run |
| Lorenz | **unstable** | 0.57121 | HRF-aware diverges on generation step 288; no valid full-horizon JSD |

**Training-observation MSE** (not a measure of autonomous reconstruction):

| Benchmark | HRF-aware | Instantaneous |
|---|---:|---:|
| van der Pol | 0.001946 | 0.001580 |
| Lorenz | 0.002513 | 0.001364 |

The instantaneous model's smaller training MSE does **not** establish better dynamics. Likewise, the oracle decoder's very low error does **not** demonstrate unsupervised latent recovery.

**Limitations and interpretation:** the full MAP extension uses **one training seed and one observed time series per benchmark**. Lorenz was not successfully reconstructed, and neither model is an implementation of the entire paper. The learned states have no inferred neural/clinical meaning. Treat these results as a **controlled synthetic methodology exercise**, not confirmation that the full PLRNN-BOLD-SSM outperforms an LDS on real fMRI. In particular, do **not** combine this oracle-decoder result with the LDS/PLRNN results from earlier phases as if all were the same inference/evaluation task.

## How to incorporate into the final paper / December progress report

**Suggested subsection:** “Synthetic BOLD Observations: Hemodynamic Filtering and its Impact on PLRNN Reconstruction.” Describe the two separate experiments and the distinction between a known-state decoder and joint latent inference. Include the HRF response, the decoder comparison chart and one autonomous-generation comparison. State that a biologically correct observation model is helpful but does not remove the challenge of nonlinear latent identification.

**Next stage:** examine actual preprocessed ROI-level fMRI data from a well-documented public source (with sampling interval, task labels and confounds) **only if the project schedule permits**; otherwise, focus on methodological limitations and writing the report. Do not claim real fMRI findings before acquiring and analyzing real fMRI data.

## GitHub

Use `git status` before staging; `.venv`, `__pycache__` and notebook checkpoints should remain ignored. If your `.gitignore` excludes `*.npz`, `results/` or `data/`, those generated outputs may not be staged by `git add .`; that is expected and you can keep the code and notebook reproducible while choosing separately whether to track the small generated results.

```powershell
git add .
git status
git commit -m "Complete phase 6: synthetic BOLD and HRF-aware PLRNN"
git push
```
