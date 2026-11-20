# Oral presentation — Team 15

**Progress meeting: 8 December 2026 · Suggested pace: approximately 7–8 minutes**

These notes are also embedded in the PowerPoint speaker notes. The progress report is a **two-page interim document**, not the full January report.

## 1 · Introduction (~30 seconds)

Good morning. We are Team 15, and our project investigates whether generative recurrent neural networks can identify nonlinear dynamical systems from noisy observations. Our main reference is Koppe and colleagues' 2019 paper. Today, we will present our implemented models, synthetic experiments, what actually improved, and the limitations we still have to address.

## 2 · Research question (~45 seconds)

Our key distinction is between predicting a few future samples and learning a dynamical system. A model can achieve a low mean squared prediction error yet fail when asked to produce a new trajectory by itself. For chaotic systems like Lorenz, two valid trajectories rapidly diverge, so direct point-by-point matching is inadequate for long-run assessment. We therefore examine both forecast accuracy and the distributions and shapes of autonomously generated trajectories.

## 3 · Models and implementation (~60 seconds)

We use two state-space approaches. The linear baseline, LDS-SSM, is trained with Kalman filtering and an expectation-maximization procedure. Our nonlinear PLRNN contains a ReLU interaction term with an off-diagonal connection matrix. We trained it through alternating maximum-a-posteriori estimates of latent states and parameters, increasing temporal consistency during fitting. This implementation is inspired by the paper but is not its full Laplace-EM inference. That distinction is important for interpreting the results.

## 4 · Initial prediction results (~50 seconds)

We simulated 1,000 observations from van der Pol and Lorenz, fitting on the first 700 and holding out the last 300. On the seed-42 initial run, the PLRNN has lower 10-step MSE: about 0.145 versus 0.313 for van der Pol, and 0.868 versus 2.949 for Lorenz. However, this early comparison is not parameter-matched: the PLRNN uses 8 or 12 latent states, while the first LDS implementation uses 2 or 3. We treat it as indicative rather than definitive evidence.

## 5 · Free-running dynamics (~60 seconds)

We then stopped using the observations to guide the models and generated independent trajectories. The geometric metric is histogram Jensen-Shannon divergence, or JSD, with lower values indicating a closer distribution. In a later, latent-dimension-matched comparison, PLRNN had a median JSD of about 0.431 on van der Pol and 0.422 on Lorenz, better than the LDS values of 0.500 and 0.589. Nevertheless, visual inspection clearly shows that the PLRNN does not recover the full Lorenz chaotic butterfly. We show all three fitted seeds, each with three related initial conditions, rather than selecting only a successful trajectory.

## 6 · Refinement (~50 seconds)

To improve autonomous generation, we introduced a multi-step loss and selected stable parameter updates using training-only diagnostics. This substantially improved van der Pol: the autonomous JSD falls from 0.431 to 0.059 for seed 42, 0.500 to 0.127 for seed 7, and 0.202 to 0.049 for seed 21. The Lorenz results are more limited: only one seed improves modestly, and two retain the original parameters. Thus, this refinement helps periodic behavior without solving the reconstruction of chaos.

## 7 · BOLD observation model (~45 seconds)

The paper is motivated partly by fMRI. fMRI BOLD measurements are filtered versions of underlying neural activity. We model this filtering by convolving a simulated latent signal with a fixed, causal double-gamma hemodynamic response function. Here, the inputs are ODE-based proxies for neural activity, not brain recordings, and our sample-to-time mapping is only illustrative. The experiment tests observation-model effects rather than clinical or neuroscientific claims.

## 8 · Two BOLD experiments (~65 seconds)

We separate two questions. In experiment A, true proxy latent states are supplied to linear decoders: with the correct known HRF, van der Pol held-out clean-BOLD MSE is around 0.000011, compared with 0.121 for an instantaneous model. This is expected because we generated the synthetic data using that HRF; it does not demonstrate hidden-state inference. In experiment B, the model must infer latent states from noisy BOLD alone. The HRF-aware PLRNN has lower free-running JSD on van der Pol, 0.123 versus 0.238, but on Lorenz its autonomous trajectory diverges. We explicitly report that failure.

## 9 · Conclusions and next steps (~50 seconds)

Our provisional answer is that nonlinear state-space models can improve both short-horizon forecasts and aspects of long-run geometry relative to a linear baseline. Yet successful nonlinear identification is harder, as the unresolved Lorenz reconstruction shows. Our work is exploratory: few seeds, simplified inference, synthetic signals, and no formal significance testing or real fMRI. Before January 19, we plan to repeat controlled experiments, improve stability or inference, consider publicly documented ROI-level fMRI if feasible, and complete the final written report and 15-minute defense.

## Likely questions from the lecturer

**Did you reproduce Koppe et al.'s full algorithm?** No. We implemented the PLRNN structural equations with simplified joint-MAP fitting; the paper uses a Laplace-EM scheme and a specific annealing protocol. We state the distinction explicitly.

**Does lower MSE imply the same dynamical system was identified?** No. Short-horizon prediction and autonomous attractor reconstruction are different tasks. Lorenz shows the gap.

**Why use Jensen–Shannon divergence?** It provides a symmetric, bounded comparison of estimated state-space occupancy distributions from generated versus reference trajectories. It is not the paper's normalized KL metric.

**Why does the HRF-aware oracle decoder do so well?** It is given the proxy latent path and the exact HRF used to generate the signals. It isolates observation-model mismatch and is not an inference benchmark.

**Do you have real fMRI results?** No. All BOLD-like data used here are synthetic. We will only claim real fMRI analysis once genuine records are acquired, processed and analyzed.

**Why only three training seeds?** The study is exploratory and compute-limited. Three seeds plus multiple initial states reveal failure modes, but do not support formal significance claims.
