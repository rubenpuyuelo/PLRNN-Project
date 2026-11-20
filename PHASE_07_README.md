# Phase 7 — December 8 Progress Report & Presentation

This folder contains **ready-to-edit interim academic deliverables** based exclusively on the results produced in Phases 1–6 of the PLRNN-SSM repository.

## What is included

- `reports/progress_report/Progress_Report_Team15.pdf` — verified **two-page** English interim report.
- `reports/progress_report/Progress_Report_Team15.tex` — editable LaTeX source; requires the two PNGs in the adjacent `figures/` folder.
- `presentation/progress/Progress_Presentation_Team15.pptx` — **nine English-language slides**, with embedded speaker notes.
- `presentation/progress/Progress_Presentation_Team15.pdf` — portable slide preview.
- `presentation/progress/Presentation_Speaker_Notes.md` — approximately 7–8-minute oral script, slide by slide.

## How to integrate with GitHub

Copy the **contents** of `Team15_Progress_Package.zip` into the **root** of your existing repository. This adds files under `reports/` and `presentation/` without changing the model code or earlier phases.

```powershell
git status
git add reports/progress_report presentation/progress PHASE_07_README.md
git commit -m "Add December progress report and oral presentation"
git push
```

If your existing `.gitignore` excludes PDFs or generated files, check `git status` after staging.

## Editing the report

The LaTeX source uses common `pdflatex` packages. In the report directory:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error Progress_Report_Team15.tex
pdflatex -interaction=nonstopmode -halt-on-error Progress_Report_Team15.tex
```

Keep the two report figures in `reports/progress_report/figures/`. The pre-rendered PDF has been checked to contain exactly two A4 pages.

## Presentation suggestions

This is a **progress** presentation, approximately 7–8 minutes, not the final 15-minute defense. The slide notes are in the PPTX and as a separate Markdown file. Rehearse to make the distinction between forecast MSE, autonomous JSD, oracle BOLD observation regression and unknown-latent BOLD inference clear.

## Source/results traceability

- Proposal: `Team 15 - Project Proposal (1).pdf`.
- Main reference: Koppe et al. (2019), *PLOS Computational Biology*, doi:10.1371/journal.pcbi.1007263.
- Forecast MSE: `results/metrics/plrnn_{van_der_pol,lorenz}.json` and corresponding LDS outputs (Phases 2–3).
- Autonomous JSD: `results/phase4/metrics/phase4_summary.json` (Phase 4).
- Refinement across seeds: `results/phase5/metrics/phase5_summary.csv` (Phase 5).
- BOLD observation and latent MAP: `results/phase6/metrics/phase6_summary.json` (Phase 6).

## Scientific safeguards

The report and slides distinguish:
1. simplified joint-MAP PLRNN **vs.** original Laplace-EM PLRNN-SSM;
2. initial unmatched-dimensional MSE comparison **vs.** dimension-matched autonomous JSD comparison;
3. three fitted seeds **vs.** nine correlated free-run initializations;
4. an oracle decoder given true proxy states **vs.** unknown-state MAP inference;
5. synthetic BOLD-like data **vs.** real fMRI measurements.

Do not present current synthetic BOLD results as human fMRI findings, or state that a Lorenz chaotic attractor has been faithfully reconstructed.
