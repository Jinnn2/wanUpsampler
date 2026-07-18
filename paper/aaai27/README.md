# AAAI-27 paper workspace

## Current source of truth

- `main_zh.md`: current Chinese manuscript. Edit this file first.
- `main.tex`: English AAAI layout draft. Synchronize it after the Chinese argument and experiment tables stabilize.
- `references.bib`: verified bibliography used by the LaTeX draft.
- `reproducibility_checklist.tex`: unmodified official checklist; complete it when the final experimental evidence is available.
- `aaai2027.sty`, `aaai2027.bst`: official files from the AAAI-27 Author Kit. Do not modify them.

The AAAI style does not allow non-Roman body fonts. `main_zh.md` is therefore a writing source, not a submit-ready AAAI file. The anonymous submission PDF must ultimately be English and compiled from `main.tex` with PDFLaTeX.

## AAAI-27 constraints

Checked against the official AAAI-27 pages on 2026-07-15:

- Main technical content: at most 7 pages.
- Maximum paper length: 9 pages; pages after page 7 are references only.
- Reproducibility checklist: required and uploaded separately.
- Review: double blind; omit authors, affiliations, acknowledgments, and identity-revealing links.
- Format: US Letter, two columns, high-resolution PDF, Type 1 or TrueType fonts.
- Abstract deadline: 2026-07-21, 23:59 UTC-12.
- Full paper deadline: 2026-07-28, 23:59 UTC-12.
- Supplement and code deadline: 2026-07-31, 23:59 UTC-12.
- Supplement is optional and reviewers are not required to read it; all evidence critical to acceptance must remain in the seven-page body.

Official sources:

- Author Kit: <https://aaai.org/authorkit27/>
- Main track call: <https://aaai.org/conference/aaai/aaai-27/main-technical-track-call/>
- Submission instructions: <https://aaai.org/conference/aaai/aaai-27/submission-instructions/>
- Supplement policy: <https://aaai.org/conference/aaai/aaai-27/supplementary-material/>

## Recommended areas

- Primary: `CV: Diffusion & Generative Models for Vision`
- Secondary: `CV: Computational Photography, Image & Video Synthesis`
- Optional secondary: `ML: Deep Generative Models & Autoencoders`

The primary contribution uses video-latent and resolution-specific structure, so the CV diffusion area is a better fit than a general machine-learning area.

## Page budget

| Part | Target body space |
|---|---:|
| Abstract and introduction | 1.25 pages |
| Related work | 0.65 page |
| Method and overview figure | 2.35 pages |
| Experiments | 1.25 pages |
| Results, ablations, and qualitative figure | 1.20 pages |
| Limitations and conclusion | 0.30 page |

The current Chinese draft deliberately keeps four result tables visible. Before submission, combine or move secondary tables only after the central 2x2 factorial table and quality-efficiency comparison are complete.

## Build

The required workflow is:

```powershell
pdflatex -output-directory=build main.tex
bibtex build/main
pdflatex -output-directory=build main.tex
pdflatex -output-directory=build main.tex
```

The current machine did not have PDFLaTeX installed at workspace setup time. Do not switch to XeLaTeX for the submission: the official author kit requires PDFLaTeX.

## Result status (2026-07-18)

The frozen base core and closure archive have been merged under
`results/integrated_20260718/`. The Chinese result analysis is in
`results/RESULTS_ANALYSIS_ZH.md`; `main_zh.md` v0.8 contains the paper-facing
tables and conclusions. The merged evidence now covers both Clean Latent Lifter (CLL) operators,
final step40/step45 endpoint tests, the Wan50 and distill4 factorials, prompt-level
human review, Native-HR quality-efficiency, and the Endpoint Re-entry Baseline.

The following experiments are deliberately out of scope and will not be run;
they must be stated as limitations rather than future required evidence:

- TAA rank/module/loss and CLL architecture/loss controlled ablations.
- Unseen-prompt/domain/checkpoint generalization.
- A matched quantitative CLL-versus-JTSL comparison; JTSL remains only
  a qualitative motivation.

Reported training resources are 4×NVIDIA H100, approximately 33 wall-clock
hours for TAA (implemented with LoRA) and 8 wall-clock hours for CLL. Non-experimental paper work
still includes the final failure-case figure and LTX-2 attribution/license
review.
