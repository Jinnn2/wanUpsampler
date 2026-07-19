# AAAI-27 paper workspace

## Current source of truth

- `main.tex` + `body_polished.tex`: current AAAI-27 submission source. `main.tex` directly inputs `body_polished.tex`; this is the source used for every current PDF export.
- `main_polished.md`: earlier full-length English writing source. It is retained for prose comparison but is no longer synchronized with the condensed LaTeX submission source.
- `body_polished_restored.tex`: archived restored full-length LaTeX body. It is not included by `main.tex`.
- `body_condensed_7page.tex`: archived seven-page condensation experiment. It is retained only for comparison and is not included by `main.tex`.
- `supplementary.tex`: anonymous supplementary document containing the extended qualitative panel and detailed operator, alignment, and human-review tables.
- `main_zh.md`: Chinese writing source retained for terminology and formula cross-checking.
- `FORMULA_AUDIT_ZH.md`: Chinese record of the formula and notation audit.
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

## Page status

As of 2026-07-20, the submission PDF has 8 pages in total: the technical
content ends on page 7 and page 8 contains references only. The former main-paper
Table 3 is retained as the more complete Supplementary Table S2, Figure 4 has
been removed, and the remaining result floats are distributed across pages 6--7.
The verified export is `main_7page_latest.pdf`. Copy it to `main.pdf` after
closing any PDF viewer that holds the latter file open.

## Page budget (future submission pass)

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

The reproducible build is:

```powershell
.\build_paper.ps1
```

Its underlying PDFLaTeX workflow is:

```powershell
pdflatex -output-directory=build main.tex
bibtex build/main
pdflatex -output-directory=build main.tex
pdflatex -output-directory=build main.tex
```

The current machine now has a user-local TinyTeX/TeX Live 2026 installation. The exported files are `main.pdf` and `supplementary.pdf`; intermediate logs remain under `build/`. Do not switch to XeLaTeX for the submission: the official author kit requires PDFLaTeX.

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
