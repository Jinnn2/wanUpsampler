# InTraScale AAAI 2027 submission source

This directory is the self-contained anonymous-submission workspace. It
contains the manuscript, official AAAI support files, bibliography,
reproducibility checklist, the five PDF figures referenced by `main.tex`, and
editable sources for the terminology-sensitive figures.

## Submission contents

- `main.tex`: AAAI 2027.1 anonymous manuscript.
- `aaai2027.sty`, `aaai2027.bst`: official AAAI 2027 Author Kit files.
- `references.bib`: manuscript bibliography.
- `ReproducibilityChecklist.tex`: official checklist source.
- `figures/fig_teaser.pdf`: real-video temporal teaser.
- `figures/fig_overall_framework.pdf`: sampling budget and framework.
- `figures/fig_challenge_interpolation.pdf`: interpolation challenge.
- `figures/fig_challenge_alignment.pdf`: trajectory-alignment challenge.
- `figures/fig_quality_efficiency.pdf`: Wan50 quality--efficiency operating
  points.
- `figure_sources/`: editable templates, recovered framework assets, and the
  reproducible figure renderers.

Build from this directory with:

```powershell
latexmk -pdf main.tex
```

## Canonical terminology

- Paper title: **InTraScale: Accelerating Video Generation via In-Trajectory
  Resolution Scaling**
- Framework: **InTraScale**
- Upsampling module: **In-Trajectory Upsampler (ITU)**
- Distillation mechanism: **Trajectory-Tail Distillation (TTD)**,
  implemented through a step-specific LoRA update; Chinese: **轨迹尾段蒸馏**

These names replace the previous TrajScale, CRLU, and EAA terminology in all
newly rewritten text.
