# InTraScale AAAI 2027 submission source

This directory is the self-contained anonymous-submission source package.
It contains only the manuscript, official AAAI support files, bibliography,
reproducibility checklist, and the four PDF figures referenced by `main.tex`.

## Submission contents

- `main.tex`: AAAI 2027.1 anonymous manuscript.
- `aaai2027.sty`, `aaai2027.bst`: official AAAI 2027 Author Kit files.
- `references.bib`: manuscript bibliography.
- `ReproducibilityChecklist.tex`: official checklist source.
- `figures/fig_teaser.pdf`: real-video temporal teaser.
- `figures/fig_overall_framework.pdf`: sampling budget and framework.
- `figures/fig_challenge_interpolation.pdf`: interpolation challenge.
- `figures/fig_challenge_alignment.pdf`: trajectory-alignment challenge.

Build from this directory with:

```powershell
latexmk -pdf main.tex
```

## Canonical terminology

- Paper title: **InTraScale: Accelerating Video Generation via In-Trajectory
  Resolution Scaling**
- Framework: **InTraScale**
- Upsampling module: **In-Trajectory Upsampler (ITU)**
- LoRA module: **Trajectory-Tail Distillation Adapter (TTDA)**,
  Chinese: **轨迹尾段蒸馏适配器**

These names replace the previous TrajScale, CRLU, and EAA terminology in all
newly rewritten text.
