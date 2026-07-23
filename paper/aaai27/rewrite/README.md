# AAAI 2027 rewrite workspace

This directory is the self-contained AAAI 2027 anonymous-submission workspace
for the rewritten paper.

- `main.tex`: AAAI 2027.1 anonymous-submission manuscript.
- `aaai2027.sty`: unchanged style file from the repository's official AAAI
  2027 Author Kit.
- `aaai2027.bst`: unchanged bibliography style from the same official kit.
- `references.bib`: bibliography database used by the manuscript.
- `figures/`: figure assets referenced by `main.tex`.
- `ReproducibilityChecklist.tex`: unchanged checklist from the official kit.

Build from this directory with:

```powershell
latexmk -pdf main.tex
```

Generated LaTeX build artifacts are ignored by the local `.gitignore`.

## Canonical terminology

- Paper title: **InTraScale: Accelerating Video Generation via In-Trajectory
  Resolution Scaling**
- Framework: **InTraScale**
- Upsampling module: **In-Trajectory Upsampler (ITU)**
- LoRA module: **Trajectory-Tail Distillation Adapter (TTDA)**,
  Chinese: **轨迹尾段蒸馏适配器**

These names replace the previous TrajScale, CRLU, and EAA terminology in all
newly rewritten text.
