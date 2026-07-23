# AAAI 2027 rewrite workspace

This directory is a clean anonymous-submission template for rewriting the
paper from scratch. It intentionally contains no prose, figures, tables, or
bibliography entries from the current manuscript.

- `main.tex`: minimal AAAI 2027.1 anonymous-submission document.
- `aaai2027.sty`: unchanged style file from the repository's official AAAI
  2027 Author Kit.
- `references.bib`: empty bibliography database.
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
