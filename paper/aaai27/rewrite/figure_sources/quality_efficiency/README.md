# Quality--efficiency figure

`render_fig_quality_efficiency.py` renders the single-column Wan50 result
figure from:

`paper/aaai27/results/warm_quality_efficiency_20260722.csv`

The script validates the eight plotted cases against the rounded values in
Table 1 before producing:

- `rewrite/figures/fig_quality_efficiency.pdf`
- `rewrite/figures/fig_quality_efficiency.png`

Run from the repository root:

```powershell
python paper/aaai27/rewrite/figure_sources/quality_efficiency/render_fig_quality_efficiency.py
```

`render_fig_quality_efficiency_pareto_v2.py` is an alternative candidate that
computes a tolerance-aware Pareto frontier over the eight Wan50 operating
points reported in Table 1. Speedups within 1% are treated as practically
equivalent, so the higher-quality point dominates; in particular,
`InTraScale@45` dominates `Trilinear@45`. Endpoint-ITU 0/1/2 HR
sweep points are intentionally excluded; only the representative 5 HR
baseline remains. The script writes separate
`fig_quality_efficiency_pareto_v2.{pdf,png}` files and does not replace the
figure referenced by `main.tex`.

`render_fig_quality_efficiency_distill4_pareto.py` renders a separate
single-column Pareto candidate for the five Distill4 operating points reported
in the paper's main table. It reads warm latency and VBench-5 from the newest
validated P0/P1/P3 export and writes
`fig_quality_efficiency_distill4_pareto.{pdf,png}`. The candidate is not
referenced by `main.tex`.
