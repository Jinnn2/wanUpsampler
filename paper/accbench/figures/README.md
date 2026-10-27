# Figures for the AccBench / prompt-guided routing paper

Each figure has a script in `src/` that reads only files in `data/` (exported from the main line's
`outputs/`, with the export date in the file name) and writes the PDF/PNG next to this file. Numbers in the
figures must be traceable to `data/`; nothing is typed in by hand.

| figure | script | data | status |
|---|---|---|---|
| fig1_overview | `src/fig1_overview.py` | `data/route_table_20261027.json` (profiles inset) | skeleton; frame strips and scorecard numbers are placeholders |

## Pending from the main line (requested 2026-10-27)

1. Router frontier: `route_eval.csv`, `route_rules.json`, `evaluate.log` from `outputs/route_v1*` (router vs
   prompt-blind mixture per tau; dev->dev2 and CV). Feeds the fig1 scorecard and the frontier figure.
2. Frame strips for fig1(a): export script is on main (commit 7b5b969), waiting for the server run. RCM4 and
   LR_FLASH x four cells, dev2 seed 42; per (arm, cell) the lower median by min(retention, quality_kept) (ties by
   prompt id). Each group: 5 evenly spaced frames (first and last included) for FULL and the arm, `flow.csv`
   (per adjacent-frame pair, both clips), `samples.json` (prompt, scores, rank); README lists all candidates.
   Caption must say: the flow bar is OpenCV Farneback mean pixel magnitude at 416 px width, drawn only as a per-frame
   trend; it is not the RAFT flow of the fixed layer and the two are not comparable.
3. Done: `scoreboard_dev2_20261027.csv` (fixed arms, router and blind at every tau), `fixed_arms_by_cell_dev2_20261027.csv`,
   `scoreboard_dev2_policies_20261027.json`, `route_eval_20261027.csv`, `route_rules_20261027.json`.

## Data provenance

- `route_table_20261027.json`: the `D` object embedded in `doc/figures/method_by_cell_profiles.html`
  (main line, 2026-10-27), itself from `outputs/route_v1_q2/route_table.csv`. `r[arm][cell]` =
  `[retention, quality_kept, relative advantage, speedup]`; dev + dev2, 3 seeds averaged per prompt.
