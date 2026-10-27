# Figures for the AccBench / prompt-guided routing paper

Each figure has a script in `src/` that reads only files in `data/` (exported from the main line's
`outputs/`, with the export date in the file name) and writes the PDF/PNG next to this file. Numbers in the
figures must be traceable to `data/`; nothing is typed in by hand.

| figure | script | data | status |
|---|---|---|---|
| fig1_overview | `src/fig1_overview.py` | `route_table_*.json`, `scoreboard_dev2_*.csv`, `fixed_arms_by_cell_dev2_*.csv`, `scoreboard_dev2_policies_*.json`, `fig1_samples_v2/` | complete draft; open items below |

## fig1_overview: what is shown

- Both bands use the same four dev2 prompts (one per motion x detail cell, seed 42), chosen by the pre-registered
  v2 rule in `data/fig1_samples_v2/README.md`: within a cell, sort prompts by the mean over {RCM4, LR_FLASH} of
  min(retention, quality_kept) (ties by prompt id) and take the lower median. No hand picking; all candidates are
  listed there. Chosen: hh 119 (wooden tower collapses), hl 114 (leaf blown across floor), lh 145 (pinball
  playfield, static), ll 112 (leaf lying still).
- Band (a) rows: FULL frame 40 | rCM 4-step frames 0/40/80 | LR+FlashVSR frames 0/40/80. Box tint and the
  "kept / dVB" label are the arm's **dev2 cell means** (`fixed_arms_by_cell_dev2_*.csv`), not the shown clip's
  own scores (those are in `fig1_samples_v2/samples.json`).
- Band (b) rows: FULL frame 40 | routed arm frames 0/40/80, routed arm = dev-fit argmax at tau (default 0.80:
  hh->RCM4, hl->SCALING10 (98%), lh->F41_RIFE, ll->TEA015). Tint and "kept / dVB" are the routed arm's dev2 cell
  means from the same file as (a).
- Scorecards: identical axes in both bands (speedup vs min(semantic, quality) kept, dev2). (a) highlights the two
  fixed arms with their dVBench-5; (b) adds the router and prompt-blind frontiers over tau 0.95..0.60.

## Open items

1. Paper name for SCALING10 (`ARM_NAME` in the script).
2. The hh row of (b) routes to rCM, whose dev2 hh kept is 0.48 and whose shown clip scores min 0.17: the LP
   constrains the **mean** over prompts, so hh is traded for speed at tau = 0.80. Decide whether the figure should
   show tau = 0.85 (hh -> rCM 68% / FULL) or keep 0.80 and say this in the caption.
3. Router frontier below tau = 0.60 (needs the main line to extend TAUS).
4. Caption must say: the flow values in `fig1_samples_v2/*/flow.csv` are OpenCV Farneback mean pixel magnitude at
   416 px width, drawn only as a per-frame trend; not the RAFT flow of the fixed layer and not comparable to it.
   (Not drawn in the current figure.)

## Data provenance

- `route_table_20261027.json`: the `D` object embedded in `doc/figures/method_by_cell_profiles.html`
  (main line, 2026-10-27), itself from `outputs/route_v1_q2/route_table.csv`. `r[arm][cell]` =
  `[retention, quality_kept, relative advantage, speedup]`; dev + dev2, 3 seeds averaged per prompt.
- `scoreboard_dev2_20261027.csv`, `scoreboard_dev2_policies_20261027.json`, `route_eval_20261027.csv`,
  `route_rules_20261027.json`: copies of `outputs/fig1_v1/` and `outputs/fig1_v1/route_v1_q2/` (main line,
  2026-10-27). Router / blind policies are fit on dev, evaluated on dev2.
- `fixed_arms_by_cell_dev2_20261027.csv`: copy of `results/derived/fig1_v1/fixed_arms_by_cell_dev2.csv` (main line
  commit 2801355, all 15 arms x 4 cells, dev2, 3 seeds averaged). The DVG arms' dVBench-5 is relative to the FULL
  the DVG study ran itself (same prompt and seed); every other arm is relative to the published FULL50.
- `fig1_samples_v1/`: first export (`run_univ_figure_samples.sh fig1`, commit 7b5b969): per (arm, cell) the lower
  median prompt by min(retention, quality_kept), RCM4 and LR_FLASH only; different prompts per arm. Superseded by
  v2 for the figure, kept for the candidate tables.
- `fig1_samples_v2/`: `run_univ_figure_samples.sh fig1-v2` (commit 94ef1a1), server run 2026-10-27. Frames are
  416-px JPEG (q90) copies of the server PNGs; `README.md`, `samples.json`, `select.log` and the per-cell `flow.csv`
  are verbatim. FULL of every cell was checked against its prompt before use (all four match).
