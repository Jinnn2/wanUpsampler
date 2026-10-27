# Figure 1 data (line A export, 2026-10-27)

Source: route_v1_q2 (quality layer v2, column `quality_kept`), the current main router run. 125 non-test prompts,
of which dev (train+validation) = 64 and dev2 = 61 scored (3 dev2 prompts dropped: missing scores). Test split
(ids 64-79) not used.

- `route_v1_q2/route_eval.csv`, `route_rules.json`, `route_report.md`, `route_table.csv`, `table_inputs.json`:
  unchanged server outputs. No separate evaluate.log was exported; `route_report.md` holds the same numbers.
  `route_rules.json` = rules fit on ALL non-test prompts (deployment). Classifiers: oracle (true cell), bow.
- `scoreboard_dev2.csv`: same-protocol numbers on dev2 for fixed arms and for router / blind at every tau, policies
  fit on dev only (dev->dev2). Router speeds equal route_eval.csv dev->dev2 rows (check).
  - speedup = 1 / mean over prompts of (arm time / FULL time).
  - retention, quality_kept = means over prompts; min_ret_q = min(mean retention, mean quality_kept) (the router's
    floor is on these two means).
  - d_vbench5_pp = mean over prompts of (VBench-5 of arm - FULL50), seeds averaged per prompt, x100.
  - router/blind values are expected values under the per-cell arm distribution (no sampling).
  - _lo/_hi = 95% family bootstrap (2000).
- `scoreboard_dev2_policies.json`: the dev-fit per-cell distributions (router) and the blind mixture per tau.
- `fixed_arms_by_cell_dev2.csv`: RCM4 / LR_FLASH / LR_BICUBIC per cell on dev2.
- `fig1_export.py`: the script (reuses route_v1.Data / solve).
