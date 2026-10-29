# Results (local copy of the server's key outputs)

The server (`/mnt/afs_2/houze/wanUpsampler/outputs/...`) holds everything; this folder keeps the summary tables
(csv/json/md/txt) from each downloaded tgz so analysis and the paper can work without re-downloading. Videos, PNGs,
embeddings (npz), per-item caches (cache/, vlm/, pair/, records/, raw/ ...) and path manifests (plan.json,
dataset_manifest.json, evaluation_inputs.json) stay on the server and in the tgz.

**Adding a new result:** download the tgz as before, then

    python results/ingest.py E:/Downloads/<name>.tgz --name <group>/<name> --note "<one line>"

and add a row below. Each folder has `SOURCE.json` (archive name, sha256, date, kept files).

## Prompts and splits

Prompt files live in `prompts/` (tracked). dev = train + validation (ids 0-63), test = ids 64-79 (locked, never
used), dev2 = ids 100+ (`univ_controlled_factor_dev2.jsonl`), natural = ids 1000-1255 (`univ_natural_v1.jsonl`).
Cells: motion x detail, written `hl` = high motion, low detail.

## Index

| Folder | What | Source tgz |
| --- | --- | --- |
| `generation/platform_v1` | VBench per video + paired deltas, platform arms (FULL50, STEP20/25, TEA008/015, SCALING10), dev, 3 seeds | published_wan21_analysis.tgz |
| `generation/twostage_v1` | VBench per video, two-stage v1 arms (LR_FLASH, LR_BICUBIC, F41_RIFE, F41_DUP), dev | twostage_wan21_analysis.tgz |
| `generation/platform_dev2` | same as platform_v1 on dev2 | batch_dev2_results.tgz |
| `generation/twostage_dev2` | two-stage + rCM + LR41_FLASH_RIFE on dev2 (`metrics/report.json` = per-arm, per-cell summary) | batch_dev2_results.tgz |
| `generation/twostage_v2b` | v2b arms (RCM*, LR41_FLASH_RIFE) on dev | batch_dev2_results.tgz |
| `layers/fixed_v1` | fixed layer (flow, flicker, sharpness ... and paired deltas), all videos | fixed_metrics_v1_analysis.tgz |
| `layers/dynamic_v1_all` | semantic layer per video (checklist retention); `dynamic_by_question.csv` (12 MB) not kept | dynamic_v1_all_analysis.tgz |
| `layers/dynamic_v1_pilot` | semantic layer pilot | dynamic_v1_pilot_analysis.tgz |
| `layers/quality_v1` | quality layer **v1** per video. route_v1_q2 used the **v2** file (sha 894d33cb...), which is not here yet | quality_v1_analysis.tgz |
| `layers/degradation_v1` | degradation sanity check (synthetic damage) through all layers | degradation_v1_analysis.tgz |
| `checklists/requirements_v2_full` | final machine checklist (127 prompts, 1246 questions) used by the dynamic layer | requirements_v2_gen_full2.tgz |
| `checklists/requirements_v2_pilot` | checklist pilot + compare.md | requirements_v2_gen_pilot.tgz |
| `requirements_v1/*` | earlier requirement-retention analyses (controlled factor v1, platform v1/v2a/v2b/dev2, twostage dev2, DVG v1/dev2) | *_requirements_v1_analysis.tgz, batch_dev2_results.tgz |
| `dvg/study_v1`, `dvg/study_dev2`, `dvg/vbench_ext_v1` | DVG reproduction: quality, repro, VBench scores | dvg_wan21_study_v1 / vbench_ext_v1 tgz, batch_dev2_results.tgz |
| `route/route_v1_q2` | **main router run**: route_table (prompt x arm tf/retention/quality_kept), CV + dev->dev2 eval, rules, report | route_v1_q2_analysis.tgz |
| `route/route_v1_gate` | same router with `quality_kept_gate` | route_v1_gate_analysis.tgz |
| `route/route_v1_q2_embed` | vector router (umT5 / Qwen kNN, ridge) vs cell router at matched quality | route_v1_q2_embed_analysis.tgz |
| `derived/fig1_v1` | Figure 1 numbers: dev2 scoreboard (fixed arms vs router/blind), per-cell fixed-arm profile; computed locally by `fig1_export.py` | computed locally |
| `figures/figure_samples_v1` | sample selection for qualitative figures (candidates, pins, manifest); images only in the tgz | figure_samples_v1.tgz |
| `figures/fig1_samples_v1` | Figure 1a v1: dev2 seed-42 lower-median clip per (RCM4 / LR_FLASH, cell); candidate pools, scores, per-pair flow; PNG frames only in the tgz | fig1_samples_v1.tgz |
| `natural/natural_v1_prompts` | natural prompt set v1 build: candidates, Qwen screen, selection log, configs | natural_v1_prompts.tgz |
| `natural/natural_ext_v1_prompts` | natural ext v1 build: Wan-extender raw replies, summary (prompts in prompts/univ_natural_ext_v1.jsonl) | natural_ext_v1_prompts.tgz |
| `natural/natural_v1_results` | natural set v1: generation metrics, fixed / dynamic / quality v2 layers, checklist, vector-router eval (236 scored prompts) | natural_v1_results.tgz |
| `derived/natural_v1` | natural set v1 verdict against the pre-registered rule, and arm x cell interaction replication vs dev2; computed locally | computed locally |
| `misc/official_check` | official Wan sample check log | official_check.tgz |
| `misc/batch_dev2.log`, `misc/batch_dev2_SOURCE.json` | dev2 batch log; source record of the batch tgz split into the folders above | batch_dev2_results.tgz |

## Not here yet

- Quality layer v2 per-video csv (input of route_v1_q2).
- Human annotation results (annotation sessions own those files).
