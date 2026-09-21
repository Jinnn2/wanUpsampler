# Phase 4 reference-centered matched-star supplement

This batch completes clean single-axis `R/S/T/C` comparisons without
enumerating all action combinations. It reuses the finalized Phase 3 dataset
and adds 24 prompt-only selected training prompts.

The locked actions are:

| id | spatial | temporal | LR NFE | switch |
|---|---:|---:|---:|---:|
| `REFERENCE` | 0.75 | 0.80 | 0.55 | 0.80 |
| `STAR_S` | 0.625 | 0.80 | 0.55 | 0.80 |
| `STAR_T` | 0.75 | 0.67 | 0.55 | 0.80 |
| `STAR_C` | 0.75 | 0.80 | 0.40 | 0.80 |

Phase 3 `SA_100` and `SA_010` are deliberately not reused: both use
`lr_nfe_ratio=0.7`, so neither is a clean spatial-only or temporal-only change
from the true reference. Exact Phase 3 references and `SA_001`-equivalent
`STAR_C` artifacts are reused by action hash.

With the finalized Phase 3 v1 root and prompt offset 181, the immutable plan
must report:

- 93 training prompts: 69 Phase 2-derived plus 24 new prompts;
- 20 existing fresh development prompts;
- 339 prompt-seed groups;
- 990 generated videos;
- 366 reused videos;
- 1,356 videos to score.

## Preflight and plan

```bash
cd /mnt/afs_2/houze/wanUpsampler

SOURCE_PHASE3_ROOT=/mnt/afs_2/houze/wanUpsampler/outputs/univ_sparse_action_phase3_v1 \
bash UNIV_adaptor/scripts/run_univ_matched_star_phase4_8gpu.sh check

SOURCE_PHASE3_ROOT=/mnt/afs_2/houze/wanUpsampler/outputs/univ_sparse_action_phase3_v1 \
bash UNIV_adaptor/scripts/run_univ_matched_star_phase4_8gpu.sh plan
```

Do not start generation unless `plan` prints exactly
`generated_videos=990`, `reused_videos=366`, `scored_videos=1356`, and
`prompt_seed_groups=339`. Preparation verifies the SHA-256 identity of every
reused Phase 3 video and sidecar and never reads Phase 3 scores.

The defaults select 24 non-comment prompts beginning at filtered prompt offset
181 from `prompts/univ_controller_pilot_500.txt`. Override
`NEW_PROMPTS_FILE` and `NEW_PROMPT_OFFSET` only with a disjoint prompt-only
selection. A changed prompt source requires a new `OUT_ROOT`.

## Generate and finalize

```bash
SOURCE_PHASE3_ROOT=/mnt/afs_2/houze/wanUpsampler/outputs/univ_sparse_action_phase3_v1 \
bash UNIV_adaptor/scripts/run_univ_matched_star_phase4_8gpu.sh generate

SOURCE_PHASE3_ROOT=/mnt/afs_2/houze/wanUpsampler/outputs/univ_sparse_action_phase3_v1 \
bash UNIV_adaptor/scripts/run_univ_matched_star_phase4_8gpu.sh finalize
```

The default output is:

```text
/mnt/afs_2/houze/wanUpsampler/outputs/univ_matched_star_phase4_v1
```

`RESUME=1` verifies complete timing, MP4 and runtime-sidecar coverage before
skipping a job. Eight workers receive weighted, disjoint chunks. A root lock
prevents concurrent writers.

For a short smoke run, set `MAX_JOBS_PER_WORKER=1`, but do not finalize that
partial run. Rerun without the limit to complete the same immutable plan.

## Score

The output remains compatible with the existing sparse-action scorer:

```bash
DATASET_ROOT=/mnt/afs_2/houze/wanUpsampler/outputs/univ_matched_star_phase4_v1 \
bash UNIV_adaptor/scripts/run_univ_sparse_action_score_8gpu.sh check

DATASET_ROOT=/mnt/afs_2/houze/wanUpsampler/outputs/univ_matched_star_phase4_v1 \
EXPECTED_VBENCH_COMMIT="$(git -C /mnt/afs_2/houze/VBench rev-parse HEAD)" \
bash UNIV_adaptor/scripts/run_univ_sparse_action_score_8gpu.sh all
```

The command substitution pins scoring to the exact VBench checkout. Do not
enter the documentation placeholder `<locked-commit>` literally: Bash treats
angle brackets as redirection operators. Formal scoring also requires the
tracked VBench checkout to be clean.

Quality and measured pipeline time remain separate, so lambda is applied only
during router training.

## Early-state boundary

This script generates the matched action labels only. It does not silently
claim the completed REFERENCE video proxy is an online observation. A separate
prefix-only collection should save step-12 `x_t` and predicted-clean state and
account for prefix/restart cost. Keeping that collection separate prevents an
untested runner modification from contaminating this 990-video batch.
