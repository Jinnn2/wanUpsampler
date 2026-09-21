# Sparse prompt-action Phase 3

This development experiment continues from the Phase 2 videos without learning a
hard per-prompt winner.  It learns from incomplete prompt-action blocks:

- one common reference: the Phase 2 `P2_B30_SPATIAL` action;
- three probe actions per prompt, fixed across all three seeds;
- explicit binary action levels for spatial compression, temporal compression,
  and LR compute;
- VBench-5 quality and measured time stored separately, with no fixed lambda and
  no oracle class label.

The checked-in protocol uses 69 Phase 2 train prompts and 20 fresh development
prompts.  It generates at most 999 new videos, reuses at least 69 Phase 2
videos, and scores 1,068 observations in 267 prompt-seed groups.  In addition to
the reference, the planner automatically reuses any probe whose complete action
and transition exactly match a Phase 2 artifact; currently `SA_001` matches
`P2_B25_SKIP`.  That Phase 2 name remains an identifier for retained-grid cache
reuse, not evidence of an independently implemented timestep-skip operator.  It does not
open Phase 2 score artifacts, and prompt selection never uses validation
outcomes.  The eight-row probe library is shared across prompts; each prompt
evaluates only three rows.  The data volume therefore does not enumerate the
library for every prompt, and future protocols may keep a fixed explicit library
when adding dimensions.

## Prompt source

By default the launcher reads prompts 161 through 180 from
`prompts/univ_controller_pilot_500.txt`.  These follow the 161 prompts reserved
by Phase 2.  To use a separately curated 20-prompt file, set
`FRESH_PROMPTS_FILE` and `FRESH_PROMPT_OFFSET=0`.  Preparation rejects overlap
with all Phase 2 train and validation prompts.

## Generate on eight GPUs

Run the non-generating gates first:

```bash
cd /path/to/wanUpsampler

SOURCE_PHASE2_ROOT=/path/to/complete/univ_prompt_budget_phase2_v1 \
bash UNIV_adaptor/scripts/run_univ_sparse_action_phase3_8gpu.sh check

SOURCE_PHASE2_ROOT=/path/to/complete/univ_prompt_budget_phase2_v1 \
bash UNIV_adaptor/scripts/run_univ_sparse_action_phase3_8gpu.sh plan
```

`plan` verifies every reused video and sidecar by size and SHA-256, freezes the
prompt/action assignments, materializes per-action runtime configs and explicit
job inputs, and prints the eight-worker load.  Use a new `OUT_ROOT` if any input
or protocol changes.

Launch or resume generation, then finalize immutable records:

```bash
SOURCE_PHASE2_ROOT=/path/to/complete/univ_prompt_budget_phase2_v1 \
bash UNIV_adaptor/scripts/run_univ_sparse_action_phase3_8gpu.sh generate

SOURCE_PHASE2_ROOT=/path/to/complete/univ_prompt_budget_phase2_v1 \
bash UNIV_adaptor/scripts/run_univ_sparse_action_phase3_8gpu.sh finalize
```

`all` runs prepare, plan display, generation, and finalization.  Eight workers
write disjoint job timing files and video paths.  A root-level lock prevents two
launchers from writing the same output concurrently.  `RESUME=1` only skips a
job after verifying its full timing/video/sidecar coverage.

## Score and build paired targets

```bash
DATASET_ROOT=/path/to/univ_sparse_action_phase3_v1 \
bash UNIV_adaptor/scripts/run_univ_sparse_action_score_8gpu.sh check

DATASET_ROOT=/path/to/univ_sparse_action_phase3_v1 \
EXPECTED_VBENCH_COMMIT="$(git -C /mnt/afs_2/houze/VBench rev-parse HEAD)" \
bash UNIV_adaptor/scripts/run_univ_sparse_action_score_8gpu.sh all
```

This pins the run to the exact clean VBench checkout. Do not use a literal
`<locked-commit>` value because Bash interprets angle brackets as redirection.

The scorer verifies the frozen dataset and all video hashes before staging
hardlinks/copies.  It runs the five quality dimensions plus Dynamic Degree and
Overall Consistency, one resumable distributed VBench request per dimension.
It preserves absolute measured seconds and the ratio to the common accelerated
reference.  Native-HR normalization, if needed by `Q - lambda*T`, must later be
fit from training-only Phase 2 native/action pairs rather than validation data.
Important outputs under `metrics/sparse_action_vbench` are:

- `evaluation_inputs.json`: content-bound scoring request;
- `scores.json`: VBench values and per-dimension provenance;
- `quality_by_video.csv`: quality and time for every observation;
- `relative_quality_pairs.csv`: probe-minus-reference quality and time targets;
- `seed_stability.csv`: within-prompt/action variation across the three seeds;
- `action_summary.csv` and `report.md`: descriptive diagnostics only;
- `scored_dataset.json`: immutable hashes and the explicit declarations
  `lambda_bound=false` and `hard_oracle_labels_created=false`.

This stage does not train the action-conditioned predictor.  The next stage
should split by prompt and fit `(prompt, named action vector, active mask) ->
delta quality`; utility is computed afterward as `Q - lambda * T`.
