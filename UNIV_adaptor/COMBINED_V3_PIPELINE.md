# UNIV combined-v3 scoring and prompt-prior pipeline

This pipeline consumes the finalized Primary and Reserve low-budget extension
roots. It scores and indexes train plus validation only. It never reads or
generates test records.

## Data contract

The expected selection dataset is:

```text
600 train prompts      x 1 seed  = 600 trajectories
200 validation prompts x 3 seeds = 600 trajectories
1,200 trajectories x (Native-HR + 9 actions) = 12,000 video references
```

The five quality dimensions are subject consistency, background consistency,
motion smoothness, aesthetic quality, and imaging quality. Their arithmetic
mean is `vbench5`. Dynamic Degree is stored separately as a diagnostic and is
not included in utility.

Generation records stay immutable. Scoring writes a separate tree:

```text
outputs/univ_combined_v3_scoring_v1/
  score_manifest.json
  staging/                    # hard links, not video copies
  metrics/vbench/             # content-bound VBench runs
  case_scores/                # one verified bundle per case
  scored_records/             # immutable scored record copies
  scored_dataset_manifest.json
```

`SCORE_ROOT` must be on the same filesystem as the source videos because the
strict VBench staging tree uses hard links. The default AFS output path meets
this requirement.

## Run stages

Use a fixed VBench checkout. Supplying the commit explicitly makes the intended
identity visible before a long run:

```bash
cd /mnt/afs_2/houze/wanUpsampler
export EXPECTED_VBENCH_COMMIT="$(git -C /mnt/afs_2/houze/VBench rev-parse HEAD)"

bash UNIV_adaptor/scripts/run_univ_combined_v3_pipeline.sh check
bash UNIV_adaptor/scripts/run_univ_combined_v3_pipeline.sh prepare
```

`prepare` validates 1,200 generated records and creates 40 scoring cases:

```text
2 shards x 2 splits x (Native-HR + 9 actions) = 40 cases
```

Run scoring inside tmux:

```bash
tmux new -s univ_combined_v3_score
cd /mnt/afs_2/houze/wanUpsampler
export EXPECTED_VBENCH_COMMIT="<commit printed by check>"
bash UNIV_adaptor/scripts/run_univ_combined_v3_pipeline.sh score
```

Monitor from another shell:

```bash
bash UNIV_adaptor/scripts/run_univ_combined_v3_pipeline.sh status
tail -f outputs/univ_combined_v3_scoring_v1/logs/*.log
```

Rerunning `score` is the supported resume path. Each case reuses a prior result
only when its video inventory, video hashes, prompt map, dimensions, scorer,
Python, and VBench identity all match.

After status reports `40/40`:

```bash
bash UNIV_adaptor/scripts/run_univ_combined_v3_pipeline.sh finalize
bash UNIV_adaptor/scripts/run_univ_combined_v3_pipeline.sh merge
bash UNIV_adaptor/scripts/run_univ_combined_v3_pipeline.sh embed
bash UNIV_adaptor/scripts/run_univ_combined_v3_pipeline.sh train
```

The merged dataset is written to:

```text
outputs/univ_combined_v3_trainval_v1/
  dataset_index.json
  prompts.txt
  t5_embeddings/
```

Primary and Reserve local prompt ids are not used as global identity. The merge
rejects prompt overlap across shards and assigns `global_prompt_id` from the
prompt SHA-256 after grouping by the frozen train/validation split.

## Training selection

The model predicts a nine-action relative VBench-5 curve from the frozen Wan
UMT5 pooled prompt embedding. Action selection is performed as:

```text
argmax_action predicted_quality(action | prompt)
              - lambda * train_normalized_latency(action)
```

The latency profile is the per-action median generation pipeline time from the
train split divided by the train Native-HR median. Validation timing never
calibrates action cost. The default selection grid is every `.01` from `.01` to
`.10`, and three training seeds are compared on validation macro policy regret.
Completed seed runs have their own hashed summaries, so rerunning `train` resumes
at the first unfinished seed without replacing a completed checkpoint.

The generation records contain synchronized wall time but not a GPU model name.
Set `HARDWARE_LABEL` when the generation hardware is known; it is stored as an
operator-declared label. This train-normalized profile is a selection coordinate,
not by itself a formal cross-hardware speed claim.

Outputs include:

```text
outputs/univ_combined_v3_budget_prior_v1/
  latency_profile.json
  seed_*/budget_prior.pt
  selected_budget_prior.pt
  validation_results.csv
  validation_predictions.csv
  validation_paired_bootstrap.csv
  selection_summary.json
```

The report separates a prompt oracle upper bound, a train-selected fixed action,
and the learned prompt prior. Positive bootstrap improvement means the learned
prior is better than the fixed baseline.

This completes the prompt-prior selection stage. It does not yet claim that the
stored action-specific endpoints provide a common online branch observation;
an endpoint-conditioned correction model remains a separate experiment after
the prompt prior is validated.

## True B4 control

The quality-curve prior above is not the original B4 objective. After it has
finished, run the isolated B4 control with:

```bash
DATASET_ROOT=/mnt/afs_2/houze/wanUpsampler/outputs/univ_combined_v3_trainval_v1 \
QUALITY_CURVE_ROOT=/mnt/afs_2/houze/wanUpsampler/outputs/univ_combined_v3_quality_curve_prior_v1 \
B4_OUT_ROOT=/mnt/afs_2/houze/wanUpsampler/outputs/univ_combined_v3_b4_control_v1 \
HARDWARE_LABEL=H100 \
bash UNIV_adaptor/scripts/run_univ_combined_v3_pipeline.sh train-b4
```

This trains two validation-only controls:

- `b4_fixed_lambda_bank`: one original-style B4 soft-utility classifier per
  lambda and training seed. This is the closest comparison with the old B4,
  but it is a bank of ten controllers rather than one variable-lambda policy.
- `b4_variable_lambda`: one B4 soft-utility classifier per seed with normalized
  lambda appended to the prompt embedding. This is the deployable shared-model
  extension for the combined-v3 lambda grid.

Both use the B4 `4096 -> 256 -> 128 -> 9` hidden backbone, soft targets
`softmax((Q - lambda*C) / tau)`, and KL plus `0.5 * Wasserstein` loss. The
Wasserstein term orders the nine non-ordinal actions by the locked train cost
profile. Validation decisions use probability ensembles across training seeds;
individual-seed metrics are retained as stability diagnostics. The report also
reloads the prior quality-curve checkpoints and compares all learned methods
with the same prompt oracle and train-selected fixed action.

The B4 output root includes per-action predicted probabilities and soft targets,
per-lambda and macro prompt-bootstrap comparisons, and every seed/lambda
training history. It does not read a test split.
