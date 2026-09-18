# Phase 2 fixed-lambda prompt utility pilot

This development experiment reuses the existing Phase 2 videos and VBench
scores. It tests whether prompt features can select among the existing action
presets under one predeclared utility:

```text
K(prompt, action) = VBench5(prompt, action)
                    - lambda * normalized_latency(action)
```

Normalized latency is the train-only median of the paired action/native
pipeline-time ratio. Validation timing never changes the profile. The default
launcher value is `lambda=0.05`; set it explicitly when recording a result.

This pilot is not a three-bit factorial experiment. `ACTION_SET=three` means
the existing CACHE, SKIP, and B30_SPATIAL presets. Those presets change several
execution parameters together. A positive result justifies generating the
complete `2^3` action grid later; it does not identify independent operation
effects.

## Run

```bash
cd /mnt/afs_2/houze/wanUpsampler
export QUALITY_DIR=/mnt/afs_2/houze/wanUpsampler/outputs/univ_prompt_budget_phase2_20260918_chunk25/metrics/phase2_quality

UTILITY_LAMBDA=0.05 \
bash UNIV_adaptor/scripts/run_univ_phase2_utility_prior.sh all

# Reuse the already verified Wan T5 embeddings, or extract them once on GPU 0.
FEATURES=t5 GPU_ID=0 UTILITY_LAMBDA=0.05 \
bash UNIV_adaptor/scripts/run_univ_phase2_utility_prior.sh all
```

Modes are `check`, `embed`, `train`, and `all`. `check` validates the saved
score artifacts and builds the train-only latency profile. No video generation
or VBench scoring is performed.

The model regresses per-action utility gains relative to CACHE. Five-fold
train-only CV selects ridge strength by out-of-fold policy regret, with utility
gain MSE as a tie-break. Validation is read once after fitting and remains
explicitly `validation_development`, because it was already used during Phase
2 action development. Test is never read.

Outputs under `utility_prior_FEATURES_ACTIONSET_lambda_VALUE/` include:

- `latency_profile.json`: paired train-only normalized action costs;
- `train_lambda_diagnostics.csv`: train-only oracle geometry for lambda 0--.10;
- `oracle_labels.csv`: per-prompt utility vector, best action, runner-up, margin;
- `cross_validation.csv`: train-only alpha selection by policy regret;
- `utility_predictions.csv`: OOF/development predictions and regret;
- `policy_summary.csv/json`, `policy_by_prompt.csv`, and `report.md`;
- `model.json/npz` and a content-bound `training_request.json`.

The main comparison is `prompt_utility` against both `fixed_train` and
`shuffled_router_hist_expected`. The shuffled control preserves the prompt
router's action proportions and normalized cost, isolating whether action to
prompt matching helps. Continue to a new `2^3` dataset only if the prompt model
improves utility with acceptable material-harm rate and non-degenerate action
use.
