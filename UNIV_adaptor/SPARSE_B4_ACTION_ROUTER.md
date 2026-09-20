# Sparse B4-style action router

This experiment returns the method-selection line to the supervision that made
the earlier timestep B4 router effective, while reusing every Phase 3 video and
score. It does not enumerate all binary action combinations and does not launch
video generation.

## What changes

The old sparse prompt/state audit regressed each observed action row and chose
ridge strength by row MSE. This version instead:

1. computes `K = delta_vbench5 - lambda * (time_ratio - 1)`;
2. averages `K` across generation seeds for each prompt/action cell;
3. trains the prompt prior on a B4-style soft distribution over `REFERENCE`
   plus the three sampled actions;
4. trains the state branch only on the seed residual around the prompt/action
   mean; and
5. selects regularization by prompt-disjoint OOF policy regret, with soft
   cross-entropy and row MSE used only as tie-break diagnostics.

The action score uses main and pairwise action terms plus low-order
context/action interactions. Parameter growth is polynomial in the number of
axes. Phase 3 evaluation remains limited to the actions actually observed for
each prompt/seed group.

## Existing-data run

First validate the decomposition without loading state features:

```bash
cd /mnt/afs_2/houze/wanUpsampler

SCORED_DIR=/mnt/afs_2/houze/wanUpsampler/outputs/univ_sparse_action_phase3_v1/metrics/sparse_action_vbench \
bash UNIV_adaptor/scripts/run_univ_sparse_b4_action_router.sh check
```

Reuse the already extracted proxy and T5 embeddings:

```bash
SCORED_DIR=/mnt/afs_2/houze/wanUpsampler/outputs/univ_sparse_action_phase3_v1/metrics/sparse_action_vbench \
bash UNIV_adaptor/scripts/run_univ_sparse_b4_action_router.sh train-t5
```

TF-IDF is available as a cheap control:

```bash
SCORED_DIR=/mnt/afs_2/houze/wanUpsampler/outputs/univ_sparse_action_phase3_v1/metrics/sparse_action_vbench \
bash UNIV_adaptor/scripts/run_univ_sparse_b4_action_router.sh train-tfidf
```

Defaults are `UTILITY_LAMBDA=0.05`, `SOFT_TEMPERATURE=0.02`, and
`MAX_ITERATIONS=250`. Use a new `OUT_DIR` when changing any setting; the driver
refuses to mix incompatible requests.

## Comparisons

- `action_main_soft`: prompt-independent soft action surface;
- `prompt_prior_soft`: seed-averaged prompt/action prior;
- `action_main_state_residual_soft`: state residual over the global prior;
- `prompt_state_residual_soft`: proposed prompt prior plus state correction;
- `prompt_state_residual_shuffled`: within-prompt seed-shuffled control.

The main existing-video question is whether `prompt_prior_soft` improves over
`action_main_soft`. The state capacity gate additionally requires fusion to
beat both prompt-only and shuffled-state policies on the untouched 20-prompt
fresh development cohort.

`decomposition_diagnostics.json` reports prompt-mean versus seed-residual
variance. `model_cv.csv` records policy regret, soft cross-entropy, row MSE and
the selected regularization for every family. Policy decisions and paired
prompt-bootstrap intervals are stored in `policy_by_group.csv` and
`policy_summary.csv`.

## Evidence boundary

The current 21-dimensional state comes from the completed REFERENCE video, so
it remains a post-hoc proxy. Phase 3 also lacks complete matched single-axis
coverage. A positive result supports collecting a causally available early
latent; a negative result does not rule out richer latent state. The next new
video supplement should use matched `R/S/T/C` probes before adding selectively
sampled interactions.
