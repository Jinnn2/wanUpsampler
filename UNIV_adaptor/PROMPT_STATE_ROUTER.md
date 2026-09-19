# Prompt prior + realized-state router audit

This development experiment asks whether a realized, seed-specific visual state
adds action-selection signal beyond a prompt prior. It reuses all quality and
latency labels from Sparse Action Phase 3 and reads only the common `REFERENCE`
video for each of the 267 prompt-seed groups. No candidate video is regenerated
or rescored.

The first state source is intentionally a **post-hoc proxy**: 16 grayscale
frames at 64x64 and 4 fps are decoded from the completed reference video and
reduced to appearance, spatial-detail and temporal-change statistics. This is
not a deployable early latent and must not be reported as one. Its purpose is a
cheap capacity gate before collecting true early denoising observations.

The action-conditioned model has two parts:

- action main effects plus pairwise interactions over the binary axes;
- regularized prompt/state interactions with the constant and each signed axis.

Thus the action surface grows quadratically and the conditional component grows
linearly with the number of action axes; neither requires observing all `2^d`
combinations for every prompt. Evaluation
is restricted to the three probes actually observed for a group plus the common
reference.

## Run on the existing Phase 3 root

```bash
cd /mnt/afs_2/houze/wanUpsampler

SCORED_DIR=/mnt/afs_2/houze/wanUpsampler/outputs/univ_sparse_action_phase3_v1/metrics/sparse_action_vbench \
bash UNIV_adaptor/scripts/run_univ_sparse_prompt_state_router.sh check

SCORED_DIR=/mnt/afs_2/houze/wanUpsampler/outputs/univ_sparse_action_phase3_v1/metrics/sparse_action_vbench \
bash UNIV_adaptor/scripts/run_univ_sparse_prompt_state_router.sh all-tfidf
```

The proxy extraction is CPU-only and content-binds the 267 reference videos by
SHA-256 by default. Set `VERIFY_VIDEO_HASHES=0` only for a quick local retry; the
choice is recorded in the state manifest.

For Wan-native T5 prompt embeddings:

```bash
SCORED_DIR=/mnt/afs_2/houze/wanUpsampler/outputs/univ_sparse_action_phase3_v1/metrics/sparse_action_vbench \
bash UNIV_adaptor/scripts/run_univ_sparse_prompt_state_router.sh all-t5
```

The T5 mode needs one GPU only for embedding 89 unique prompts. Model fitting is
CPU/NumPy. The default utility is
`delta_vbench5 - 0.05 * (time_ratio_to_reference - 1)`.
`OBSERVATION_COST_RATIO` can subtract a declared normalized observation cost
from every state-using policy; keep it at zero for the post-hoc capacity proxy.

## Required comparisons

The report contains:

- `action_main`: prompt-independent action surface;
- `prompt_only`: the prior;
- `state_only`: visual state without prompt;
- `prompt_state`: the proposed fusion;
- `prompt_state_shuffled`: states cyclically permuted among seeds of the same
  prompt, preserving content identity while destroying seed alignment.

The primary result is the untouched 20-prompt `fresh_development_holdout`; the
69 existing prompts use prompt-disjoint OOF predictions but also select ridge
strength, so they are selection diagnostics. Bootstrap units are prompts, not
the three seeds.

Only if fusion improves over both prompt-only and within-prompt shuffled state,
with acceptable harm after observation cost, should the next run collect true
early state. That follow-up needs 267 early probes rather than another 1,068
action videos; existing Phase 3 outcomes remain the labels. A formal paper claim
still requires a causally available observation and new prompt confirmation.
