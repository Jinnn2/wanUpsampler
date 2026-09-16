# UNIV online policy existence pilot

This 8-GPU pilot tests whether useful sample-dependent decisions exist before
training a controller. It does not claim learned-policy performance.

## Question

The initial balanced low-resolution plan runs 12 fully computed Wan solver
positions. At that decision point the experiment asks whether a sample benefits
from:

1. changing the full-compute density over the remaining LR reference grid;
2. spending zero or four NFE on direct-sigma HR repair; or
3. abandoning the attempted prefix and restarting with a spatial-heavy or
   temporal-heavy LR geometry.

The six continuation cases form an exact `3 x 2` matrix:

```text
remaining LR full-compute positions: 8 / 20 / 38
HR repair:                         HR0 / sigma=.30, HR4
```

The other two cases use near-matched initial token densities:

```text
spatial-heavy:  spatial=.72, temporal=.50
temporal-heavy: spatial=.50, temporal=1.00
```

Initial-strategy headroom is computed only over the matched trio
`balanced/spatial-heavy/temporal-heavy` with remaining LR compute fixed at 20
and HR repair fixed at sigma `.30`, HR4. This prevents later LR/HR differences
from being misreported as evidence for prompt-level initial routing.
The manifest records each resolved LR latent shape and exact token ratio after
Wan temporal/spatial rounding, so the near-matched geometry assumption is
visible in the final report.

Every continuation case uses the same prompt, seed, coordinate noise, geometry,
and 12-step full-compute prefix. The analyzer requires exact decision-state and
predicted-clean hashes across the six cases. It also requires HR0/HR4 to share
the exact LR endpoint for each remaining-LR setting. A failed hash check stops
analysis rather than producing invalid counterfactual evidence.

## Run

The defaults target the existing remote Wan, LightX2V, VBench and 500-prompt
paths. Start with 16 prompts; the first prompt is retained as a timing warmup,
leaving 15 measured prompt clusters.

```bash
cd /mnt/afs_2/houze/wanUpsampler

bash UNIV_adaptor/scripts/run_univ_online_policy_existence_8gpu.sh check

LIMIT=16 \
OUT_ROOT=/mnt/afs_2/houze/wanUpsampler/outputs/univ_online_policy_existence_v1 \
bash UNIV_adaptor/scripts/run_univ_online_policy_existence_8gpu.sh plan

LIMIT=16 \
OUT_ROOT=/mnt/afs_2/houze/wanUpsampler/outputs/univ_online_policy_existence_v1 \
bash UNIV_adaptor/scripts/run_univ_online_policy_existence_8gpu.sh all
```

Use `generate`, `vbench`, and `analyze` to resume phases independently. The
manifest is immutable; changing prompts, source files, actions, seed, or output
scope under an existing root is rejected. Generation is resumable per complete
case. An interrupted incomplete case is cleared and restarted from its first
prompt, while completed cases are retained.
The manifest also records hashes for the transitive UNIV runner dependencies
and the LightX2V commit/tracked status; generation rejects a checkout change
after planning.

For a larger follow-up after checking pilot videos:

```bash
LIMIT=64 \
PROMPT_OFFSET=16 \
OUT_ROOT=/mnt/afs_2/houze/wanUpsampler/outputs/univ_online_policy_existence_64p_v1 \
bash UNIV_adaptor/scripts/run_univ_online_policy_existence_8gpu.sh all
```

Use a new output root whenever the selected prompt slice changes.

## Outputs

```text
<OUT_ROOT>/generation_manifest.json
<OUT_ROOT>/configs/<case>.json
<OUT_ROOT>/videos/<case>/*.mp4
<OUT_ROOT>/videos/<case>/*.mp4.univ.json
<OUT_ROOT>/videos/<case>/*.mp4.endpoint.pt
<OUT_ROOT>/timings/<case>.jsonl
<OUT_ROOT>/metrics/vbench_scores.json
<OUT_ROOT>/reports/POLICY_EXISTENCE.md
<OUT_ROOT>/reports/policy_existence.json
<OUT_ROOT>/reports/policy_existence_per_sample.csv
```

VBench is run once over all case videos and reports the five working quality
dimensions, Overall Consistency and Dynamic Degree. The utility uses only the
five-dimension mean. Overall Consistency stays separate because its scale is not
identical to the five quality scores; Dynamic Degree remains diagnostic because
larger motion is not always better.

The cost term uses synchronized LR, transition, HR and online-observation time
from the runner. It excludes text encoding, final decoding and experiment-only
endpoint checkpoint I/O. Prompt-level initial selection uses candidate
denoising cost without an online observation. A continuation decision uses a
common median balanced-prefix and observation cost plus its measured remaining
LR, transition and HR cost. A restart candidate adds the abandoned common
balanced prefix and observation exactly once to the alternative strategy's
full denoising cost. This assumes prompt embeddings can be reused and only the
retained candidate is decoded. It is a denoising-cost pilot, not yet a measured
end-to-end service policy.

The controller observation for every online action, including restart, is read
from the common balanced prefix. Alternative restart trajectories supply only
their outcome and full-denoising cost. The per-sample CSV identifies the
canonical balanced sidecar that stores the state, predicted-clean and prompt
context features.

For each configured lambda, the report compares:

- the best fixed continuation;
- the per-sample LR+HR sampled oracle;
- the best fixed single-attempt case;
- the three matched initial strategies under an identical LR/HR suffix;
- the per-sample single-attempt sampled oracle; and
- the restart-aware sampled oracle, including abandoned-prefix cost;
- its gain over the best fixed expanded decision; and
- the restart-only increment over the per-sample continuation oracle.

The initial and online existence flags require a positive clustered-bootstrap
lower bound and decisive action diversity. The restart flag is stricter: adding
restart must have a positive lower-bound gain beyond the per-sample continuation
oracle, and a restart action must win decisively. A negative pilot result means
that the tested action set did not expose stable headroom; it does not prove
that no useful policy exists elsewhere. A positive result establishes headroom
only. The next experiment must train a predictor from prompt and the recorded
online observation features on prompt-disjoint splits.

Each clustered-bootstrap replicate resamples prompts and reselects its own best
fixed action before computing oracle gain. This includes uncertainty from
choosing the fixed baseline on a finite pilot set.
