# 360p -> 720p Four-Way Evaluation Protocol

## Cases

The completed panel contains four aligned prompt/seed cases:

1. `ori45_stage2`
2. `lora45_stage2`
3. `teacher50_interp`
4. `teacher50_stage2`

These cases answer two different questions and must not be collapsed into one
PSNR ranking. There is no real 720p ground truth. Using
`teacher50_stage2` as the reference for all four cases would favor outputs that
share the same Stage2 operator.

## Question A: Is LoRA45 useful in the full chain?

Compare:

```text
ori45_stage2  vs  lora45_stage2
reference: teacher50_stage2
```

This comparison is valid because all three videos use the same Stage2 model.
Only the clean latent entering Stage2 differs.

Primary metric:

- L1: lower is better and matches the earlier 360p clean-prediction evaluation.

Supporting metrics:

- temporal L1 against teacher motion deltas: lower is better;
- LPIPS and MSE: lower is better;
- PSNR and SSIM: higher is better;

Run the same original/LoRA comparison against two anchors:

1. `teacher50_stage2` is the primary, same-operator reference.
2. `teacher50_interp` is the secondary robustness reference.

The LoRA conclusion is strongest when L1, LPIPS, and temporal L1 improve in
the same direction under both references. The second anchor does not replace
the primary reference because it introduces an operator mismatch.

Recommended LoRA acceptance gate over the ten prompts:

1. L1 mean improves and LoRA wins at least 7/10 samples under the primary reference.
2. LPIPS mean improves and LoRA wins at least 7/10 samples.
3. Temporal L1 mean does not regress and LoRA wins at least 6/10 samples.
4. At least one of PSNR or SSIM improves in mean and wins at least 6/10.
5. L1 and LPIPS do not reverse direction under the interpolation reference.
6. Human review finds no new severe identity, geometry, or flicker failure.

If conditions 1, 2, or 6 fail, do not claim that the LoRA improves the full chain.

## Question B: Is learned Stage2 better than interpolation?

Compare:

```text
teacher50_interp  vs  teacher50_stage2
```

Both start from the same teacher50 clean LR latent. Since neither is ground
truth, pixel similarity between them cannot decide the winner. Use blinded
pairwise review as the primary result.

Score four dimensions for every prompt:

- Detail/clarity: 30% — texture, edges, small objects, readable structure.
- Artifact cleanliness: 25% — ringing, checkerboard, oversharpening, broken geometry.
- Temporal stability: 25% — flicker, texture crawl, edge breathing, frame-to-frame deformation.
- Structure/identity: 20% — subject identity, object shape, layout, motion continuity.

Allowed winner values are `left`, `right`, or `tie`. Review at native
resolution and hide the case identity where possible.

Recommended Stage2 acceptance gate:

1. `teacher50_stage2` is the overall winner on at least 6/10 prompts.
2. It wins detail on at least 6/10 prompts.
3. Artifact cleanliness and temporal stability each have no more losses than wins.
4. No severe failure occurs on any prompt.

Sharpness or high-frequency energy alone is not a pass condition: ringing and
flicker can increase both.

## Final deployment decision

Use `LoRA45 + Stage2` only when both gates pass:

```text
LoRA gate:     lora45_stage2 beats ori45_stage2 toward teacher50_stage2
Stage2 gate:   teacher50_stage2 beats teacher50_interp in blinded review
```

Also report runtime and peak VRAM separately. They are deployment costs, not
visual-quality metrics.

## Commands

Generate automatic LoRA metrics and the human-review CSV template:

```bash
bash changing_resolution/scripts/eval/evaluate_tail_skip_lora_stage2_360p_four_way.sh
```

LPIPS is enabled by default. If its pretrained dependency is unavailable on an
offline machine, run with `ENABLE_LPIPS=0`; L1, MSE, PSNR, SSIM, and temporal
L1 will still be produced.

Outputs are written under:

```text
outputs/changing_resolution_tail_skip_lora_stage2_four_way_360p/evaluation
```

The two numerical summaries are:

```text
lora_vs_teacher_stage2_summary.csv  # primary
lora_vs_teacher_interp_summary.csv  # robustness anchor
```
