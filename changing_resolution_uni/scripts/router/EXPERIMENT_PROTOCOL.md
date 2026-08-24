# Timestep Router Experiments 1-3

This protocol turns the exploratory lambda sweep into a leakage-controlled,
prompt-level confirmation experiment. The primary operating point is
`lambda=0.08`; lambda changes require a new validation selection run.

## Evidence boundary

- Development data may use `quality_valid_legacy_vbench5_v1` only with the
  explicit `--allow_estimated_latency` flag. Its speedups remain estimates.
- Confirmation requires `strict_vbench5_v1`, `formal_evidence=true`, and
  `warm_pipeline_seconds` for Native-HR and every candidate branch.
- The prompt is the independent statistical unit. The three generation seeds
  are averaged into one prompt label; they are not three independent samples.
- The current router is prompt-only. These experiments do not establish a
  latent-conditioned sequential stopping policy.

## Experiment 1: multi-seed robustness and prompt bootstrap

Hypothesis: the learned router reduces policy regret relative to the best
training-selected fixed timestep across training initializations.

Protocol:

1. Freeze one prompt-disjoint split with `SPLIT_SEED=42`.
2. Train B1, B3, and B4 with five initialization seeds:
   `42 100 2024 31415 27182`.
3. Selection runs evaluate validation only and emit one row per validation
   prompt in `router_validation_predictions.csv`.
4. Average each prompt across training seeds, then bootstrap prompts 10,000
   times. Report the 95% interval and the standard deviation of run means.
5. Use paired prompt deltas against the best fixed timestep. Positive deltas in
   `multiseed_paired_deltas.csv` always mean that the learned router is better.

Run:

```bash
PRIMARY_LAMBDA=0.08 \
DATASET_DIR=/mnt/afs_2/houze/wanUpsampler/data/changing_resolution_uni/oracle_dataset_500_quality_valid \
OUT_ROOT=/mnt/afs_2/houze/wanUpsampler/outputs/router_selection_500_quality_valid_lambda008 \
ALLOW_ESTIMATED_LATENCY=1 \
bash changing_resolution_uni/scripts/router/run_multiseed_router_selection.sh
```

Acceptance for proceeding to confirmation:

- all configured seeds finish with identical split/evidence metadata;
- B4 or another architecture is selected by mean validation policy regret;
- the paired regret improvement has a positive point estimate;
- test access remains `false` in `architecture_selection.json`.

Do not require Top-1 timestep accuracy to improve. Regret is primary; VBench-5,
latency, step MAE, and Top-1/Top-3 are secondary diagnostics.

## Experiment 1b: B4 soft distillation versus relative quality curves

After the original B4 passes Experiment 1, compare it against B4-Q without
changing the MLP widths, prompt split, initialization seeds, optimizer, or
evaluation metrics. B4-Q regresses the 13-point target
`candidate_vbench5 - step50_vbench5` with SmoothL1 loss. At evaluation time it
combines the predicted relative quality curve with the selected lambda and the
recorded normalized candidate latency. This isolates the supervision change
and does not access the test split.

Run in a new output root so the accepted B4 selection is not overwritten:

```bash
PRIMARY_LAMBDA=0.08 \
DATASET_DIR=/mnt/afs_2/houze/wanUpsampler/data/changing_resolution_uni/oracle_dataset_500_quality_valid \
OUT_ROOT=/mnt/afs_2/houze/wanUpsampler/outputs/router_b4_quality_curve_comparison_lambda008 \
ALLOW_ESTIMATED_LATENCY=1 \
bash changing_resolution_uni/scripts/router/run_multiseed_quality_curve_comparison.sh
```

Keep B4-Q only if its validation policy regret improves over B4, or is
statistically indistinguishable while providing the desired lambda-independent
quality representation. Estimated-latency results remain development evidence.
Use `selection/multiseed_reference_paired_deltas.csv` for the direct B4-Q
minus B4 comparison; positive values always mean B4-Q is better.

## Experiment 2: formal quality, measured latency, and router overhead

Hypothesis: the locked router remains useful after replacing branch estimates
with measured resident-process latency and including router inference overhead.

The confirmation command fails closed unless every record has formal VBench-5
provenance and measured `warm_pipeline_seconds`. It measures the selected
router at batch size 1 after warm-up, synchronizes CUDA around every repeat, and
adds median router latency to learned-policy latency and utility. Fixed policies
and the prompt oracle do not pay router overhead.

Required report fields:

- VBench-5 and the five individual dimensions from the formal dataset;
- measured branch latency plus router median/p90/p95 overhead;
- peak router memory;
- speedup versus matched measured Native-HR;
- prompt-bootstrap intervals and paired deltas versus best fixed.

Router overhead is measured separately from model/checkpoint loading. The same
GPU type, scheduler, frame count, resolution, prompt, and generation seeds must
be used for all branch timings.

## Experiment 3: validation-only architecture selection, one locked test

`summarize_multiseed_selection.py` writes `architecture_selection.json` using
only validation predictions. The confirmation launcher reads the model type,
lambda, and split seed from that file. It refuses to overwrite an existing
`router_benchmark_summary.json`. Immediately before the first test iteration it
also writes `test_access_guard.json`; a failed partial run therefore still
blocks an accidental second test read unless the explicit override is supplied.

Run exactly once after the selection manifest is frozen:

```bash
DATASET_DIR=/mnt/afs_2/houze/wanUpsampler/data/changing_resolution_uni/oracle_dataset_1k_strict \
SELECTION_JSON=/mnt/afs_2/houze/wanUpsampler/outputs/router_selection_500_quality_valid_lambda008/selection/architecture_selection.json \
OUT_DIR=/mnt/afs_2/houze/wanUpsampler/outputs/router_confirmation_1k_strict_lambda008 \
bash changing_resolution_uni/scripts/router/run_locked_router_confirmation.sh
```

These paths follow `configs/local_paths.sh` (`PROJECT_ROOT` defaults to
`/mnt/afs_2/houze/wanUpsampler`) and the existing legacy/strict router scripts.
The confirmation command still fails if `oracle_dataset_1k_strict` contains
`estimated_warm_pipeline_seconds`; its path is real and traceable, but it is not
declared confirmation-ready until every branch latency is measured.

The newer 1,500-prompt generator writes raw physical datasets to
`data/changing_resolution_uni/oracle_dataset_1500_8gpu/train` and `.../eval`.
They are not interchangeable with the single-directory `DATASET_DIR` above:
the current router loader expects one scored manifest with uniform seed
coverage, while the 1,500 design intentionally uses one train seed and three
evaluation seeds.

The confirmation output includes:

- `router_test_predictions.csv`;
- `router_overhead.json`;
- `confirmation_test_intervals.csv`;
- `confirmation_test_paired_deltas.csv`;
- `confirmation_bootstrap_report.json`;
- B4 token attribution under `token_attribution_b4/` when B4 is selected.

## B4 token attribution

The old B3 attribution `w^T h_i` is invalid for nonlinear B4. B4 attribution now
targets the probability-weighted expected candidate timestep. For token `i`,
the reported value is:

```text
expected_step(full prompt) - expected_step(prompt pooled without token i)
```

Positive values push toward a later switch/staying LR; negative values push
toward an earlier HR switch. Leave-one-out effects are local perturbation
effects and are not additive Shapley values. Rankings therefore support
exploratory interpretation only; they are not causal claims.
