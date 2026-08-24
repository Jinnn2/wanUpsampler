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
DATASET_DIR=/path/to/router_ready_dataset \
bash changing_resolution_uni/scripts/router/run_multiseed_router_selection.sh
```

Acceptance for proceeding to confirmation:

- all configured seeds finish with identical split/evidence metadata;
- B4 or another architecture is selected by mean validation policy regret;
- the paired regret improvement has a positive point estimate;
- test access remains `false` in `architecture_selection.json`.

Do not require Top-1 timestep accuracy to improve. Regret is primary; VBench-5,
latency, step MAE, and Top-1/Top-3 are secondary diagnostics.

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
DATASET_DIR=/path/to/formal_measured_router_dataset \
SELECTION_JSON=outputs/router_selection_lambda008/selection/architecture_selection.json \
bash changing_resolution_uni/scripts/router/run_locked_router_confirmation.sh
```

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
