# AAAI-27 experiment suite

This directory provides a resumable layer over the existing training and
evaluation scripts. Run it on the Linux GPU host where `/mnt/afs_2/houze`
and the existing `outputs/` tree are available.

## 1. Audit existing remote results

```bash
cd /mnt/afs_2/houze/wanUpsampler
python paper/aaai27/experiments/run_experiments.py audit
```

Paths can be overridden without editing the manifest. For example:

```bash
WAN50_LORA40_CKPT=/other/run/step40.safetensors \
python paper/aaai27/experiments/run_experiments.py audit
```

The audit counts only non-empty expected artifacts. Existing videos in the
old four-way result folders are hard-linked into the canonical factorial
folder when prompt, seed, method, and handoff step match.

## 2. Run experiments incrementally

Start with paired Stage2 operator measurements:

```bash
python paper/aaai27/experiments/run_experiments.py run --group operator
```

Then generate the complete Base/LoRA x interpolation/Stage2 factorial:

```bash
python paper/aaai27/experiments/run_experiments.py run --group factorial
```

For a persistent remote run:

```bash
bash paper/aaai27/experiments/tmux_run_experiments.sh operator
bash paper/aaai27/experiments/tmux_run_experiments.sh factorial
```

Run one task or preview commands:

```bash
python paper/aaai27/experiments/run_experiments.py run --task wan50_factorial --dry-run
python paper/aaai27/experiments/run_experiments.py run --task wan50_factorial
```

Commands are sequential and fail closed. Re-running is safe: completed
evidence is reused, and individual video runners skip non-empty files. Logs
and task state are written to `outputs/aaai27_experiments/_state/`.

## 3. Blinded review and collection

```bash
python paper/aaai27/experiments/run_experiments.py run --task blind_review_package
python paper/aaai27/experiments/run_experiments.py collect
```

Collection writes three complementary artifacts under
`outputs/aaai27_experiments/`:

- `result_inventory.json`: versioned machine-readable inventory, strict
  factorial filename/seed/config validation, provenance, and missing evidence;
- `paper_tables.md`: operator, endpoint, strength, transfer, factorial,
  timing, ablation, VBench, and human-review tables;
- `compiled_tables/*.csv`: normalized tables for direct paper import.

Missing evidence is reported but does not stop normal collection. Once every
required result is expected to exist, use strict mode as the final gate:

```bash
python paper/aaai27/experiments/collect_results.py --strict
```

Give raters only `review/human_ratings.csv` and `review/blinded/`. Keep
`_private/human_review_key.csv` hidden. After rating, place the completed file
at `review/human_ratings_completed.csv`.

VBench remains an external task because the repository does not vendor an
official VBench environment. Put its per-video and aggregate JSON outputs in
each factorial folder's `metrics/` directory; `audit` will then mark that
evidence complete.

### Canonical VBench execution

Pin an official VBench checkout and override `VBENCH_ROOT` when it is not at
the manifest default. The preparation step verifies every expected factorial
video and writes the required absolute-video-path-to-prompt JSON mappings.
VBench rejects PyTorch builds newer than CUDA 12.1, so keep it in a separate
environment and set `VBENCH_PYTHON`; do not downgrade the Wan inference
environment.

```bash
python paper/aaai27/experiments/run_experiments.py run --task vbench_factorial_inputs

VBENCH_ROOT=/path/to/VBench \
VBENCH_PYTHON=/path/to/vbench/environment/bin/python \
python paper/aaai27/experiments/run_experiments.py run --task vbench_factorials
```

`run_vbench_factorials.py` evaluates each factorial case separately using the
six dimensions supported by VBench custom-input mode and creates one canonical
`metrics/vbench_v1_custom.json` per family. It refuses to collect a family if
any case lacks numeric official output.

### Independent human ratings

Create one copy of `human_ratings.csv` for each rater. Each rater must fill all
winner fields with `A`, `B`, or `tie`, confidence with an integer from 1 to 5,
and severe failure with `A`, `B`, or `neither`. Merge three completed ballots
without exposing `_private/human_review_key.csv`:

```bash
python paper/aaai27/experiments/aggregate_human_review.py merge \
  --factorial-root outputs/aaai27_experiments/factorial_wan50 \
  --rater r1=/path/wan50_rater1.csv \
  --rater r2=/path/wan50_rater2.csv \
  --rater r3=/path/wan50_rater3.csv

python paper/aaai27/experiments/aggregate_human_review.py merge \
  --factorial-root outputs/aaai27_experiments/factorial_distill4 \
  --rater r1=/path/distill_rater1.csv \
  --rater r2=/path/distill_rater2.csv \
  --rater r3=/path/distill_rater3.csv

python paper/aaai27/experiments/run_experiments.py audit --task human_blind_review
```

The merge validates that every rater judged every blind pair exactly once.
The summary then unblinds locally and writes preference and severe-failure
statistics while retaining the original completed ratings.

### Controlled ablation registries

The LoRA and Stage2 table tasks consume registries rather than silently mixing
unmatched legacy runs. A LoRA registry entry has this shape:

```json
{
  "variants": [{
    "axis": "rank",
    "variant": "qkvo_ffn_rank16_main_loss",
    "target_modules": "qkvo+ffn",
    "rank": 16,
    "loss": "main",
    "train_steps": 10000,
    "train_seed": 202707,
    "lora_strength": 0.75,
    "checkpoint": "/absolute/path/step_0010000.safetensors",
    "metrics_csv": "/absolute/path/strength_metric_summary.csv",
    "columns": {"metric": "metric", "value": "lora_mean", "samples": "samples", "better": "better"}
  }]
}
```

Place at least six controlled variants covering `target_modules`, `rank`, and
`loss` in
`outputs/aaai27_experiments/ablations/lora_registry.json`. Stage2 uses the same
layout but covers `architecture` and `loss`, and replaces the LoRA fields with `architecture`, `prediction_mode`,
`loss`, `train_steps`, and `train_seed`; map `columns.value` to the Stage2 mean
column. Store it as `stage2_registry.json`. Then run:

```bash
python paper/aaai27/experiments/run_experiments.py run \
  --task lora_architecture_loss_ablation
python paper/aaai27/experiments/run_experiments.py run \
  --task stage2_architecture_loss_ablation
```

Both outputs include checkpoint SHA-256 provenance.

### Quality-efficiency benchmark

After both canonical VBench files exist, generate the four final fresh-process
cases automatically:

```bash
python paper/aaai27/experiments/run_experiments.py run \
  --task quality_efficiency_spec
```

The cases are Wan50 step45 and Distill4 step3, each comparing `base_interp`
against the selected `lora_stage2` configuration. Quality is linked from the
five continuous custom-input VBench dimensions; dynamic degree remains a
separate metric. Each benchmark command uses one fixed prompt/seed, disables
skip-existing, and runs in a fresh Wan process. Run on an otherwise idle GPU
from the base Wan environment, not the isolated VBench environment:

```bash
python paper/aaai27/experiments/run_experiments.py run \
  --task peak_memory_and_loading_overhead
```

The benchmark uses one warm-up and five measured fresh processes per case,
reporting mean/std/median wall time, raw repetitions, and peak `nvidia-smi`
memory above the pre-command baseline. This is cold-start single-video
end-to-end cost and whole-process GPU memory, not PyTorch allocator-only memory.

### Unseen-prompt generalization

The repository includes a frozen 20-prompt set covering motion, camera motion,
identity/structure, fine detail, and occlusion/lighting. Generate 160 videos:

```bash
python paper/aaai27/experiments/run_experiments.py run --task generalization_videos
VBENCH_ROOT=/path/to/VBench \
python paper/aaai27/experiments/run_experiments.py run --task generalization_vbench
python paper/aaai27/experiments/run_experiments.py run --task generalization_review_package
```

Obtain and merge three ratings per family using the same merge command above,
but point `--factorial-root` at the two roots under
`outputs/aaai27_experiments/generalization/`. Finally:

```bash
python paper/aaai27/experiments/run_experiments.py run \
  --task generalization_and_failures
```

For paired metric columns, produce a bootstrap confidence interval and exact
sign-test result with:

```bash
python paper/aaai27/experiments/paired_statistics.py \
  --input path/to/metrics.jsonl \
  --a-field base_lpips --b-field talh_lpips \
  --output path/to/lpips_stats.json --lower-is-better
```
