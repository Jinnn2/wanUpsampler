# AAAI-27 experiment suite

This directory provides a resumable layer over the existing training and
evaluation scripts. Run it on the Linux GPU host where `/mnt/afs_2/houze`
and the existing `outputs/` tree are available.

Paper-facing terminology is functional: TAA (internal LoRA labels), CLL
(internal Stage2 labels), HTR (re-noise plus HR suffix), JTSL (internal Stage3),
Native-HR (Full-HR), and VBench-5 (the internal `Quality5` field). Commands,
registries, case names, and CSV schemas deliberately retain their original
internal labels for reproducibility.

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

### Final closure experiments: step40, efficiency, and step45 review

The final step40 experiment evaluates strengths 0.5, 0.75, and 1.0 at three
levels: LR endpoint distance, end-to-end VBench, and blinded temporal/detail
preference. The endpoint task also writes paired bootstrap intervals and exact
sign-test results. The end-to-end factorial reuses Base cases and generates one
LoRA+interpolation and one LoRA+Stage2 case per strength.

```bash
python paper/aaai27/experiments/run_experiments.py run \
  --task wan50_lora40_endpoint_strength
python paper/aaai27/experiments/run_experiments.py run \
  --task wan50_lora40_strength_vbench
python paper/aaai27/experiments/run_experiments.py run \
  --task wan50_lora40_strength_vbench_statistics
python paper/aaai27/experiments/run_experiments.py run \
  --task wan50_lora40_strength_review_package
```

Give three raters separate copies of
`outputs/aaai27_experiments/wan50_step40_strength/review/human_ratings.csv`.
After they finish, merge and summarize them:

```bash
python paper/aaai27/experiments/aggregate_human_review.py merge \
  --factorial-root outputs/aaai27_experiments/wan50_step40_strength \
  --rater r1=/path/step40_strength_r1.csv \
  --rater r2=/path/step40_strength_r2.csv \
  --rater r3=/path/step40_strength_r3.csv
```

The summary includes raw vote counts, prompt-level majority preferences with
bootstrap confidence intervals and exact sign tests, and Fleiss' kappa for
each comparison/dimension. Select the final step40 strength from the complete
endpoint/VBench/human evidence, then pass it to the unified Pareto benchmark
through the environment. Both final Wan50 strengths default to 0.75; keep the
explicit environment overrides below when freezing a reproducible run. The
default 11 cases share model, prompts, seeds, frame count, and output
resolution:

- Native-HR50;
- LightX2V changing-resolution handoffs at steps 40, 45, and 48;
- TALH handoffs at steps 40 and 45;
- Full-LR50 + CLL with 0, 1, 2, and 5 additional HR refinements;
- one full RALU Quality adaptation with 5 LR, 6 mixed-resolution, and 7 HR
  evaluations.

Every Endpoint case retains the canonical 50-step LR schedule, lifts the clean
endpoint once, and then performs exactly K extra evaluations on the final K
timesteps of the HR-shifted schedule. The RALU case implements the complete
three-stage region-adaptive pipeline: VAE/Canny top-r edge selection, packed
mixed-resolution Wan `1x2x2` latent tokens, official integer/half-offset
position IDs, unit noise on unchanged tokens, correlated `I-c11^T` noise on
expanded four-token groups, both analytical noise/timestep transitions, and
geometry A (368x640 to aligned 736x1280, followed by a patch-aligned crop to
720x1248 at the second handoff).

```bash
WAN50_LORA40_STRENGTH_FINAL=0.75 WAN50_LORA45_STRENGTH_FINAL=0.75 \
python paper/aaai27/experiments/run_experiments.py run \
  --task wan50_final_quality_efficiency_vbench

WAN50_LORA40_STRENGTH_FINAL=0.75 WAN50_LORA45_STRENGTH_FINAL=0.75 \
python paper/aaai27/experiments/run_experiments.py run \
  --task wan50_final_quality_efficiency_benchmark
```

The benchmark uses one warm-up and five measured fresh processes per case.
It records LR/mixed/HR/total denoiser evaluations, cold-start latency, raw repeats,
whole-process peak GPU memory, individual VBench components, and the explicitly
labeled five-dimension convenience mean.

Collect the manifest-driven 11-case suite independently of the larger AAAI
inventory with:

```bash
python paper/aaai27/experiments/collect_quality_efficiency.py \
  --suite-root outputs/aaai27_experiments/quality_efficiency_final_v2 \
  --probe-videos \
  --require-metrics \
  --require-timing \
  --archive
```

Add `--include-videos` for a transferable archive containing all 110 MP4s.
The collector requires each expected MP4 to be larger than 1 KiB, optionally
validates its video stream with `ffprobe`, enforces exact case coverage in the
paper-facing warm timing and VBench summaries, snapshots the RALU
implementation, and records but excludes legacy `ralu_nt40/45/48` directories.

After the zero-strength dynamic-LoRA bypass optimization, rerun only the two
affected TALH timing cases and merge them into the existing 11-case table with:

```bash
python paper/aaai27/experiments/rerun_optimized_taa_timing.py \
  --suite-root outputs/aaai27_experiments/quality_efficiency_final_v2 \
  --python /path/to/wan/python \
  --warmup 1 \
  --repeats 5
```

The script inherits the physical GPU identifier from the existing summary, so
the replacement rows remain comparable with the other nine cases. Raw timing
rows are checkpointed after every fresh process; rerunning the same command
resumes missing warm-up or measured repeats. Outputs are written under
`optimized_taa_timing/` and the original CSVs are never overwritten.

For a smoke test or a partial rerun, select method families explicitly:

```bash
python paper/aaai27/experiments/run_final_quality_efficiency.py check \
  --methods lightx2v endpoint ralu \
  --lightx2v-handoff-steps 40 45 48 \
  --endpoint-refinement-steps 0 1 2 5 \
  --ralu-stage-steps 5 6 7 \
  --ralu-end-times 0.30 0.45 1.0
```

Step45 ratings are isolated from the existing step40 package, so preparing
them cannot overwrite earlier ballots or private keys:

```bash
python paper/aaai27/experiments/run_experiments.py run \
  --task wan50_step45_review_package

python paper/aaai27/experiments/aggregate_human_review.py merge \
  --factorial-root outputs/aaai27_experiments/factorial_wan50 \
  --review-name step45 \
  --rater r1=/path/step45_r1.csv \
  --rater r2=/path/step45_r2.csv \
  --rater r3=/path/step45_r3.csv
```

The step45 ballot is under `review/step45/`, with its hidden key under
`_private/step45/`. Audit the three final groups with `--group final_step40`,
`--group final_step45`, and `--group final_efficiency`; manual human tasks are
complete only after their three ballots have been merged.

### Distill4 endpoint-domain quality--efficiency suite

The Distill4 main suite deliberately excludes interpolation handoff step 1.
It contains 18 configurations: Native-HR4, interpolation handoffs at steps 2
and 3, the complete step-3 TAA/CLL factorial, and the Cartesian product of
endpoint refinement budgets `0/1/2/4` with three lifting domains:

- `stage2`: the trained clean-latent lifter;
- `interp`: trilinear latent interpolation;
- `rgb`: Wan VAE decode, Real-ESRGAN x2, center crop from 736x1280 to
  720x1248, and encoding with the same Wan VAE.

The RGB path follows MrFlow's released x2 Real-ESRGAN protocol. Install the
official Real-ESRGAN/BasicSR package in the Wan environment and download
`RealESRGAN_x2plus.pth`. The `bicubic` RGB backend is only for integration
smoke tests and must not be reported as the MrFlow-style result.

Validate paths and materialize all configs before launching the GPU run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash changing_resolution_distill/scripts/eval/run_distill4_final_18case_4gpu.sh check

CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash changing_resolution_distill/scripts/eval/run_distill4_final_18case_4gpu.sh run
```

The generation launcher runs cases in parallel across the four physical GPUs.
It uses longest-estimated-cost-first packing, writes the exact assignment to
`generation_schedule.json`, and exposes only one GPU to each subprocess. Thus
each case remains ordinary single-GPU inference and produces the same
prompt/seed outputs as a serial launch. It validates all model paths and the
Real-ESRGAN Python imports, records resolved settings under `logs/`, and resumes
existing videos by default. If `RealESRGAN_x2plus.pth` is absent, it atomically
downloads the official v0.2.1 release asset and verifies its expected byte
size. Set `AUTO_DOWNLOAD_REALESRGAN=0` for strict offline mode. Override the
other environment variables for non-default paths; set `SKIP_EXISTING=0` to
force regeneration.

Prepare/run VBench and then create the quality-linked benchmark spec:

```bash
python paper/aaai27/experiments/run_vbench_factorials.py prepare \
  --factorial-root outputs/aaai27_experiments/quality_efficiency_distill4
python paper/aaai27/experiments/run_vbench_factorials.py run \
  --factorial-root outputs/aaai27_experiments/quality_efficiency_distill4 \
  --vbench-root /path/to/VBench --python /path/to/vbench/python
python paper/aaai27/experiments/run_distill4_quality_efficiency.py benchmark-spec \
  --realesrgan-x2-checkpoint /path/to/RealESRGAN_x2plus.pth
```

Finally run resident-model timing. Each case uses one initialization, one
warm-up video, and five measured videos. The endpoint `1hr` pairs are the main
test of early `3 LR + 1 HR` handoff against endpoint-first `4 LR + 1 HR`:

```bash
python paper/aaai27/experiments/benchmark_warm_quality_efficiency.py \
  --suite-root outputs/aaai27_experiments/quality_efficiency_distill4 \
  --gpu 0 --warmup 1 --repeats 5
```

Keep the latency benchmark on one otherwise-idle GPU. Running the timing cases
concurrently would introduce shared CPU, storage, and VAE/SR contention into
the reported time differences; the four-GPU path is for video generation.

The same pipeline is registered as the `distill4_final_efficiency` group in
`experiment_manifest.json`.

Freeze the completed Distill4 suite into a standalone, checksummed result
archive after both VBench and the warm benchmark finish:

```bash
python paper/aaai27/experiments/collect_quality_efficiency.py \
  --suite-root outputs/aaai27_experiments/quality_efficiency_distill4 \
  --output-root exports/distill4_quality_efficiency_final_YYYYMMDD \
  --probe-videos \
  --require-metrics \
  --require-timing \
  --archive
```

This compact archive contains the 18-case manifest and configs, VBench and
paired statistics, warm timing tables and audit manifests, checkpoint
fingerprints, generation schedule, video checksum inventory, and the relevant
implementation files. Add `--include-videos` for the internal master archive
containing all 180 MP4s; omit it for a lightweight paper-results bundle.

## 4. Export a frozen result bundle

Run collection immediately before export. The exporter requires an exact
allowlist match: it stops if a declared omission unexpectedly exists or if any
additional issue appears. For the final AAAI snapshot where the two controlled
architecture/loss ablations and unseen-prompt generalization were intentionally
not run:

```bash
python paper/aaai27/experiments/collect_results.py

python paper/aaai27/experiments/export_results.py \
  --output-root exports/aaai27_final_YYYYMMDD \
  --allow-missing sources.lora_architecture_loss \
  --allow-missing sources.stage2_architecture_loss \
  --allow-missing sources.generalization \
  --include-videos \
  --include-checkpoints \
  --archive
```

The output directory must be new and must be outside `outputs/`. It contains:

- `core/`: inventory, paper tables, normalized CSVs, and declared omissions;
- `evidence/`: canonical and legacy result trees, factorials, and ablations;
- `models/`: final referenced checkpoints when `--include-checkpoints` is set;
- `provenance/`: task state, tracked code, Git state/diff, environment, and the
  original-to-exported path map;
- `SHA256SUMS` and `export_manifest.json` for integrity and export settings.

Videos are excluded unless `--include-videos` is set. `_private` review keys
and task logs are excluded by default; add `--include-private` and
`--include-logs` only for the internal master archive. Do not publish that
private archive as anonymous supplementary material.

If the canonical `outputs/aaai27_experiments` directory is accidentally
deleted after the export directory was completed, verify and preview a restore
without writing anything:

```bash
python paper/aaai27/experiments/restore_results.py \
  --export-root exports/aaai27_final_YYYYMMDD
```

On the same filesystem, restore videos with hard links to avoid allocating a
second copy of the large immutable media. Mutable JSON, CSV, manifests, and
other metadata are copied so later collection cannot modify the export backup:

```bash
python paper/aaai27/experiments/restore_results.py \
  --export-root exports/aaai27_final_YYYYMMDD \
  --execute --hardlink
```

The restore verifies every entry in `SHA256SUMS`, accepts only paths that were
originally under the recorded canonical result root, assembles them in a new
temporary directory, and atomically renames that directory into place. It
refuses to overwrite an existing result root.

## 5. Merge a base core with a closure archive

When a closure run intentionally exports only newly completed evidence, merge
it with the last verified base `core` before editing paper tables:

```powershell
python paper/aaai27/experiments/integrate_result_snapshots.py `
  --base-core "C:\path\to\aaai27_final_20260717\core\core" `
  --incremental-archive "C:\path\to\aaai27_closure_20260718_incremental.tar.gz" `
  --output-root "paper\aaai27\results\integrated_20260718"
```

The merger verifies every archive checksum in the tar stream, so raw VBench
files whose timestamp names contain Windows-illegal colons do not need to be
extracted. It prefers complete closure tables, falls back to the base core for
declared omissions, unions factorial coverage, and recomputes the final
step45 strength=0.75 paired endpoint statistics from raw samples. The generated
`integration_manifest.json` records provenance, final checkpoint strengths,
remaining evidence gaps, and interpretation constraints.
