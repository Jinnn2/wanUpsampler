# Phase 2: quality, measured cost, and prompt preferences

This evaluates the finalized v2 dataset: 140 train prompt/seed groups and 20 validation prompts × 3 seeds, each with native HR50 plus five P2 candidates (1200 videos). It does not invoke prepare, generate, or finalize, and does not require the current generation config/git revision to match the old run. The frozen manifest/plan and record/video/optional sidecar identities are checked. Record coverage must exactly match the frozen train/validation assignments.

## Remote commands

If the output directory is uncertain, or `check` reports missing records, first run this read-only inventory:

```bash
bash UNIV_adaptor/scripts/run_univ_phase2_eval_8gpu.sh locate
```

It inspects sibling output roots, validates Phase2 manifests/plans, and shows expected/found/missing/extra train and validation record counts. `OUT_ROOT`, if set, is marked as selected. `SEARCH_ROOT` optionally specifies the directory to scan; otherwise the selected root's parent (or `PROJECT_ROOT/outputs`) is scanned. Choose the intended completed run (normally train 140/140 and validation 60/60), then run `check` to verify actual artifact identities. Coverage alone does not prove video integrity. The inventory never switches roots, regenerates, finalizes, or moves records. Missing records in one root do not establish that videos are missing from all roots.

Sync the new evaluation files to `/mnt/afs_2/houze/wanUpsampler` first. Set `OUT_ROOT` to the directory that actually completed finalize; if you used `_chunk25` or another suffix, use that directory instead. Do not create a new generation root.

```bash
cd /mnt/afs_2/houze/wanUpsampler
export OUT_ROOT=/mnt/afs_2/houze/wanUpsampler/outputs/univ_prompt_budget_phase2_20260918

# CPU checks + measured runtime tables (no VBench dependency)
bash UNIV_adaptor/scripts/run_univ_phase2_eval_8gpu.sh check

# Score all videos on 8 GPUs and produce the analysis
bash UNIV_adaptor/scripts/run_univ_phase2_eval_8gpu.sh all
```

The launcher probes working Python environments; it does not require `/opt/conda/envs/vbench/bin/python`. Override with `VBENCH_PYTHON=/opt/conda/bin/python` if this is your VBench environment. An explicit invalid override fails with the original import/CUDA error. `WAN_PYTHON` controls CPU check/report modes. `GPU_IDS` defaults to `0,1,2,3,4,5,6,7` and must contain eight distinct IDs.

VBench runs one distributed eight-GPU job per dimension, sequentially. Each completed dimension has a content/provenance-bound cache; rerunning `all` resumes by reusing completed matching dimensions. An incomplete dimension restarts. Model weights/dependencies must already be available or downloadable in your VBench installation. Scoring modes verify eight visible GPUs, while `check`/`report` need no CUDA.

```bash
# Rebuild reports from saved scores without scoring again
bash UNIV_adaptor/scripts/run_univ_phase2_eval_8gpu.sh report

# Optional absolute budget caps, in seconds; choose before examining validation
BUDGETS_SECONDS="80 100 120 150" \
bash UNIV_adaptor/scripts/run_univ_phase2_eval_8gpu.sh report
```

`score` only scores; `all` scores and reports. `TIE_EPSILON` defaults to `0.001` VBench5 units. `FORCE_RESCORE=1` explicitly bypasses caches. `EVAL_OUT` overrides the evaluation directory. An exclusive `.evaluation.lock/owner.json` prevents concurrent runs; after an abnormal termination, verify the recorded host/PID is no longer running before manually removing that lock directory. The launcher propagates scoring failures through `tee`.

## Outputs

All outputs default to `OUT_ROOT/metrics/phase2_quality/`:

| File | Meaning |
|---|---|
| `evaluation_inputs.json`, `action_catalog.json` | Frozen input identity, prompts, actual actions and resolved schedules |
| `runtime_by_video.csv`, `runtime_summary.csv` | Paired native speedups; mean and median observed latency, available in `check` |
| `scores.json`, `vbench/` | Seven dimensions, provenance and resumable per-dimension score runs |
| `quality_by_video.csv`, `quality_by_prompt.csv` | Raw and seed-averaged quality, cost, native-relative deltas |
| `quality_cost_summary.csv` | Separate train/validation summaries with equal prompt weights |
| `prompt_preferences.csv`, `pairwise_preferences.csv` | Five-action ranking, top-two margin, tie counts, pairwise cross-seed signs |
| `seed_stability.csv` | Winner agreement and leave-one-seed-out quality transfer on multi-seed prompts |
| `train_calibration.csv` | Train-only mean latency/quality for each action |
| `budget_oracle_summary.csv`, `budget_oracle_by_prompt.csv` | Budget feasibility, coverage and paired fixed/oracle comparisons |
| `analysis_settings.json`, `report.md` | Analysis choices and readable findings |

For review, download the JSON/CSV/Markdown tables; `inputs/` holds staged hardlinks or copies of all 1200 videos and need not be downloaded. Symlinks are rejected because VBench resolves paths when matching custom prompts. Check verifies hashes, not full video decoding; decoder failures are surfaced during VBench scoring.

## Statistical interpretation

VBench5 is an arithmetic working proxy over subject consistency, background consistency, motion smoothness, aesthetic quality and imaging quality. It is not the official VBench total. Dynamic degree and overall consistency are reported separately; inspect them when a combination appears to improve quality, since static output can score well on consistency/smoothness.

Native HR50 provides a same-prompt/seed generation baseline for speedup and score deltas. Different spatial/temporal noise geometry means it is not pixel ground truth. This script does not fabricate `native_fidelity`, overwrite generated records, or declare them formal training-ready records.

All overall quality summaries first average seeds per prompt. Validation therefore has 20 statistical units, not 60. Preference stability includes deterministic top-1 agreement plus epsilon-aware pairwise wins/ties. Leave-one-seed-out (LOSO) selects using the other seeds of the same prompt and evaluates the held-out seed, compared to the global action selected on train. LOSO is an unconstrained diagnostic of repeatable preferences; it is not a learned prompt predictor or a fixed-budget result.

Default budget caps are the five train action mean-latency knots. For each cap, freeze the eligible actions using only train mean latency, and choose the fixed baseline by train mean VBench5. At evaluation, an eligible action must also fit that prompt's observed seed-mean latency. The oracle selects the highest-quality feasible action in this frozen action set. It can be slower than the fixed action, but both must satisfy the cap; this is a cap comparison, not exactly equal latency.

The reported gain uses only the common prompts where fixed and oracle are feasible, and always reports paired coverage, oracle coverage, and fixed violation rate. Low coverage cannot support a whole-dataset improvement claim. Caps below all train action costs produce explicit `NA` comparisons, not a fallback that violates budget. Costs are means across the prompt's seeds, not hard per-video latency guarantees; consult per-video times for outliers. Timing comes from generation logs, not a dedicated warmed benchmark.

Paired bootstrap intervals resample prompts (2000 replicates, fixed RNG seed). They describe uncertainty in the empirical hindsight gap; oracle quality selection bias remains. A reliable positive gap and cross-seed stability motivate training a prompt-only allocator, whose gains must then be measured with an independent evaluation. Do not tune caps/features on validation and then claim it is untouched. Phase 2's first-161-prompt selection overlaps earlier Phase 1 splits: do not pool Phase 1 validation into an independent test of a Phase 2-trained model.

`P2_B25_SKIP` still keeps solver steps and reuses cached predictions; this run does not isolate true timestep skipping. `B25/B30` names are identifiers, not calibrated cost bins. The candidates also vary HR switch timing. Inspect `action_catalog.json` for actual schedules rather than attributing each action's outcome to a single mechanism.
