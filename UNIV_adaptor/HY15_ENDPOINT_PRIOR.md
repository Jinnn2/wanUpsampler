# HY1.5 prompt method and budget pilot

Paper claim under test: with a fixed endpoint restoration protocol, a prompt
predicts the cross-seed expected utility of choosing spatial, temporal, or
cache acceleration and a main-stage budget. This is development evidence, not
a confirmation result for a trained prompt policy.

The protocol is frozen in `configs/hy15_endpoint_prior_v1.json`: 50 main
updates, RGB restoration where a latent axis was reduced, re-noise at sigma
0.2, then four full-resolution updates at `[0.2, 0.15, 0.1, 0.05, 0]`.
Main-stage proxy budgets 0.5 and 0.25 do not include RGB restoration or HR4.
Report actual end-to-end seconds for comparisons. Cache is uniform reuse of
the most recent guided velocity, not TeaCache.

The pilot contains 16 development prompts, four seeds and eight arms per
prompt-seed: native FULL50, FULL50+HR4, then S/T/C at both budgets. This is
512 videos. All four prompt families are development data; this pilot cannot
establish held-out-family generalization.

On the Linux server, use the same output root throughout:

```bash
bash UNIV_adaptor/scripts/run_hy15_endpoint_prior_8gpu.sh setup
bash UNIV_adaptor/scripts/run_hy15_endpoint_prior_8gpu.sh download
bash UNIV_adaptor/scripts/run_hy15_endpoint_prior_8gpu.sh plan
bash UNIV_adaptor/scripts/run_hy15_endpoint_prior_8gpu.sh check
bash UNIV_adaptor/scripts/run_hy15_endpoint_prior_8gpu.sh smoke
bash UNIV_adaptor/scripts/run_hy15_endpoint_prior_8gpu.sh generate
```

Generation verifies existing completion records and resumes missing jobs.
`generate` finalizes automatically after all 512 records pass integrity checks.
Each worker writes `outputs/hy15_endpoint_prior_v1/logs/gpu_<rank>.log`.

## Early inspection during generation

```bash
bash UNIV_adaptor/scripts/run_hy15_endpoint_prior_8gpu.sh partial-check
```

This CPU-only command verifies completed video hashes and prints action
coverage, complete eight-arm prompt-seed groups, represented factor cells,
and mean total/main/transition/refinement seconds. A video counts only when
its completion record exists. It can run while generation holds the GPU
launcher lock. With 180 videos the number of usable groups may be much lower
than 180/8 because workers can be partway through several groups.

To score available videos, let generation finish or stop it at a job boundary
and ensure its launcher lock is released. VBench uses the same eight GPUs.
Do not run generation and VBench at the same time on this machine.

```bash
bash UNIV_adaptor/scripts/run_hy15_endpoint_prior_8gpu.sh partial-score
```

The scorer takes a snapshot of **complete eight-arm groups only**, verifies
each record/video against the frozen plan, and scores that snapshot. Results
are under `outputs/hy15_endpoint_prior_v1/metrics/hy15_endpoint_vbench_partial/<snapshot-hash>/`.
It writes `snapshot.json`, `scores.json`, `quality_by_video.csv`,
`relative_to_full.csv`, `prompt_mean_targets.csv`,
`factor_summary.csv`, `st_pairs_by_seed.csv`, and `report.md`.
Re-running the same snapshot reuses matching VBench results. New complete
groups create a new snapshot and directory.

The first diagnostic is actual time by arm. Next compare paired S/T/C
quality at the same nominal budget and their actual latency, then inspect
whether the represented prompt factors have both winners and losers. A partial
snapshot with only a few prompts or one factor cell cannot validate a prompt
classifier. Report number of prompts and families along with every result;
seeds are repeated observations of a prompt, not independent prompts.

Once all videos are finalized:

```bash
bash UNIV_adaptor/scripts/run_hy15_endpoint_prior_8gpu.sh score-check
bash UNIV_adaptor/scripts/run_hy15_endpoint_prior_8gpu.sh score
```

The full-score output is separate at
`outputs/hy15_endpoint_prior_v1/metrics/hy15_endpoint_vbench/`.
