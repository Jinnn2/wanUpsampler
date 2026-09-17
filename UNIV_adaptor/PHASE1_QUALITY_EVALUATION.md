# Phase1 evaluation

Evaluate finalized standalone records with six candidates and no native teacher.

```bash
cd /mnt/afs_2/houze/wanUpsampler
bash UNIV_adaptor/scripts/run_univ_phase1_eval_8gpu.sh check
bash UNIV_adaptor/scripts/run_univ_phase1_eval_8gpu.sh all
```

Defaults match generation: output root `outputs/univ_prompt_budget_phase1_pilot_20260917`,
VBench `/mnt/afs_2/houze/VBench`, evaluator Python `/opt/conda/envs/vbench/bin/python`.
Override `OUT_ROOT`, `VBENCH_ROOT`, `VBENCH_PYTHON`, `WAN_PYTHON`, or `GPU_IDS` as needed.
`check` needs no GPU; `all` uses one eight-GPU distributed VBench batch.
The scorer reuses the existing strict content-bound VBench cache when inputs match.
`report` regenerates CSV/Markdown from saved scores without running VBench.

Outputs under `metrics/phase1_quality`:
- `scores.json`: video-level dimensions, input identity, VBench provenance.
- `quality_by_video.csv`: complete quality and observed pipeline costs.
- `quality_by_prompt.csv`: seed-averaged per-prompt/action values.
- `quality_cost_summary.csv`: split/action summaries.
- `paired_gains.csv`: B20->B25, B25->B30, B30->B40, B30->B35 deltas.
- `report.md`: readable quality/cost and paired gain summary.

Five working dimensions: subject consistency, background consistency, motion
smoothness, aesthetic quality, imaging quality. Their unweighted average is a
working proxy, not the official VBench total score. Dynamic degree and overall
consistency are reported separately to detect static-video and semantic failures.
No native-HR speedup, native fidelity, causal prompt predictability or fixed-budget
allocator superiority is inferred. This small pilot is descriptive; repeated
seeds are averaged before prompt-level comparisons. Inspect per-dimension results
and qualitative videos before selecting the next action probes.

Runtime summaries distinguish unavailable stage timings from zero. LR+HR compute
is reported where both fields exist; VAE timing remains null unless explicitly
recorded. Pipeline latencies are observed generation times, not separately warmed
benchmarks. No model generation is invoked by this workflow.
