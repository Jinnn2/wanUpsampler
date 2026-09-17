# Phase 1 prompt-to-budget candidate shard

This shard starts the first experiment phase for prompt-guided multi-axis
allocation. It intentionally does not enumerate the Cartesian product of
temporal sampling, spatial scale, timestep skipping, cache reuse, and HR
refinement. The six candidates in
`configs/univ_prompt_budget_phase1.json` provide four anchors and two local
single-axis probes per prompt.

## Candidate roles

| id | display | role | changed axis |
| --- | --- | --- | --- |
| `P1_B15_BASE` | B15 | anchor | low-cost baseline |
| `P1_B20_BASE` | B20 | anchor | temporal/spatial baseline |
| `P1_B25_SPATIAL` | B25 | probe | spatial scale |
| `P1_B30_SPATIAL` | B30 | anchor | medium spatial budget |
| `P1_B35_HR` | B35 | probe | HR refinement steps |
| `P1_B40_SPATIAL` | B40 | anchor | higher spatial budget |

Using the planning proxy, the six candidates are approximately B15=0.150,
B20=0.201, B25=0.236, B30=0.305, B35=0.345, and B40=0.406. These are proxy
ordering values; measured warm latency is required before making speed claims.

The old B10--B30 extension is reference material only. Its actions are not
copied into this shard, and the new artifact ids are deliberately disjoint.

## Collection protocol

Start with a pilot of 40 training prompts and 20 validation prompts. For every
prompt/seed, collect the six candidates with the same prompt and seed. Archive
the endpoint state and the measured warm latency. Do not run the test split
until the action family and cost calibration are frozen.

On the eight-GPU generation host, the existing path contract can be used:

```bash
bash UNIV_adaptor/scripts/run_univ_prompt_budget_phase1_8gpu.sh plan
bash UNIV_adaptor/scripts/run_univ_prompt_budget_phase1_8gpu.sh all
```

The wrapper defaults to `PROMPT_LIMIT=80`, `SPLITS=train,validation`, eight
GPU ids `0,1,2,3,4,5,6,7`, and output root
`outputs/univ_prompt_budget_phase1_pilot_v1`. Override these with the same
environment variables used by the previous generation wrapper.

The first analysis should answer three questions:

1. Does increasing spatial scale from B25 to B30/B40 produce a prompt-dependent
   quality gain?
2. Does adding two HR steps at B35 improve quality enough to justify its cost?
3. Which prompt attributes predict the preferred axis (spatial versus HR)?

Only after these questions are answered should the next shard add temporal,
step-skip, or cache probes. Those probes should be selected per prompt or per
prompt cluster, rather than appended to every prompt.

## Cost accounting

The planning proxy is

```text
spatial_ratio^2 * temporal_ratio * true_lr_steps / 50 + hr_steps / 50
```

It excludes transition, endpoint I/O, VAE, codec, and fixed runtime overhead.
The target values in the JSON are therefore ordering targets, not measured
speed claims.
