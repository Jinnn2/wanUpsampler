# Full-LR direct-sigma HR refinement

This experiment completes all 50 Wan solver updates on the reduced
spatio-temporal grid, lifts the clean endpoint with `dvg_latent_anchor`, and
then starts independent high-resolution refinement trajectories from explicit
noise strengths. It tests the MrFlow scheduling idea while keeping the
transition comparable with the earlier fixed-boundary and fixed-total runs.

The default matrix contains a transition-only control and nine refinement
branches:

| Sigma | HR evaluations |
| ---: | --- |
| 0 | 0 |
| 0.12 | 1, 2, 4 |
| 0.20 | 1, 2, 4 |
| 0.30 | 1, 2, 4 |

Each nonzero branch uses a linear direct-sigma grid from its stated sigma to
zero. All branches share the exact clean LR endpoint, clean HR transition and
coordinate-noise tensor. Branches with the same sigma also share an identical
re-noised starting tensor.

## Server commands

```bash
cd /mnt/afs_2/houze/wanUpsampler

bash UNIV_adaptor/scripts/run_univ_mrflow_refinement_ablation.sh plan
GPU_ID=0 bash UNIV_adaptor/scripts/run_univ_mrflow_refinement_ablation.sh check
GPU_ID=0 bash UNIV_adaptor/scripts/run_univ_mrflow_refinement_ablation.sh run
```

The default output is `outputs/univ_mrflow_refinement_v1/`. Use a fresh
`OUT_DIR` for a different prompt, seed or matrix. The matrix can be overridden:

```bash
SIGMAS=0.10,0.15,0.30 HR_STEPS=1,2,3,5 \
OUT_DIR=/mnt/afs_2/houze/wanUpsampler/outputs/univ_mrflow_custom_v1 \
GPU_ID=0 bash UNIV_adaptor/scripts/run_univ_mrflow_refinement_ablation.sh run
```

The run writes `comparison_plan.json`, `resolved_config.json`,
`shared_clean_transition.pt`, one video and runtime sidecar per case, and
`comparison_summary.json`. Existing artifacts are never overwritten.

## Evaluation

```bash
GPU_ID=0 bash UNIV_adaptor/scripts/run_univ_mrflow_refinement_eval.sh check
GPU_ID=0 bash UNIV_adaptor/scripts/run_univ_mrflow_refinement_eval.sh run
```

Set `OUT_DIR`, `VBENCH_ROOT`, `VBENCH_PYTHON` and `VBENCH_COMMIT` when their
defaults do not match the server. Reports are written under
`<OUT_DIR>/metrics/mrflow_refinement/`.

The report gives raw VBench-5 dimensions, Dynamic Degree, deltas against the
transition-only control, HR time, and per-candidate denoising time. The latter
adds the shared LR50 and transition costs back to every candidate. This is a
single-prompt, single-seed ablation and is not an official VBench result.

## Local validation

```bash
python -m unittest \
  UNIV_adaptor.tests.test_mrflow_refinement \
  UNIV_adaptor.tests.test_mrflow_refinement_eval
```

Real video generation still requires the Linux/CUDA server environment.
