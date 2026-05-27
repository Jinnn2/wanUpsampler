# changing_resolution scripts

Scripts are grouped by responsibility. Long remote jobs should use the tmux
entrypoints.

## data

Build the Stage 2 training data:

```bash
bash changing_resolution/scripts/data/tmux_build_clean_lmdb_480p720p_1k_multigpu.sh
```

## train

Train the Stage 2 clean-latent 480p -> 720p resizer:

```bash
bash changing_resolution/scripts/train/tmux_run_clean_480p720p_stage2_lmdb_training.sh
```

## eval

Run Stage 2 operator compare:

```bash
bash changing_resolution/scripts/eval/tmux_run_clean_480p720p_stage2_operator_compare_multigpu.sh
```

Run Stage 2 generation-chain A/B:

```bash
bash changing_resolution/scripts/eval/tmux_run_clean_480p720p_stage2_chain_ab_compare_multigpu.sh
```

Sweep the changing-resolution handoff step and produce three-way panels:

```bash
bash changing_resolution/scripts/eval/run_clean_480p720p_stage2_change_step_sweep.sh
```

Run the step sweep on four GPUs:

```bash
bash changing_resolution/scripts/eval/tmux_run_clean_480p720p_stage2_change_step_sweep_multigpu.sh
```

Run a small Stage 3 x0-pred sweep test:

```bash
bash changing_resolution/scripts/eval/run_x0pred_480p720p_stage3_change_step_sweep.sh
```

Run the Stage 3 x0-pred sweep on four GPUs:

```bash
bash changing_resolution/scripts/eval/tmux_run_x0pred_480p720p_stage3_change_step_sweep_multigpu.sh
```

Run a 10-prompt, four-column comparison: interp baseline plus Stage 3 models
trained for change steps 45, 46, and 47:

```bash
bash changing_resolution/scripts/eval/run_x0pred_480p720p_stage3_three_model_compare.sh
```

Compare wall-clock generation time between direct Wan 720p generation and the
Stage 3 bridge path:

```bash
bash changing_resolution/scripts/eval/benchmark_generation_time.sh
```

To also time the one-step `x0_pred` data-build call, include it explicitly:

```bash
BENCH_CASES=direct_720p,stage3_bridge_720p,x0pred_call \
  bash changing_resolution/scripts/eval/benchmark_generation_time.sh
```

Operator compare output can be converted to CSV and Markdown tables:

```bash
python changing_resolution/scripts/eval/summarize_operator_compare_table.py \
  --input outputs/changing_resolution_operator_compare_stage2 \
  --split val
```

## bridge

`scripts/bridge/run_lightx2v_clean_bridge_infer.py` registers the Stage 2
clean-resizer bridge for local LightX2V inference.

Retired Stage 1 and V1 scripts are under `.archive/`.
