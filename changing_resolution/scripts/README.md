# changing_resolution scripts

Scripts are grouped by responsibility. Long remote jobs should use the tmux
entrypoints.

## data

Build the Stage 2 training data:

```bash
bash changing_resolution/scripts/data/tmux_build_clean_lmdb_480p720p_1k_multigpu.sh
```

Build the 50-step tail-skip LoRA LMDB on four GPUs:

```bash
TRAIN_STEP=45 TOTAL_SAMPLES=1000 GPU_IDS=0,1,2,3 \
  bash changing_resolution/scripts/data/build_tail_skip_lora_lmdb_multigpu.sh
```

Resume an interrupted tail-skip LoRA LMDB build without discarding completed
part samples:

```bash
TRAIN_STEP=45 TOTAL_SAMPLES=1000 GPU_IDS=0,1,2,3 RESUME=1 \
  bash changing_resolution/scripts/data/build_tail_skip_lora_lmdb_multigpu.sh
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

Evaluate whether the 50-step step-45 LoRA transfers to the 360p-class
`368x640` resolution. `360x624` produces an odd latent height and is not valid
for this Wan patch path. The output
contains `ori_45`, `lora_45`, and an `ori_50` reference; it also writes a
three-column visual comparison and a metric summary against `ori_50`:

```bash
LORA_CKPT=/path/to/latest.safetensors \
bash changing_resolution/scripts/eval/run_tail_skip_lora_360p_clean_pred_compare.sh run
```

Run a configuration/path check only:

```bash
LORA_CKPT=/path/to/latest.safetensors \
bash changing_resolution/scripts/eval/run_tail_skip_lora_360p_clean_pred_compare.sh check
```

Run the 480p ten-prompt, three-column Stage2 comparison. The column order is
`LoRA@45 + Stage2`, `x_pred@45 + Stage2`, and `teacher@50 + Stage2`:

```bash
LORA_CKPT=/path/to/latest.safetensors \
bash changing_resolution/scripts/eval/run_tail_skip_lora_stage2_480p_three_way_compare.sh run
```

Run the Stage 3 x0-pred sweep on four GPUs:

```bash
bash changing_resolution/scripts/eval/tmux_run_x0pred_480p720p_stage3_change_step_sweep_multigpu.sh
```

Run a 10-prompt, six-column comparison: interp baseline, Stage 2 clean 10k
checkpoint, z_predhr chain, and Stage 3 models trained for change steps 45,
46, and 47:

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

Run one 50-step tail-skip LoRA + Stage2 full-chain sample from `480x832` to
`720x1248`:

```bash
LORA_CKPT=/path/to/latest.safetensors \
STAGE2_CHECKPOINT=/path/to/latest.pt \
PROMPT="A cinematic shot of a sailboat at sunset." \
bash changing_resolution/scripts/bridge/run_tail_skip_lora_stage2_480p720p.sh run
```

Validate paths and emit the resolved LightX2V JSON without starting inference:

```bash
LORA_CKPT=/path/to/latest.safetensors \
STAGE2_CHECKPOINT=/path/to/latest.pt \
bash changing_resolution/scripts/bridge/run_tail_skip_lora_stage2_480p720p.sh check
```

Retired Stage 1 and V1 scripts are under `.archive/`.
