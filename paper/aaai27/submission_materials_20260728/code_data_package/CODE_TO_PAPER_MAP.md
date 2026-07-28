# Code-to-paper map

This file is the authoritative map from the supplied implementation to the
paper. Core files also carry a short `Paper map` comment.

| Paper location | Implemented operation | Supplied source |
|---|---|---|
| Sec. 3.1, Challenges of In-Trajectory Transition | scheduler-consistent clean-state reconstruction and transition bookkeeping | `code/changing_resolution/lightx2v_clean_bridge.py`, `code/changing_resolution_distill/lightx2v_distill_bridge.py` |
| Sec. 3.2, In-Trajectory Upsampler (ITU) | 3D projections, spatiotemporal residual blocks, rational/2x spatial lifting, crop, and forward pass | `code/wan_sr/models/stage2_resizer.py` |
| Sec. 3.2, ITU objective | reconstruction, low-frequency, and temporal-difference losses | `code/wan_sr/losses/latent_losses.py` |
| Sec. 3.2, ITU data construction | RGB resize followed by independent frozen-VAE encoding | `code/changing_resolution/scripts/data/build_480p720p_lmdb.py`, `code/changing_resolution_distill/scripts/data/build_clean_368x640_720x1248_distill_lmdb.py` |
| Sec. 3.2, ITU optimization | deterministic split, DDP, AdamW, EMA, validation, and checkpointing | `code/changing_resolution/scripts/train/train_clean_latent_resizer_stage2.py` |
| Sec. 3.3, Trajectory-Tail Distillation (TTD) | transition-local LoRA activation and scaling | `code/changing_resolution/dynamic_lora.py` |
| Sec. 3.3, TTD paired supervision | cached pre-transition state and frozen-teacher endpoint construction | `code/changing_resolution/scripts/data/build_tail_skip_lora_lmdb.py`, `code/changing_resolution_distill/scripts/data/build_last_step_skip_lora_lmdb.py` |
| Sec. 3.3, TTD objective and optimization | frozen base denoiser, rank-32 LoRA, L1/MSE/temporal loss, deterministic split | `code/changing_resolution/scripts/train/train_tail_skip_lora.py`, `code/changing_resolution_distill/scripts/train/train_last_step_skip_lora.py` |
| Sec. 4, Experiments | final Wan50 and Distill4 generation suites | `code/paper/aaai27/experiments/run_final_quality_efficiency.py`, `code/paper/aaai27/experiments/run_distill4_quality_efficiency.py` |
| Sec. 4, evaluation/statistics | VBench aggregation, paired bootstrap intervals, and sign tests | `code/paper/aaai27/experiments/compile_vbench_paired_statistics.py`, `code/paper/aaai27/experiments/paired_statistics.py` |

External LightX2V, DiffSynth-Studio, VBench, and public model code is not
vendored. Their revisions are dependencies recorded by
`reproduction_assets.json` and the remote environment exporter.
