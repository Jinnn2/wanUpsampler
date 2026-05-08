# scripts

当前主线脚本不放在这里，而是在：

```text
changing_resolution/scripts/
```

本目录保留 V1 历史流程：

```text
scripts/v1/data/
  build_latent_pairs.py
  download_davis2017.sh

scripts/v1/train/
  train.py
  run_lightx2v_training.sh

scripts/v1/eval/
  eval_latent.py
  eval_decode.py

scripts/v1/infer/
  infer_transition_wan.py
  apply_wan_upsampler_to_video.py
  run_lightx2v_wanupsampler_compare.py

scripts/v1/lightx2v/
  run_wan_t2v_wanupsampler_v1.sh
  run_wan_t2v_wanupsampler_v1_batch20.sh
```

V1 路线用于回溯早期 noisy-to-clean upsampler，不再作为 480p -> 720p changing_resolution 主入口。
