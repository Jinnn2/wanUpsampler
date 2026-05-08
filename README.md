# wanUpsampler

`wanUpsampler` 是一个面向 Wan / LightX2V 采样链路的 latent 分辨率切换项目。

当前主线是 `changing_resolution`：训练并接入一个 clean latent resizer，在 LightX2V 的 changing_resolution 过程中替换固定插值算子，用于 480p -> 720p。

## 当前主线

优先阅读：

```text
changing_resolution/README.md
changing_resolution/TRAINING_PLAN.md
```

推荐远端运行入口：

```bash
# 1. 构建 1k clean-latent LMDB
bash changing_resolution/scripts/data/tmux_build_clean_lmdb_480p720p_1k_multigpu.sh

# 2. 训练 stage1 resizer
bash changing_resolution/scripts/train/tmux_run_clean_480p720p_stage1_lmdb_training.sh

# 3. operator compare: 有 ori720_decode 参考，输出 PSNR / SSIM / LPIPS
bash changing_resolution/scripts/eval/tmux_run_clean_480p720p_operator_compare_multigpu.sh

# 4. generation-chain A/B: 同一 LightX2V 链路内比较 interp720 / trained720
bash changing_resolution/scripts/eval/tmux_run_clean_480p720p_chain_ab_compare_multigpu.sh
```

## 项目结构

```text
wan_sr/
  核心 Python 包：数据集、模型、loss、scheduler、训练工具、Wan VAE wrapper。

changing_resolution/
  当前 V2 主线。目标是 clean latent 480p -> 720p，并接入 LightX2V changing_resolution。

scripts/v1/
  早期 V1 noisy-to-clean upsampler 流程。保留用于回溯和兼容。

configs/
  机器路径配置和 V1 配置。V2 配置放在 changing_resolution/configs/。

experiments/
  外部参考或早期实验代码，不作为当前主线入口。

PROGRESS.md
  理论进度和当前阶段判断。
```

## 路径配置

机器相关路径集中在：

```text
configs/local_paths.sh
```

默认面向远端环境：

```text
PROJECT_ROOT=/mnt/afs_2/houze/wanUpsampler
LIGHTX2V_REPO=/mnt/afs_2/houze/LightX2V
MODEL_ROOT=/mnt/afs_2/houze/Wan-AI/Wan2.1-T2V-1.3B
```

临时切换路径时使用环境变量覆盖：

```bash
PATH_CONFIG=/path/to/local_paths.sh bash changing_resolution/scripts/train/tmux_run_clean_480p720p_stage1_lmdb_training.sh
```

## V1 历史入口

V1 是早期 noisy latent upsampler 路线，脚本已归档到 `scripts/v1/`。

```bash
bash scripts/v1/train/run_lightx2v_training.sh build
bash scripts/v1/train/run_lightx2v_training.sh train
```

V1 配置位于：

```text
configs/v1/
```

## 安装依赖

```bash
pip install -r requirements.txt
```

完整训练和推理需要在 Linux GPU 机器上运行，并保证 LightX2V、Wan2.1 模型权重、Wan VAE 权重路径可用。
