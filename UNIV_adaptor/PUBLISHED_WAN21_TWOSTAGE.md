# Prompt-method platform v2a：只压空间、只压时间的两阶段方法

## 目的

v1 缺少只压空间、只压时间两类方法。本平台补上这两类，回答"每种机制在哪类 prompt 上失效"。prompt、seed、采样设置都与 v1 相同（64 个 dev prompt × 3 个 seed，Wan2.1-1.3B，50 步 UniPC，shift 8，CFG 6）。locked test 不进入。

## 方法臂

| 臂 | 第一阶段（官方 Wan generate.py） | 第二阶段 | 作用 |
|---|---|---|---|
| FULL50 | 832×480，81 帧 | 无 | 参考，在同一台机器上重跑，保证计时同硬件 |
| LR_FLASH | 416×240，81 帧 | FlashVSR v1.1 Tiny ×4，再 Lanczos 缩到 832×480 | 只压空间 |
| LR_BICUBIC | 与 LR_FLASH 同一份低分辨率帧 | 双三次 ×2 | 对照：区分"低分辨率生成本身变差"和"超分没修好" |
| F41_RIFE | 832×480，41 帧 | Practical-RIFE v4.26 插帧 ×2，得到 81 帧 | 只压时间 |
| F41_DUP | 与 F41_RIFE 同一份 41 帧 | 每帧重复（第 i 帧取第 i//2 帧） | 对照：区分"少生成帧"和"插帧质量" |

LR 和 HR41 的第一阶段各只生成一次，被两个臂共用。共 576 次生成和 768 次后处理，最终 960 个视频。

FlashVSR 沿用 `FLASHVSR_WAN21_COMPLETE.md` 已验证的做法：
- 作者参数：sparse 2.0、KV 3.0、local range 11、color fix。
- 边缘填充到 416×256，尾帧补到 89 帧，输出后去掉填充部分。

RIFE 直接调用官方 `model.inference`（timestep 0.5，scale 1.0，fp32）。官方命令行里有两条启发式规则：静止帧跳过和场景切换保持。这两条会改变帧数，所以没有使用。

## 预先写下的假设

- **LR_FLASH**：在高细节 prompt 上失效，比如小字、细纹理、小物体。在低细节、大块结构的场景上应接近 FULL。
- **F41_RIFE**：在高运动、快动作、有遮挡和多事件的 prompt 上失效；在静态和慢动作 prompt 上接近无损。

## 已知的混杂因素（分析时必须处理）

1. **2 倍慢动作。** Wan 不能指定帧率，生成 41 帧时画的是约 2.5 秒的内容。插值到 81 帧再按 16fps 播放，就成了 2 倍慢动作。影响如下：
   - 每帧光流天然减半。冻结判定（光流低于 FULL 的 0.5 倍）在比较前必须把光流乘 2，或者用原始 41 帧去和 FULL 每隔一帧取一帧比较。
   - VBench 的 motion_smoothness 会偏高，dynamic_degree 会偏低。
2. **训练分布外的分辨率。** 416×240 超出了 1.3B 模型的 480p 训练分布。LR_BICUBIC 用来区分问题出在生成还是超分。
3. **计时。** 两个阶段运行在不同的 Python 环境里，中间帧要经过一次无损 uint8 文件交接。
   - 文件读写和 MP4 编码不计时。
   - 计时包括第一阶段（文本编码、去噪、VAE 解码）和第二阶段的所有计算（含前后缩放）。
   - 加速比按同一组的 FULL50 耗时除以本臂的（第一阶段 + 第二阶段）耗时计算。
4. **编码。** 所有最终视频都用与 Wan `cache_video` 相同的 imageio/libx264 quality=8 编码。

## 新机器：安装（一次）

先在仓库根目录 `git pull`，然后执行：

```bash
cd /mnt/afs_2/houze/wanUpsampler
# 可选，但建议：先在 VBench 能正常运行的旧机器上导出模型缓存，避免新机器联网下载失败
bash UNIV_adaptor/scripts/install_twostage_wan21_env.sh cache-export
# 在新机器上执行
bash UNIV_adaptor/scripts/install_twostage_wan21_env.sh
```

安装脚本按顺序执行以下步骤：preflight → wan → vbench → flash → rife → verify。每一步先检查，只安装缺少的部分；失败时会停下，可以用 `install_twostage_wan21_env.sh <步骤名>` 单独重跑某一步。

- **preflight**：检查 GPU 数量、ffmpeg/ffprobe、基础 CUDA torch、Wan 权重，以及剩余磁盘空间（不少于 40 GB）。
- **wan**：如果共享的 `.venv_published_wan21` 能在本机运行，就直接复用；FA2 必须能在 GPU 上实际跑通。否则新建 `.venv_published_wan21_<GPU型号>`，不会覆盖旧环境。如果缺 flash-attn，加 `INSTALL_FLASH_ATTN=1` 重跑。
- **vbench**：
  - 以本机 torch、torchvision、numpy 的版本为约束，只补装缺少的包：clip、omegaconf、pyiqa、timm、decord 等。
  - 如果 `CACHE_SEED`（默认 `/mnt/afs_2/houze/cache_seed`）存在，从中复制模型缓存。
  - 把 DINO 仓库 clone 到 torch.hub 缓存，并下载 DINO 和 CLIP 权重。这样可以绕开上次遇到的 GitHub API 403 限流。
- **flash**：优先复用共享的 `.venv_flashvsr_diagnostic`，前提是 block-sparse 算子包含本机 GPU 的架构（sm80 或 sm90）。否则调用原 FlashVSR 诊断脚本的 setup 新建环境。新建需要满足：
  - 有 nvcc（CUDA ≥ 12.4）。
  - 有 Python 3.11；如果 base 版本不对，会用 conda 自动创建。
- **rife**：把 Practical-RIFE 固定到 `bbfd2ea`；从作者本人的 HF 仓库 `hzwer/RIFE` 下载 v4.26 压缩包，校验 sha256 后解压到 `checkpoints/rife_v426`。
- **verify**：
  - 先用一个已有的 FULL50 视频在单卡上跑 VBench 全部 7 个维度。也可以用 `SMOKE_VIDEO=...` 指定视频。
  - 再执行 `check`：冻结 plan，然后在 GPU 上逐个实跑全尺寸探针，包括 3 种生成尺寸、FlashVSR（检查稀疏注意力和颜色校正确实被调用）、RIFE 和两个对照。

能正常运行的解释器路径会写入 `outputs/twostage_env/<GPU型号>.env`，运行脚本会自动读取。

需要代理时，照常设置 `http_proxy`/`https_proxy`。Hugging Face 不通时，可以设 `HF_ENDPOINT` 走镜像。

## 运行

```bash
S=UNIV_adaptor/scripts/run_univ_twostage_wan21_8gpu.sh
bash $S all        # check → stage1 → post → finalize → score → export
bash $S status     # 另开终端查看进度
```

- 中途失败时，按打印出的日志路径查看原因，修复后重新执行 `bash $S all`。已有有效回执的任务会自动跳过。
- 没有回执的残留文件不会被覆盖，需要先手动挪走。

**预计耗时（以实测为准）**：H100 上 FULL50 每条约 1 分钟。第一阶段约 576 条生成，合约 5–6 GPU 小时；后处理不到 1 GPU 小时；8 卡加起来约 1 小时。VBench 打分约 1 小时。

**输出**：`outputs/published_wan21_twostage_v1/exports/twostage_wan21_analysis.tgz`。

## 需求清单与光流评分

完成 `finalize` 和 `score` 后，可以直接用 v1 的需求评分流程，输入格式兼容：

```bash
export PUBLISHED_ROOT=/mnt/afs_2/houze/wanUpsampler/outputs/published_wan21_twostage_v1
export REQ_ROOT=/mnt/afs_2/houze/wanUpsampler/outputs/platform_v2a_requirements
export GPU_IDS=0,1,2,3,4,5,6,7
R=UNIV_adaptor/scripts/run_univ_controlled_factor_requirements.sh
bash $R plan && bash $R signals && bash $R vlm && bash $R report
```

RAFT 权重在新机器上下载失败过。如果仍然下载不了，就在旧机器上执行这一步。

## 代码

- 配置：`configs/published_wan21_twostage_v1.json`
- 驱动：`scripts/data/twostage_wan21.py`
- worker：`scripts/data/twostage_wan21_worker.py`
- 测试：`tests/test_twostage_wan21.py`

v1 的驱动和 worker 只被导入用作工具函数，没有任何改动，v1 已冻结的 plan 仍然有效。plan 冻结了以下内容，其中任何一项改变都需要换一个新的输出目录：

- 5 个代码文件的哈希。
- Wan、FlashVSR、Block-Sparse-Attention、Practical-RIFE 的提交。
- FlashVSR 和 RIFE 权重的哈希。
