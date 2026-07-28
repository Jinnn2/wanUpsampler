# InTraScale 完整端到端复现清单

## 复现目标

建议把“完整复现”分成两个层级，避免把推理复现与重新训练混在一起：

1. **R1：论文结果端到端复现**  
   从固定 prompt/seed 开始，加载公开基础模型和最终 ITU/TTD 权重，生成视频，
   再运行 VBench 与配对统计。投稿材料首先需要闭环这一层。
2. **R2：从头训练复现**  
   在 R1 之外，还需要原始训练视频或其许可、确定的数据划分、完整 LMDB、
   优化器/EMA 状态和训练日志。该层通常不适合直接放入投稿 ZIP，但应建立内部归档。

## R1 必需资产

### 1. 公开基础模型

下载脚本已固定 Hugging Face revision：

- Wan50：`Wan-AI/Wan2.1-T2V-1.3B`
  revision `37ec512624d61f7aa208f7ea8140a131f93afc9a`
- Distill4：`lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill`
  revision `b9baaa9a0c29226dea39043db647d1ced950bbea`

两套模型关键文件合计约 58 GB；考虑配置、小文件和下载临时空间，脚本要求至少
65 GiB 可用空间。执行：

```bash
python tools/download_public_assets.py \
  --output-root /path/to/public_models

python tools/download_public_assets.py \
  --output-root /path/to/public_models \
  --execute
```

第一条只输出下载计划；第二条才实际下载。下载后会核验关键文件大小和 SHA-256。

### 2. 五个自研 checkpoint

| 逻辑名称 | 大小 | SHA-256 |
|---|---:|---|
| Wan50 ITU | 570,100,161 B | `f6525b753931698015b3bd04a5dd314efaa1cb04858a71f9fa7bef0babaa7d54` |
| Wan50 TTD step 40 | 87,559,648 B | `bfbcc4998be1fa166e3438b91d9b0bc40aced21a71ba2d15c15079cafd9358e8` |
| Wan50 TTD step 45 | 87,559,616 B | `73f9d38287386dd547aff84ad7a1c7c74d9640033dfe82b94b706c4e7a9bb79f` |
| Distill4 ITU | 570,100,417 B | `96c9980834823088506f0033d8423f69a66bc6ecfe1608e5c22a2319776e859e` |
| Distill4 TTD step 3 | 306,801,824 B | `439851cf5c5ae48f85663cb6f8e3b705ee139f289a66e7cfe3a8a8572702d06c` |

这些值来自实际运行 manifest，不是根据文件名猜测。必须在原实验机导出，因为自研
权重没有公开下载地址。执行：

```bash
bash paper/aaai27/submission_materials_20260726/code_data_package/tools/export_full_repro_bundle.sh \
  --project-root "$PROJECT_ROOT" \
  --output "$EXPORT_ROOT/intrascale_full_repro_$(date +%Y%m%d_%H%M%S)"
```

脚本会在复制前后强制核验大小和 SHA-256；任意不一致都会停止。

### 3. 精确代码与环境

同一个导出脚本会收集：

- wanUpsampler、LightX2V、DiffSynth、VBench、Real-ESRGAN 的 commit、dirty diff、
  submodule 与 remote 信息；
- 当前方法、训练、推理和评测源码快照；
- GPU 型号/UUID/显存、驱动、CUDA、cuDNN、CPU、内存、OS；
- Python、PyTorch、关键模块位置及版本；
- `pip freeze --all`、conda 完整环境和 explicit package list；
- ffmpeg、gcc、nvcc 版本；
- 五个自研 checkpoint 和整个导出目录的 SHA-256 清单。

运行前应设置真实外部仓库路径：

```bash
export LIGHTX2V_REPO=/path/to/LightX2V
export DIFFSYNTH_REPO=/path/to/DiffSynth-Studio
export VBENCH_REPO=/path/to/VBench
export REALESRGAN_REPO=/path/to/Real-ESRGAN
export MODEL_ROOT=/path/to/Wan2.1-T2V-1.3B
export DISTILL_MODEL_ROOT=/path/to/Wan2.1-T2V-14B-StepDistill-CfgDistill
```

### 4. 输入、协议和评测

当前代码数据包已包含：

- Wan50 与 Distill4 测试 prompt；
- 9700–9709、9800–9809 和验证集 16000 起始 seed 协议；
- scheduler、分辨率、81 帧和 guidance 设置；
- 最终 ITU/TTD 配置与推理入口；
- VBench-5 原始导出、配对统计和 warm-latency 表；
- 368p→720p ITU 的 50 条逐样本原始记录与重新计算结果。

VBench 应继续使用独立 CUDA 12.1 环境，不要为了评测降级生成环境。

## R2 重新训练还需补齐

- Wan50 ITU clean-latent LMDB 与确定的 train/validation split；
- Distill4 ITU clean-latent LMDB；
- Wan50 step40/45 TTD 缓存状态或可确定重建的数据生成命令；
- Distill4 step3 TTD 缓存状态；
- 生成上述 LMDB 的原始视频、prompt revision、许可和 SHA 清单；
- 最终训练日志、验证轨迹、optimizer/scaler/EMA 状态；
- 4×H100 的训练拓扑、每任务 wall-clock、峰值显存和失败重启记录。

如果原始视频不便发布，至少应保留内部不可变归档，并公开确定的数据构建脚本、输入
清单、split 和 SHA；否则只能声称 R1 推理复现完整，不能声称从头训练完全可复现。

## 当前状态

- 368p 原始证据：**已找回并校验完成**。
- 公开基础模型：**已有固定 revision 和下载/校验脚本，尚未在本机下载**。
- 自研 checkpoint：**已知五个精确路径、大小和 SHA；仍需从原实验机重新导出**。
- 历史环境归档：已恢复 Python 3.11.13、Linux 5.14/glibc 2.35、两个阶段的
  wanUpsampler commit，以及作者记录的 4×H100、ITU 约 8 小时、TTD 约
  33 小时；驱动、CUDA、PyTorch 和外部仓库 revision 仍缺。
- 旧下载目录中的两个 Distill 权重只有约 1.5 MiB，与应有大小和 SHA 不符，
  判定为截断文件，禁止用于复现。
- 精确实验机环境：**等待运行完整导出脚本**。
