# 项目进度

最后更新：2026-04-29

## 当前结论

项目代码骨架已经完成，核心训练闭环已具备：

- latent pair 构造
- sigma/noise 采样
- upsampler 模型
- loss
- train / eval / transition
- LightX2V Wan VAE 接入
- DAVIS 下载与转换脚本
- 本机路径集中配置

当前状态不是“从零开始”，而是**已经进入远程机联调和首轮数据构造验证阶段**。

## 已完成

### 代码结构

- `wan_sr/models/`：3D CNN upsampler、sigma embedding、ResBlock、PixelShuffle
- `wan_sr/data/`：latent pair dataset、video io、degradation
- `wan_sr/losses/`：latent / low-freq / temporal loss
- `wan_sr/schedulers/`：sigma sampler、flow-style add noise
- `wan_sr/pipelines/transition.py`：LR noisy latent -> HR noisy latent
- `wan_sr/vae/wan_vae_wrapper.py`：official / lightx2v / diffusers 三类 VAE backend

### 脚本

- `scripts/build_latent_pairs.py`
- `scripts/train.py`
- `scripts/eval_latent.py`
- `scripts/eval_decode.py`
- `scripts/infer_transition_wan.py`
- `scripts/run_lightx2v_training.sh`
- `scripts/download_davis2017.sh`

### 路径配置

已新增根路径配置：

```text
configs/local_paths.sh
```

当前默认基于：

```text
/data/yongyang/Jin
```

并集中管理：

- `PROJECT_ROOT=/data/yongyang/Jin/wanUpsampler`
- `WAN_REPO=/data/yongyang/Jin/Wan2.1`
- `LIGHTX2V_REPO=/data/yongyang/Jin/LightX2V`
- `MODEL_ROOT=/data/yongyang/Jin/Wan-AI/Wan2.1-T2V-1.3B`
- `VAE_PATH=/data/yongyang/Jin/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth`
- `DAVIS_ZIP=/data/yongyang/Jin/wanUpsampler/datasets/DAVIS-2017-trainval-480p.zip`
- `RAW_VIDEO_DIR=/data/yongyang/Jin/wanUpsampler/data/raw_videos`
- `LATENT_DIR=/data/yongyang/Jin/wanUpsampler/data/latent_pairs_wan21_512`
- `OUT_DIR=/data/yongyang/Jin/wanUpsampler/outputs/wan_traj_upsampler_x2`

## 已完成的联调修复

### 导入与路径

- 修复 `.gitignore` 误忽略 `wan_sr/data/`
- 修复 `LightX2V` VAE 非 `nn.Module` 包装导致的 `.eval()` 报错
- 支持从 `configs/local_paths.sh` 统一读取路径

### VAE backend

- 避免把 Wan 主模型根目录误当成 diffusers VAE 加载
- 支持：
  - `official`
  - `lightx2v`
  - `diffusers`

### 数据构造健壮性

- `build_latent_pairs.py` 先检查视频文件，再加载 VAE
- `run_lightx2v_training.sh` 在 build 前检查 raw videos，在 train 前检查 latent 数据
- `download_davis2017.sh` 自动探测 DAVIS 解压出的 `JPEGImages/480p`
- 对损坏 mp4 做 `ffprobe` 检测；坏文件会删掉并重转
- `build_latent_pairs.py` 可跳过坏视频继续处理

## 当前待验证项

以下项**代码已修**，但需要在远程机上再次确认结果：

1. DAVIS zip 解压目录自动探测是否稳定。
2. DAVIS 转 mp4 后是否全部可读。
3. `bash scripts/run_lightx2v_training.sh build` 是否能够完整跑完。
4. `data/latent_pairs_wan21_512/` 是否成功产出样本。
5. `bash scripts/run_lightx2v_training.sh train` 是否能启动第一轮训练。

## 最近一次已知阻塞

最近一次远程机阻塞不是模型本身，而是数据侧：

1. 一开始脚本找不到 raw videos。
2. 之后确认 DAVIS mp4 已生成。
3. 再之后发现 `cat-girl.mp4` 是损坏文件，ffmpeg 报：

```text
moov atom not found
```

对应修复已经提交：下载脚本会校验并重建坏 mp4，build 默认会跳过坏视频继续处理。

## 下一步

按优先级：

1. 在远程机重新执行：
   ```bash
   bash scripts/download_davis2017.sh
   bash scripts/run_lightx2v_training.sh build
   ```
2. 确认 `LATENT_DIR` 下已生成样本目录。
3. 启动：
   ```bash
   bash scripts/run_lightx2v_training.sh train
   ```
4. 记录第一轮训练：
   - loss 是否下降
   - 首个 checkpoint 是否生成
   - 是否出现显存 / shape / decode 问题

## 维护规则

后续统一维护这个文件，按下面方式追加或更新：

- 更新“最后更新”日期
- 修改“当前结论”
- 在“已完成的联调修复”里补充新修复
- 在“当前待验证项”里删除已验证项
- 在“最近一次已知阻塞”里写最新 blocker
- 在“下一步”里保持 3 到 5 条最关键动作
