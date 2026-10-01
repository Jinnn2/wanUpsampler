# 已发表视频加速方法：官方源码清单（2026-10-01）

本目录的目标是复现**原作者实现**，再在统一的 prompt、seed、模型、质量评测协议下检验加速损伤。机器可读版本锁见 `configs/published_accelerators_v1.json`。外部仓库检出到 `UNIV_adaptor/external/`，该目录被本项目忽略；本项目只提交获取脚本和版本记录，不复制上游源码、模型权重或生成视频。

## 源码状态

| 方法 | 机制 | 官方代码 | 当前锁定版本 | 已确认入口 | 额外条件 |
|---|---|---|---|---|---|
| [Wan2.1 基线](https://github.com/Wan-Video/Wan2.1) | 原生完整去噪 | 可获取 | `9737cba9c1c3` | `generate.py`、`wan/text2video.py` | 与方法仓库独立锁定；首轮使用已有 1.3B 原生权重。 |
| [TeaCache](https://github.com/ali-vilab/TeaCache) | 跨去噪步复用输出 | 可获取 | `7c10efc4702c` | `TeaCache4HunyuanVideo/teacache_sample_video.py`、`TeaCache4Wan2.1/teacache_generate.py` | 上游要求先部署对应基模型，再将入口脚本放进其源码目录；阈值决定质量和速度。 |
| [ScalingCache](https://github.com/KlingAIResearch/ScalingCache) | 动态缓存间隔与差异缩放 | 可获取 | `7834c41c5457` | `HunyuanVideo/vbench_generate_t2v.py`、`Wan2.1/scalingcache_generate.py` | 上游要求复制其修改过的 `hyvideo/` 或 `wan/` 到基模型代码；提供预计算缩放系数。Wan2.1 1.3B 配置存在于源码，但官方 README 的运行示例为 14B，需先做 1.3B smoke test。 |
| [VGDFR](https://github.com/thu-nics/VGDFR) | 动态 latent 帧率，输出阶段插帧 | 可获取 | `0f52b050312f` | `VGDFR/hunyuan_vgdfr.py`、`experiments/example.ipynb` | 原版 HunyuanVideo；另需论文仓库 release 中的 `flownet.pkl`、RIFE 和特定 CUDA/PyTorch 依赖。 |
| [Jenga](https://github.com/JIA-Lab-research/Jenga) | 稀疏注意力、渐进分辨率和步数配置 | 可获取 | `69051503a9e5` | `jenga_hyvideo.py`、`jenga_wan.py` 与 `scripts/` | 官方 Wan 1.3B Base 示例同时设置 `--teacache_thresh 0.15`；Hunyuan Base 示例设置 `--step-rate-list 0.5 1.0`。复现完整 pipeline 时应记录这些组件，不能把结果归因于单一空间操作。 |
| [PAB / VideoSys](https://github.com/NUS-HPC-AI-Lab/VideoSys) | 空间、时间和跨模态注意力广播复用 | 可获取 | `4cce477878b0` | `videosys/core/pab/`、`docs/pab.md` | 官方文档中的 PAB 支持 Open-Sora、Open-Sora-Plan、Latte；暂不作为原版 HunyuanVideo 同模型对照。 |
| [DVG](https://arxiv.org/abs/2605.21042) | 早期 latent 预览驱动的时空联合压缩 | 未找到可直接拉取的官方 GitHub 仓库 | — | — | 论文称代码在补充材料中、后续发布 GitHub。需要获得可运行官方实现后才能作为“官方 DVG”复现。 |

`Jenga` 与 `VideoSys` 使用稀疏检出，只拉代码、文档和根目录文件；上游演示媒体未纳入本地源码快照。这不会改变锁定 commit。所有源码下载不等于已经安装依赖或下载模型权重。上游性能数字也不能跨不同模型、分辨率或实现协议直接比较。

## 获取与校验

在本仓库根目录运行，Windows 和 Linux 均可：

```bash
python UNIV_adaptor/scripts/data/fetch_published_accelerators.py --list
python UNIV_adaptor/scripts/data/fetch_published_accelerators.py --fetch all
python UNIV_adaptor/scripts/data/fetch_published_accelerators.py --check
```

只取一个仓库，例如：

```bash
python UNIV_adaptor/scripts/data/fetch_published_accelerators.py --fetch teacache
```

脚本校验 origin、commit、入口文件和工作区清洁度；已存在的文件不会被更新、清理或覆盖。若锁定版本的干净稀疏检出遗漏入口文件，再次运行 `--fetch all` 只会补出这些缺失文件；下载中断时保留目录和错误信息。远端服务器只需同步本仓库并运行相同命令；`UNIV_adaptor/external/` 不随 Git 同步。

## 第一轮复现边界

原版 HunyuanVideo 同时被 TeaCache、ScalingCache、VGDFR 和 Jenga 官方代码覆盖，是机制比较最完整的共同模型。当前已有的 HunyuanVideo-1.5 数据不能直接充当原版 HunyuanVideo 的无加速基线。若先用 Wan2.1 1.3B 做较快的代码 smoke test，可覆盖 TeaCache、ScalingCache 和 Jenga；VGDFR 官方代码不覆盖它。DVG 论文评测了原版 HunyuanVideo 与 HunyuanVideo-1.5，但目前未纳入可运行清单。

已落地的首轮 Wan2.1 脚本及校准/人评流程见 [PUBLISHED_WAN21_PILOT.md](PUBLISHED_WAN21_PILOT.md)。其 120 视频 pilot 复用现有模型权重，默认先进行关闭加速的共享基线校准；并非已完成 GPU 复现。

每一个最终视频记录至少应写明：上游仓库 commit、基模型权重 hash、入口/参数、prompt、seed、帧数、尺寸、采样步数、实际耗时、输出 hash。论文中的 VBench Total、我们此前使用的 VBench5 均值、以及自定义 prompt 可计算的维度必须分别标注。完整 VBench 复现要使用其标准 prompt 和官方计算协议；自定义 prompt 不能直接把有限维度的均值称为官方 Total。
