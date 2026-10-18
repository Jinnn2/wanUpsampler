# Prompt-method platform v1（Wan2.1-1.3B，832×480×81）

## 目的

论文按三步论证：

1. 平均分不足以评价加速方法。
2. 不同加速方法有各自擅长的 prompt 区间。
3. 用我们的评价体系得到 prompt→方法的最优路由，构成 prompt 指导的视频加速框架。

本平台服务于第 2 步：所有方法共用同一套 prompt、seed 和采样设置，在相近的加速档位上比较各自的内容条件表现。

## 为什么需要新平台（2026-10-18 本地分析）

在 controlled-factor（720p，LightX2V）数据上做了按 4 个 motion×detail 格的类别路由，规则在训练 family 或训练 seed 上选，在留出部分上评估：

- **spatial 与 temporal（耗时相同，均为 FULL 的 0.37）**：路由相对最佳固定动作，Δvbench5 收益为 +0.12pp，CI [−0.12, +0.35]，需求清单收益也不显著。temporal 在多数格上都占优，两者缺少互补。
- **再加入 cache**：Δvbench5 收益为 +0.40pp，CI [+0.04, +0.77]。但 cache 的耗时是 FULL 的 0.51，部分收益来自多花的时间，不是同速比较。

结论：要让路由有价值，候选池里必须有在相近速度下表现互补的方法。上述数据只有 3 个臂、速度不齐，而且 spatial 臂并不是真正的信息瓶颈。

## 设计

配置文件为 `configs/published_wan21_platform_v1.json`，沿用已冻结的 published pilot 驱动与 worker，不改动上游代码。

- **Prompt 与 seed**：controlled-factor 的 64 个 dev prompt（16 个 family，4 个格），locked test 不进入。seed 为 42、100、2024，与 DVG 实验（`outputs/dvg_wan21_study_v1`）完全相同。
- **方法臂**：FULL50、STEP25、STEP20、TEA008、TEA015、SCALING10，共 6 臂 × 192 组 = 1152 个视频，另有 16 个校准视频。
- **TEA015**：阈值 0.15 来自 Jenga 官方 1.3B 脚本中的 TeaCache 设置，不是 TeaCache README 的预设，作为快档。
- **DVG**：DVG_D070、DVG_D050 已在 DVG 实验中生成，不重复跑。
- **预期速度分档**（以实测为准）：
  - 约 1.5–1.8×：DVG_D070、STEP25、TEA008、SCALING10。
  - 约 2.2×：DVG_D050、STEP20、TEA015。

两套代码路径（官方 Wan 和 DVG 内置的 Wan）的 FULL50 应当一致。首个分析步骤就是逐像素核对这一点：

- 若一致，所有加速臂共用同一个参考。
- 若不一致，各方法分别对照自己代码路径的 FULL，并在报告中说明。

## 运行（4 卡）

```bash
export PUBLISHED_WAN21_ROOT=/mnt/afs_2/houze/wanUpsampler/outputs/published_wan21_platform_v1
export PILOT_CONFIG=/mnt/afs_2/houze/wanUpsampler/UNIV_adaptor/configs/published_wan21_platform_v1.json
export NGPUS=4
S=UNIV_adaptor/scripts/run_univ_published_wan21_8gpu.sh
bash $S check
bash $S calibrate
bash $S audit          # 校准通过后再生成
bash $S generate
bash $S status
bash $S finalize
bash $S score
bash $S export
```

预计 1152 个视频约需 41 GPU 小时，4 卡约 10 小时，以校准实测为准。

## 需求清单与光流评分（VBench 打分之后）

和 DVG 实验使用同一套指标：Qwen2.5-VL 需求清单、RAFT 光流、细节统计。

- **group id**：映射为 DVG 的格式 `main_p###_s<seed>`，两个来源可以按 prompt 和 seed 直接合并。
- **FULL 编号**：FULL50 记为 `FULL_NATIVE_50`。

```bash
export PUBLISHED_ROOT=$PUBLISHED_WAN21_ROOT
export REQ_ROOT=/mnt/afs_2/houze/wanUpsampler/outputs/platform_v1_requirements
export GPU_IDS=0,1,2,3,4,5,6,7
R=UNIV_adaptor/scripts/run_univ_controlled_factor_requirements.sh
bash $R plan && bash $R signals && bash $R vlm && bash $R report
```
