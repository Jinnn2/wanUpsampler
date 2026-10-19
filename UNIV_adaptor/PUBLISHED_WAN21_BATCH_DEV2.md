# 一次性实验批次：v2b 新方法 + dev2 新 prompt 集（8 卡，约 10 小时）

## 目的

- **v2b**：补上两类方法。
  - 蒸馏：官方 rCM，分别跑 4、2、1 步。
  - 空间加时间同时压缩：先生成 416×240×41，再用 FlashVSR 放大，最后用 RIFE 插帧。
- **dev2**：新写 64 个 prompt（16 个新 family，每个 family 覆盖 4 个格），所有方法在上面重跑。
  - 在 v1 dev prompt 上学到的路由规则，到 dev2 上评估。这是论证第 3 步需要的"推广到没见过的 prompt"检验。
  - locked test 仍然不碰。dev2 的 family 和全部 20 个旧 family（含 test）都不重叠。

## 预先写下的假设（数据出来之前）

1. **rCM**：在高运动 prompt 上失效，表现为运动变少、动作变慢、freeze 增多；静态 prompt 上接近 FULL。步数越少越明显，可能伴随过饱和（aesthetic 和 imaging 指标异常）。速度预计 10–20×。
2. **LR41_FLASH_RIFE**：两种损失叠加，在高运动且高细节的格子里最差；速度预计 6–8×。
3. **dev2 复现性**：v1 dev 上看到的格子级符号，在 dev2 上应保持同号：
   - 高细节格子里 STEP 优于 cache，低细节格子里 cache 优于 STEP。
   - 高运动且高细节的格子里 F41_RIFE 优于 LR_FLASH。
   - DUP 和 RIFE 在 VBench 上几乎一样（说明 VBench 看不出卡顿）。

## 运行

先在仓库根目录执行 `git pull`。v2a 的需求评分如果还在跑，等它跑完，或者 Ctrl+C 停掉，之后可以接着跑。

```bash
cd /mnt/afs_2/houze/wanUpsampler
B=UNIV_adaptor/scripts/run_univ_batch_dev2_8gpu.sh
bash $B all 2>&1 | tee outputs/batch_dev2.log     # 按顺序跑完 v2b → platform → twostage → dvg → requirements
bash $B status                                     # 在另一个终端查看进度
```

- 每个阶段都能断点续跑：失败或中断后，修好问题再执行一次 `bash $B all`，已有回执的任务会自动跳过。
- 也可以只跑某一个阶段：`bash $B v2b`、`bash $B platform`，依此类推。

| 阶段 | 内容 | 预计时间（8 卡） |
|---|---|---|
| v2b | 先下载 rCM 权重（2.8 GB）；LR41、RCM4、RCM2、RCM1 生成；FlashVSR 后接 RIFE；VBench 打分。FULL50 直接复用 v2a 的结果 | 约 40 分钟 |
| platform | v1 的 6 个臂（FULL50、STEP25、STEP20、TEA008、TEA015、SCALING10）在 dev2 上运行，用的是冻结的 v1 驱动，只换了配置 | 约 3 小时 |
| twostage | v2a 和 v2b 的全部 8 个臂在 dev2 上运行。FULL50 复用 platform 的结果，不重新生成 | 约 1.5 小时 |
| dvg | DVG 0.7、DVG 0.5，以及 DVG 自己的 FULL，在 dev2 上运行 | 约 1.5 小时 |
| requirements | 对以上 4 份结果计算 RAFT 光流、细节统计和 VLM 需求清单 | 约 2 小时；RAFT 下载失败时，改到旧机器上跑 |

**需要你看一眼的地方**：v2b 的 check 阶段会把 rCM 探针视频保存到 `outputs/published_wan21_twostage_v2b/probe_videos/`。日志里会打印 `LOOK AT ...`，请打开看看画面是否正常（一个蓝色马克杯在转台上转）。如果是噪声或花屏，立刻 Ctrl+C 并告诉我。

**中途请先发给我**：v2b 结束后会生成 `outputs/published_wan21_twostage_v2b/exports/twostage_wan21_analysis.tgz`，请先把它发给我。我会核对 rCM 的分数是否正常；如果有问题，可以在 twostage 阶段之前修好。

## 最后需要发给我的文件

- `outputs/published_wan21_twostage_v2b/exports/twostage_wan21_analysis.tgz`
- `outputs/published_wan21_platform_dev2/exports/published_wan21_analysis.tgz`
- `outputs/published_wan21_twostage_dev2/exports/twostage_wan21_analysis.tgz`
- `outputs/dvg_wan21_study_dev2_analysis.tgz`
- `outputs/*_requirements_analysis.tgz`：v2b、platform_dev2、twostage_dev2、dvg_dev2 各一个

## 实现说明

- **rCM**
  - 不安装 rcm 仓库（它依赖 transformer_engine 和 megatron）。官方的 rCM 1.3B 480p DiT（`worstcoder/rcm-Wan@142d77a`，sha256 `3baa20e8…`）直接加载进官方 Wan2.1 的 WanModel。
  - 两边模块命名一致。唯一的转换是：`patch_embedding` 在 rCM 里是 Linear，权重形状 (1536, 64)，reshape 成 Conv3d 的 (1536, 16, 1, 2, 2) 即可。
  - 采样循环逐行照抄官方 `wan2pt1_t2v_rcm_infer.py`：TrigFlow 时间步取 sigma_max 80，中间时间步取 [1.5, 1.4, 1.0]，时间步送入模型前先按 bf16 取整；每一步重新注入噪声；不用 CFG，也不用负向 prompt。
  - 帧数用 81，官方脚本默认是 77，这样所有臂的画布都一样。
- **多步后处理链**：`post: "flashvsr+rife"`。每一步是一轮，在各自的 Python 环境里运行；中间帧用无损 npz 交接。计时等于各步耗时之和，不计文件读写。
- **FULL50 复用**：配置里写 `generation: "external"`。plan 记录被复用那次运行的 plan sha；finalize 时逐个核对 prompt 原文、seed 和视频哈希。
- **dev2 prompt**：由 `scripts/data/make_controlled_factor_dev2_prompts.py` 生成：
  - prompt 文件：`prompts/univ_controlled_factor_dev2.jsonl`
  - 需求清单：`prompts/univ_controlled_factor_dev2_requirements.jsonl`
  - id 为 cf100–cf163，origin 为 `controlled_factor_dev2`，split 为 `dev2`。
- **配置文件**：
  - `configs/published_wan21_twostage_v2b.json`
  - `configs/published_wan21_platform_dev2.json`
  - `configs/published_wan21_twostage_dev2.json`
- **对 v2a 的影响**：two-stage 驱动和 worker 改动后，v2a 已冻结的 plan 无法再通过实现哈希校验。v2a 已经完成并导出，不受影响，只是不能再在 v2a 目录里续跑。
