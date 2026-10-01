# 已发表加速方法：Wan2.1 首轮比较

## 当前论文主张与边界

**待检验主张**：在已发表的视频加速 pipeline 上，接近的总体视频评分可能掩盖可感知、且与内容相关的退化；更适配加速评价的方案，应通过独立人评验证这种差异。

这不是已经证实的结论。分开报告：①总体均值掩盖局部失败，②某个评分维度本身漏检，③prompt 是否可预测方法效用。前两项成立也不会自动证明第三项。不得为了得到选择器收益而调评分权重。

首轮是 pilot，不是官方全量 VBench Total 的论文表格复现，不是方法速度匹配的质量排名，也不足以训练/证明可泛化的 prompt 选择器。之后才进行校准 prompt 上的等实测耗时调参、独立 prompt-family/seed 测试与路由收益验证。

## 为什么先用现有 Wan2.1

默认复用 `/mnt/afs_2/houze/Wan-AI/Wan2.1-T2V-1.3B`。需要原生 DiT safetensors、T5、VAE 和 tokenizer；只有 LightX2V 转换权重不够。仅增加锁定的官方 Wan 推理源码，不再次下载模型。

| 实验臂 | 固定配置 | 来源/解释 |
|---|---|---|
| FULL50 | 完整 50 步 | 独立锁定的官方 Wan2.1 源码，不用改造版冒充基线 |
| STEP25 | 完整计算 25 步 | 简单步数削减对照，不是新增已发表算法；也不是 ScalingCache 表里的 20 步配置 |
| TEA008 | 50 步，threshold=0.08，无 retention | 官方 1.3B README 配置 |
| SCALING10 | 50 步，first_enhance=10，dynamic_cache/use_alpha | 官方入口及随仓库提供的 1.3B 系数；不重新用试验 prompt 拟合系数 |
| JENGA_BASE | 50 步，threshold=0.15，retention，p_remain=0.9，SA drop=0.75/0.85 | 官方 `wan_1.3B_jenga_base.sh`；**缓存+稀疏注意力复合 pipeline**，未启用 turbo |

VGDFR 无已核实的官方 Wan 入口，留待原版 HunyuanVideo 阶段；PAB/DVG 不强行移植冒充官方复现。方法源码、官方基线源码、入口参数、Python 环境均记录。

ScalingCache 锁定源码有两个非计算入口问题：`adapter/wan/__init__.py` 导入了不存在的 `textimage2video.py`，单卡路径也无条件导入未使用的 xDiT 接口。启动适配层只暴露真实上游 T2V/I2V 类并绕开缺失的无关导入；分布式接口设为**调用即报错**的 guard，不模拟多卡计算。receipt 的 `compatibility_adaptations` 记录这些处理，缓存/forward/采样代码不改写，外部 checkout 不修改。已有依赖中还需要 seaborn。

统一：832×480，81 帧，16 fps，UniPC，shift=8，CFG=6，禁用 prompt 扩写与 CPU offload。Jenga 原示例采用该 shift/CFG；这不代表已核实所有论文表格使用相同设置。速度由实际记录获得，不能套用论文声称的倍数。

所有臂的**稠密 attention 统一锁定上游已有 FlashAttention-2 分支**。当前环境的 `flash_attn_interface` 可能返回 Tensor，而锁定的 Wan/Jenga FA3 包装器把返回值当元组取 `[0]`，导致序列维被错误移除。这是后端接口兼容问题，不应解读为某方法质量失败。启动层只将模块的 `FLASH_ATTN_3_AVAILABLE` 运行标志设为 false，不修改上游源码、attention 算法或缓存阈值，不自动回退 SDPA。receipt 记录后端与接口路径；Jenga 原版稀疏 Triton 路径保持不变。首次 `check` 会运行不加载权重的稠密 CUDA 小算子，确认输出形状、dtype 和有限性。

## 冻结设计

配置：`configs/published_wan21_pilot_v1.json`。

- 12 个逐条写明的 prompt × 2 个 seed（42、3407）× 5 臂 = **120 视频**。
- 低/高运动 × 低/高细节四个名义格，每格 2 个公开 VBench 原文 + 1 个新压力 prompt。
- 公开文本从锁定的 ScalingCache 仓库附带 `assets/VBench_full_info.json` 校验，保留原维度/辅助 metadata。这是有目的的子集，不是随机代表整个 VBench。
- 名义格不是视频实测分类。例如公开的“马奔跑”未必低细节；必须在 `base_observability_review.csv` 记录 FULL 是否表达所需运动和细节、是否基模型失败。不要因看到评分后删样本。主分析保留全体；预先说明的可观察性分层另报。
- 2 个额外校准 prompt，seed=12345，不进入正式人评或路由训练：5 臂 + 3 个关闭加速的实现对照 = **16 校准视频**。
- 默认每 GPU/臂进程额外做一次完整形状的预热，使用排除的校准 prompt、不保存视频、不计时。生成阶段最多额外 40 次预热，校准最多 16 次；需把这些计算计入机器预算。只加载一次权重，随后复用 pipeline；ScalingCache 在上一条释放缓存后重新初始化。

### 校准放行

记录真实初始噪声 SHA256、负 prompt、采样参数、dtype、环境和设备；所有同组必须匹配。关闭 TeaCache、ScalingCache、Jenga 的结果，与 FULL 的解码前固定浮点网格比较（时间/空间每 4 点采样，MAE≤0.001、最大误差≤0.02）。阈值在看正式结果前固定。

**网格通过不是全像素精确相同的证明**；这是实现兼容性关卡。失败会阻止正式生成。不要随意放宽阈值：先检查采样器、负 prompt、attention、数值/排序差异；必要时另立协议使用各实现自己的关闭基线，并停止“共享一个 FULL”的归因。

### 耗时范围

预热后的 wall time 包含 prompt 编码、pipeline 设置（包括 Jenga 空间曲线构建）、去噪和 VAE；排除模型加载、MP4 编码、噪声哈希及审计采样开销。各项另存，峰值显存另存。不额外添加 RGB 超分或 HR refine。

权重身份默认按路径/大小/mtime 绑定，小 JSON 同时做 SHA256；**没有声称完成大权重逐字节密码学校验**。外部源码使用固定 commit，视频与输出审计样本做 SHA256。冻结后修改脚本/配置/源版本会被拒绝，应换输出目录。

## 服务器运行

在仓库根目录，先同步代码。默认已是你的模型路径，可用 `MODEL_ROOT=...` 覆盖；8 卡无需 torchrun，每卡独立处理。

推荐使用专用 `PUBLISHED_WAN21_ROOT` 指定输出，它优先于旧实验可能残留的 `DATASET_ROOT`；若误指定含 Phase3/Phase4 清单的目录，程序会拒绝串用，不修改旧数据。首次环境检查会写冻结 plan，更新脚本后应选新目录（例如 `published_wan21_pilot_v2`），不要删除已有计划或视频绕过校验。

运行时禁写外部仓库的 Python 字节码。源码检查只容许未跟踪的 `__pycache__/*.pyc`/`.pyo`，真实源码增删改依然拒绝并显示路径。Jenga 的 `gilbert.py` 已加入必需文件；稀疏检出遗漏时运行 `fetch` 会从锁定 commit 补齐，不要安装 pip 同名包。

如果运行失败，主进程会直接显示失败日志末尾。`diagnose` 是只读诊断，可以检查旧目录，既不加载模型，也不要求旧 plan 与新脚本 hash 匹配：

```bash
PUBLISHED_WAN21_ROOT=/mnt/afs_2/houze/wanUpsampler/outputs/univ_sparse_action_phase3_v1 \
  bash UNIV_adaptor/scripts/run_univ_published_wan21_8gpu.sh diagnose
```

```bash
bash UNIV_adaptor/scripts/run_univ_published_wan21_8gpu.sh fetch

# 如果现有环境缺少官方 Wan/Jenga 依赖，推荐独立环境。
# 不替换当前已正常工作的 torch/torchvision/CUDA/triton/flash-attn 栈。
bash UNIV_adaptor/scripts/run_univ_published_wan21_8gpu.sh setup
# 若确实缺 flash-attn 且需要本地编译，才显式执行：
# INSTALL_FLASH_ATTN=1 bash UNIV_adaptor/scripts/run_univ_published_wan21_8gpu.sh setup

bash UNIV_adaptor/scripts/run_univ_published_wan21_8gpu.sh check
bash UNIV_adaptor/scripts/run_univ_published_wan21_8gpu.sh calibrate
bash UNIV_adaptor/scripts/run_univ_published_wan21_8gpu.sh audit
```

`check` 会首次冻结 plan、检查原生权重/源码/ffmpeg，以及每个入口的参数、CUDA 导入和小型稠密 attention 算子。`calibrate` 首次运行完整形状的生成模型与 Jenga 稀疏算子。先看校准结果和耗时，再决定是否启动完整 120 条；不要跳过校准。

```bash
bash UNIV_adaptor/scripts/run_univ_published_wan21_8gpu.sh generate
bash UNIV_adaptor/scripts/run_univ_published_wan21_8gpu.sh status
bash UNIV_adaptor/scripts/run_univ_published_wan21_8gpu.sh finalize
bash UNIV_adaptor/scripts/run_univ_published_wan21_8gpu.sh score
bash UNIV_adaptor/scripts/run_univ_published_wan21_8gpu.sh blind
bash UNIV_adaptor/scripts/run_univ_published_wan21_8gpu.sh blind-score
bash UNIV_adaptor/scripts/run_univ_published_wan21_8gpu.sh export
```

默认输出 `outputs/published_wan21_pilot_v1`，每 GPU/方法的日志在 `logs/`。Ctrl+C 会停止本次调度的活跃子进程，已验证 receipt 可以原命令继续；中断后产生的无 receipt 视频不会被默默当作完成，也不会覆盖，需要检查后移动该**具体文件**再继续。

主调度每 30 秒打印已保存 receipt 数与活跃 GPU 数；详细采样进度在逐进程日志里。耗时/预算请先以 16 条校准及被记录的预热开销估算，不保证官方 README 的速度能在本机原样复现。

如仅先分析已完成匹配组，用独立兄弟目录保存快照（原视频按路径引用，不复制大视频）：

```bash
bash UNIV_adaptor/scripts/run_univ_published_wan21_8gpu.sh finalize --allow-partial \
  --snapshot-out /mnt/afs_2/houze/wanUpsampler/outputs/published_wan21_pilot_snapshot01
DATASET_ROOT=/mnt/afs_2/houze/wanUpsampler/outputs/published_wan21_pilot_snapshot01 \
  bash UNIV_adaptor/scripts/run_univ_published_wan21_8gpu.sh score
```

它明确标记 exploratory、只评分完整组。最终清单不可变，后续完成更多视频不能覆盖该快照，正式盲评要求全部匹配组完成。原目录继续生成；保留原视频以维持快照引用。

可设置 `PYTHON_WAN21`、`PYTHON_TEACACHE`、`PYTHON_SCALINGCACHE`、`PYTHON_JENGA`；校准严格检查版本一致，独立环境不能悄悄改变 PyTorch 或数学库。VBench 始终用 `VBENCH_PYTHON`，默认旧 `/opt/conda/bin/python`。评分 commit 默认沿用当前锁定值；若你的合法评分仓库不是这个版本，显式填真实 40 位 commit 并用新输出目录，不要填 `<locked-commit>`。

## 评分、人评和交付

逐维记录 subject/background consistency、motion smoothness、aesthetic/imaging quality；dynamic degree 与 overall consistency 单列诊断。`vbench5` 是原始五维均值，**不是官方归一化 Quality/Total**。自定义压力 prompt 不伪造人类动作/颜色等官方任务标签。完整官方 suite 的数值复现是后续独立任务，不由这 12 条代替。

所有 24 个 prompt-seed 组都比较 FULL 与各加速臂，**96 个主 pair**；另加 6 个随机固定重复 pair，标为 reliability_repeat、不进入主比较。抽样不读分数，不只挑“看起来失败”的视频。方法和基线身份隐藏，A/B 与顺序按 rater 独立随机。至少 3 位独立 rater，FULL 可以输；可以平局/不确定。

评价四项：整体、细节、时间稳定性/运动、prompt 实现。在 notes 中记录 `A/B: 缺陷类型; 严重度0-3; 时间戳秒; 说明`；同时说明正常观看与暂停放大是否不同。现有 UI 的 notes 是自由文本，并未自动强制结构字段或给出严重度标签。分析必须保留争议和平局。根据评分选的“近分数对”只能另作 exploratory，不能替换全体主 pair。

先运行 `blind-score` 对**实际展示副本**重新评分，防止把原视频分数直接套在转码展示上。人评完成，把 JSON 放回 `blind/private/ratings/` 后：

```bash
bash UNIV_adaptor/scripts/run_univ_published_wan21_8gpu.sh blind-report
bash UNIV_adaptor/scripts/run_univ_published_wan21_8gpu.sh export
```

`blind-report` 同时生成整体人评一致性、**方法 × 名义内容格**偏好与展示分数漏检/反转表，以及重复题的 rater 内一致性；平局、争议及未达最低人数的 pair 保留，不被消失在一个 accuracy 数字中。

导出 `exports/`：

- `published_wan21_analysis.tgz`：清单、配置、校准、逐维结果、原始评分记录、人评。**私有，只给研究者/本助手分析**。
- `blind_media_000.tgz` 等：独立小包，单包以 64MiB 未压缩视频大小分组（单个大视频例外）。全部解压到同一目录，得到 `study/`，不需要合并分卷；不含方法名/分数。

本机启动，不用远端端口转发：

```powershell
python study/acceleration_blind_audit.py serve-local --out study --port 8765
```

打开 `http://127.0.0.1:8765`。评价者编号必须唯一，恢复时用相同编号；答案存于 `study/private/ratings/`。不要用通用 HTTP 静态服务器暴露 private。为了后续本地分析，保留展示视频和独立的私有分析包。

## 本地验证范围

可运行 `python -m unittest UNIV_adaptor.tests.test_published_wan21_pilot`：协议/冻结/配对、校准拒绝路径、模拟 worker 复用/预热/计时/receipt 和模拟评分导出。无本地 CUDA 权重，不能声称已验证 Jenga Triton 算子或整套 GPU 实际结果；请以服务器校准日志为准。
