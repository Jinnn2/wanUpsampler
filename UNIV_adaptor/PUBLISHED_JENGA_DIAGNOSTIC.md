# Jenga 校准差异：独立诊断

## 主张与当前证据

本轮只诊断实现，不证明视觉退化、VBench 漏检或 prompt 路由收益。原论文待检验主张保持为：已发表加速 pipeline 的接近总体分数可能掩盖可感知、内容相关的退化，需独立人评验证。实现有效性是前置条件，不能把运行异常或基线不兼容当成质量结论。

v3 的两个校准 prompt 中，所有臂的 noise/sampling/environment 一致；TEA_OFF、SCALING_OFF 在浮点采样网格上与 FULL50 相同，只有 JENGA_OFF 超阈值。JENGA_BASE 在排除预热后的耗时为 934/877 秒，FULL50 为 212/210 秒，属于当前配置/环境下的持续开销，不能只用首次编译解释，也不是对论文速度的普遍否定。

源码中 `teacache_thresh=0` 仍使 `enable_teacache=True`；非 retention 的 1.3B 判定多项式在部分非负输入上可为负，所以零阈值并非逻辑上的严格关闭。实际命中需计数，不能只凭该可能性解释已有输出。即使关闭稀疏路径，Jenga 仍按 Gilbert 顺序重排 token 和 RoPE；数学上等价的重排也可能改变低精度求和顺序，不能单凭输出差异指称重排索引错误。

## 固定诊断设计

直接复用 `published_wan21_pilot_v3` 全部 16 条校准记录，校验冻结计划、原 worker/driver 哈希、源码 commit、权重 inventory、receipt/视频/浮点样本哈希。旧目录只读，不复制视频、不改计划/审计报告，不重跑 FULL、TeaCache、ScalingCache 或昂贵的 JENGA_BASE。

新增两个原校准 prompt、原 seed=12345、每 prompt 三个对照，共 **6 条**：

| 对照 | 缓存 | token 顺序 | 用途 |
|---|---|---|---|
| JENGA_ZERO_COUNTED | 原入口阈值 0，逻辑不变 | 原 Gilbert | 记录实际命中，检查与旧 JENGA_OFF 重跑一致性 |
| JENGA_HARD_OFF | 每次 forward 前强制 `enable_teacache=False` | 原 Gilbert | 排除缓存复用，保留重排和原前向 |
| JENGA_IDENTITY_OFF | 同样严格关闭 | 正序 identity，正反映射同步替换 | 区分重排/数值因素与其他前向差异 |

所有对照都使用原 JENGA_OFF 的 50 步、零 SA drop、p_remain=1；未启用 turbo。修改仅存在于独立诊断进程，调用原前向和原 attention 内核，不编辑外部仓库，也不更改官方 JENGA_BASE。取消重排的对照仍保留入口的曲线构建，**不能用其耗时声称去掉重排后的最优系统速度**。

逐视频计数：forward、cache enabled、cache hit、hard OFF、identity order、dense attention、sparse attention。报告严格检查 100 次条件/无条件 forward、严格关闭时 0 hit、0 sparse call，以及每次实际计算的 forward 对应 30 层 × 2 个 dense attention。计数只增整数，不做逐层 CUDA 同步；诊断耗时含计数 hook 的 CPU 开销，不混入原方法速度排名。

默认 `NGPUS=8`，实际只使用 GPU 0/1/2，各卡一个对照，复用模型处理两个 prompt。每卡另跑一次完整形状预热、不保存视频：总计 **6 次测量生成 + 3 次预热**。`NGPUS=1/2` 也可用，但各对照进程仍各自预热。校准 prompt 不进入正式评测/训练。

## 运行

在服务器仓库根目录，现有生成环境/权重可直接复用，无需 setup/fetch：

```bash
git pull
export SOURCE_PUBLISHED_ROOT=/mnt/afs_2/houze/wanUpsampler/outputs/published_wan21_pilot_v3
export JENGA_DIAGNOSTIC_ROOT=/mnt/afs_2/houze/wanUpsampler/outputs/published_wan21_jenga_diagnostic_v1

bash UNIV_adaptor/scripts/run_univ_published_jenga_diagnostic.sh check
```

`check` 冻结独立诊断计划，确认入口及小型 dense FA2 CUDA 算子；仍不是完整视频验证。成功后：

```bash
bash UNIV_adaptor/scripts/run_univ_published_jenga_diagnostic.sh generate
bash UNIV_adaptor/scripts/run_univ_published_jenga_diagnostic.sh report
```

各命令仅在上一步成功后执行。Ctrl+C 只终止本次子进程，完成的哈希有效 receipt 可续跑。更新诊断脚本后换诊断目录，不动旧 v3。

请提供 `published_wan21_jenga_diagnostic_v1/diagnostic_report.json`；也有简表 `diagnostic_report.md` 和逐进程 `logs/`。旧结果始终按引用复用，删除/移动旧视频会使验证失败。独立环境路径默认与原生成一致，可通过 `PYTHON_BIN`/`PYTHON_JENGA` 指定，但报告仍要求和 FULL 校准环境匹配。

## 如何读结果

- 原零阈值出现 cache hit：证明该次计数运行确实复用了缓存；比较严格关闭后的变化，不能把全部差异立即归给缓存。
- 严格关闭仍不同、identity 恢复：支持重排/数值路径是差异来源，**不自动证明算法或索引错误**。
- identity 仍不同：继续检查前向、采样器与模型输出精度；不放宽原 MAE≤0.001/max≤0.02。
- 原零阈值重跑与旧 OFF 不一致：优先审查可重复性/环境，因果解释暂不成立。
- 两个 prompt 不是统计确认；即便全部通过也不自动放行原 pilot，不覆盖 `calibration_audit.json`。

后续是否采用各实现自己的 OFF 基线，或把 Jenga 移到单独的实现/性能附录，需要另立显式协议。不得因为其质量结果不利而选择性排除，也不得把 identity 对照冒充官方方法。稀疏内核的真实性能定位是随后独立工作，本脚本没有重实现或优化它。
