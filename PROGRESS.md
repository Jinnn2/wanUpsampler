# 理论进度

最后更新：2026-04-29

## 当前阶段

当前已经完成：

```text
模型 V1 训练
```

这里的 V1 指第一版最小可验证路线：

- 基于 Wan2.1 latent 空间
- 只做 spatial 2x upsampling
- 输入 `x_t_lr + sigma`
- 输出 `z0_hr`
- 训练目标为 noisy-to-clean

## 当前结论

项目已经从“方案设计阶段”进入“第一轮训练结果评估阶段”。

也就是说，当前重点不再是把训练代码跑通，而是判断：

1. V1 模型是否真正学到了比简单插值更有意义的映射。
2. noisy-to-clean 目标是否带来了稳定收益。
3. 该路线是否值得进入 V2。

## 已完成进度

### P0：理论方案确定

已完成：

- 明确项目不是普通视频超分，而是 Wan 采样中途的 latent 分辨率切换。
- 明确目标是：
  ```text
  U(x_t_lr, sigma) -> z0_hr
  ```
- 明确推理链路是：
  ```text
  LR noisy latent -> upsampler -> HR clean latent -> re-noise -> HR Wan continue
  ```

结论：

- 理论方向成立。
- noisy-to-clean 比 direct noisy-to-noisy 更适合作为第一版起点。

### P1：V1 模型定义

已完成：

- 采用轻量 3D CNN 路线而不是直接上 DiT。
- 引入 sigma conditioning。
- 使用 spatial PixelShuffle 2x。
- loss 采用：
  - latent reconstruction
  - low-frequency consistency
  - temporal consistency

结论：

- V1 的设计目标是先验证“采样中途 latent 放大”是否可学。
- 该设计偏保守，优先保证训练稳定性，而不是一开始追求极限画质。

### P2：V1 训练

已完成：

- 完成第一版模型训练。

当前结论：

- 训练阶段已经进入可评估状态。
- 现在最关键的问题不是“能不能训”，而是“训出来的结果到底有没有意义”。

## 当前关注点

当前只讨论两个核心判断。

### 1. 是否优于简单插值

这是 V1 最关键的判断标准。

需要回答：

- decode 后主体是否更稳定
- 边缘是否比 latent interpolate 更清楚
- temporal consistency 是否更好
- 是否减少闪烁或块状伪影

如果结果只是接近插值，说明：

- V1 只学到了低频平滑映射
- noisy-to-clean 的收益还没有被有效学出来

### 2. 是否适合进入 V2

V1 的使命不是最终效果，而是回答这条路线值不值得继续。

如果 V1 已经表现出：

- 明显优于插值的结构恢复
- 更好的低频一致性
- decode 后较稳定的视频主体

那么可以继续进入 V2。

如果 V1 只是“能训通，但效果接近 baseline”，那下一步就要重新判断：

- 是数据问题
- 是 loss 设计问题
- 是模型容量不够
- 还是 noisy-to-clean 本身还没被训练到位

## 当前阶段判断

目前项目所处的合理描述是：

```text
V1 已训练完成
当前处于第一轮效果验证与路线判断阶段
```

不是：

```text
项目已完成
```

也不是：

```text
仍停留在基础工程搭建阶段
```

## 下一步

下一阶段目标应聚焦在效果判断，而不是继续扩展功能。

优先顺序：

1. 系统对比 V1 与 latent interpolate baseline。
2. 判断 decode 后的视频质量是否有可见提升。
3. 判断是否值得进入 V2。
4. 若值得，进入 V2 的方向包括：
   - 更贴近真实 Wan trajectory 的数据
   - 更强的 re-noise 设计
   - 更高质量的数据分布
   - 更强的模型结构或更高容量

## V2 入口条件

满足下面任意两条，就可以认为 V1 值得继续推进：

- 比简单插值有稳定可见优势
- decode 后主体更稳
- temporal artifacts 更少
- 在中后期 sigma 区间表现明显更好
- 作为 transition 模块具备继续优化价值

## 维护方式

后续这个文档只记录三类内容：

1. 当前处于哪个理论/实验阶段
2. 当前效果判断是什么
3. 下一阶段是否值得继续推进

不再记录：

- 本机路径
- 下载细节
- 环境联调过程
- 脚本修修补补
