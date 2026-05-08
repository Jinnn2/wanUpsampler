# Wan2.2 LMDB Reference

这组脚本来自早期 Wan2.2 / LMDB 数据构建思路，包含 clean latent 生成和 down latent 补写逻辑。

注意：

- 依赖官方 Wan2.2 环境。
- 部分脚本引用 `utils.*`，不是当前仓库的可直接运行入口。
- 当前 480p -> 720p clean-latent 主线已经迁移到 `changing_resolution/`。
