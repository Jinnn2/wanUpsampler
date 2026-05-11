from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class PlainResBlock3D(nn.Module):
    """不改变张量尺寸的 3D 残差块。

    输入/输出形状都是 [B, channels, T, H, W]。
    这里的 3D 卷积会同时看时间维 T 和空间维 H/W，用来在 latent 特征域里修正局部时空结构。
    """

    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()

        # 第一次归一化。GroupNorm 不依赖 batch 维统计量，适合当前 batch_size 较小的训练设置。
        self.norm1 = nn.GroupNorm(_valid_groups(channels), channels)

        # 第一个 3D 卷积。kernel_size=3 且 padding=1，因此 T/H/W 尺寸都不会变化。
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)

        # 第二次归一化，为第二个卷积前的激活做稳定化。
        self.norm2 = nn.GroupNorm(_valid_groups(channels), channels)

        # 可选 Dropout3d。默认 dropout=0，此时用 Identity，训练和推理都不额外改动特征。
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

        # 第二个 3D 卷积，同样保持通道数和 T/H/W 尺寸不变。
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, H, W]，作为残差块的主输入。

        # 1. norm1 先稳定每组通道的数值分布。
        # 2. SiLU 提供非线性表达能力。
        # 3. conv1 提取局部 3D 时空特征。
        h = self.conv1(F.silu(self.norm1(x)))

        # 1. norm2 + SiLU 继续做非线性变换。
        # 2. dropout 只在配置启用时随机屏蔽部分 3D 特征。
        # 3. conv2 输出与 x 同形状的修正量。
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))

        # 残差连接：保留原始特征 x，只把卷积分支学到的修正 h 加上去。
        return x + h


class WanCleanLatentResizer(nn.Module):
    """LightX2V changing_resolution 使用的 clean latent 空间升分模型。

    输入和输出都是 [B, C, T, H, W]：
    - B: batch size
    - C: Wan VAE latent 通道数，当前默认是 16
    - T: latent 时间帧数，本模型不改变 T
    - H/W: latent 空间尺寸，本模型只放大 H/W

    训练目标是 clean latent resize，不是 RGB 超分，也不是 noisy latent 去噪。
    当前 480p -> 720p 主线里，典型 latent 形状是：
    [B, 16, T, 60, 104] -> [B, 16, T, 90, 156]。

    整体结构：
    z0_lr -> stem -> pre_blocks -> trilinear feature resize -> post_blocks
          -> output residual -> trilinear(z0_lr) + residual -> pred_z0_hr
    """

    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 16,
        hidden_channels: int = 256,
        num_res_blocks: int = 8,
        scale_factor: float = 1.5,
        dropout: float = 0.0,
        residual_skip: bool = True,
    ) -> None:
        super().__init__()

        # 至少需要把残差块分成 resize 前和 resize 后两段，所以要求数量不小于 2。
        if num_res_blocks < 2:
            raise ValueError("num_res_blocks must be at least 2")

        # scale_factor 只在 forward 没有显式 output_size 时使用，仍然必须是正数。
        if scale_factor <= 0:
            raise ValueError("scale_factor must be positive")

        # 记录输入/输出通道。当前 Wan clean latent 默认 C=16。
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 默认 1.5x，对应当前 latent 60x104 -> 90x156。
        self.scale_factor = float(scale_factor)

        # residual_skip=True 时，最终输出 = trilinear(z0_lr) + learned residual。
        # 如果输入/输出通道不同，skip 无法直接相加，因此自动关闭。
        self.residual_skip = residual_skip and in_channels == out_channels

        # stem 把原始 latent 通道投影到更宽的 hidden feature 空间。
        # 形状从 [B, in_channels, T, H, W] 变为 [B, hidden_channels, T, H, W]。
        self.stem = nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1)

        # 前半段残差块运行在 LR latent 网格上，用于理解低分辨率 latent 的局部结构。
        pre_blocks = num_res_blocks // 2

        # 后半段残差块运行在 HR latent 网格上，用于在放大后的空间位置上修正细节。
        post_blocks = num_res_blocks - pre_blocks

        # resize 前的残差块列表。每一层都保持 [B, hidden_channels, T, H_lr, W_lr]。
        self.pre_blocks = nn.ModuleList(
            [PlainResBlock3D(hidden_channels, dropout=dropout) for _ in range(pre_blocks)]
        )

        # resize 后的残差块列表。每一层都保持 [B, hidden_channels, T, H_hr, W_hr]。
        self.post_blocks = nn.ModuleList(
            [PlainResBlock3D(hidden_channels, dropout=dropout) for _ in range(post_blocks)]
        )

        # 输出前再做一次归一化，稳定 residual 预测分支。
        self.out_norm = nn.GroupNorm(_valid_groups(hidden_channels), hidden_channels)

        # 把 hidden feature 投回目标 latent 通道数，输出的是 learned residual。
        self.out = nn.Conv3d(hidden_channels, out_channels, kernel_size=3, padding=1)

    def forward(
        self,
        z0_lr: torch.Tensor,
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        # z0_lr 是低分辨率 clean latent，必须是 [B, C, T, H, W]。
        # 这里提前检查维度，避免 F.interpolate 或 Conv3d 报出难读的底层错误。
        if z0_lr.ndim != 5:
            raise ValueError(f"z0_lr must be [B, C, T, H, W], got {tuple(z0_lr.shape)}")

        # 通道数必须和模型初始化时的 in_channels 一致，当前默认是 16。
        if z0_lr.shape[1] != self.in_channels:
            raise ValueError(f"expected {self.in_channels} channels, got {z0_lr.shape[1]}")

        # 计算目标空间尺寸 H_hr/W_hr。
        # 训练和评测通常传入 z0_hr 的真实 H/W，推理时也可以只依赖 scale_factor。
        target_h, target_w = self._target_spatial_size(z0_lr, output_size)

        # F.interpolate 处理 5D 张量时的 size 顺序是 (T, H, W)。
        # changing_resolution 只做空间升分，所以 T 保持 z0_lr.shape[2] 不变。
        target_size = (z0_lr.shape[2], target_h, target_w)

        # Step 1: stem 通道投影。
        # 从 Wan latent 的 16 通道进入 256 维 hidden feature 空间，T/H/W 暂时不变。
        h = self.stem(z0_lr)

        # Step 2: LR 网格上的特征修正。
        # 这些 block 还没有升分，感受野对应原始 LR latent 网格。
        for block in self.pre_blocks:
            h = block(h)

        # Step 3: hidden feature 升分。
        # 这是 Stage 1 的固定 resize 点：用 trilinear 把 hidden feature 放到目标 H/W。
        # align_corners=False 与 PyTorch 常用图像/特征 resize 语义一致，避免角点强行对齐。
        h = F.interpolate(h, size=target_size, mode="trilinear", align_corners=False)

        # Step 4: HR 网格上的特征修正。
        # 此时 h 已经处在目标空间尺寸上，后续卷积学习如何在 HR latent 网格中补细节。
        for block in self.post_blocks:
            h = block(h)

        # Step 5: 输出 learned residual。
        # out_norm + SiLU 先整理 hidden feature，再用 3D 卷积投回 out_channels。
        residual = self.out(F.silu(self.out_norm(h)))

        # 如果没有 residual skip，模型直接返回完整预测。
        # 这种模式适合输入/输出通道不同，或显式想让网络自己生成完整 HR latent 的实验。
        if not self.residual_skip:
            return residual

        # Step 6: 构造固定插值基线。
        # skip 是直接把 z0_lr 用同样的 trilinear 算子放大到目标 T/H/W。
        skip = F.interpolate(z0_lr, size=target_size, mode="trilinear", align_corners=False)

        # Step 7: 最终预测。
        # 输出 = 固定插值基线 + 网络学习到的残差。
        # 这样模型不需要从零生成 HR latent，只需要学习 trilinear 缺失的结构修正。
        return skip + residual

    def _target_spatial_size(
        self,
        z0_lr: torch.Tensor,
        output_size: tuple[int, int] | None,
    ) -> tuple[int, int]:
        # 优先使用调用方传入的精确目标 H/W。
        # 训练中这个值来自 z0_hr.shape[-2:]，可以避免 round(scale_factor) 的边界误差。
        if output_size is not None:
            if len(output_size) != 2:
                raise ValueError("output_size must be (height, width)")
            return int(output_size[0]), int(output_size[1])

        # 没有显式 output_size 时，用 scale_factor 根据 LR latent 尺寸估算目标尺寸。
        # round 用于处理非整数缩放后的尺寸；当前 1.5x 主线通常会得到精确整数。
        return (
            int(round(z0_lr.shape[-2] * self.scale_factor)),
            int(round(z0_lr.shape[-1] * self.scale_factor)),
        )


def _valid_groups(channels: int, preferred: int = 32) -> int:
    # GroupNorm 要求 channels 能被 groups 整除。
    # 优先尝试 32 组；如果不能整除，就逐步减小，直到找到合法组数。
    groups = min(preferred, channels)
    while channels % groups != 0:
        groups -= 1
    return groups
