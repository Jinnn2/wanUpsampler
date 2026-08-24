#!/usr/bin/env python3
"""
Model architectures and loss functions for Prompt-Conditioned Optimal Switching Routers.
Includes:
  - LinearOrdinalRouter (B3: Monotonic Thresholds + Exact Token Attribution)
  - LinearProbeRouter (B1: Direct Linear Baseline)
  - SoftDistillationMLPRouter (B4: Non-linear KL + Wasserstein Distillation)
  - RelativeQualityCurveMLPRouter (B4-Q: Relative Quality Curve Regression)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── 1. Linear Monotonic Ordinal Router (Core Model B3) ──────────────────────────

class LinearOrdinalRouter(nn.Module):
    """
    Learns a 1D Semantic Switching Direction s_p = w^T h_p + b_0,
    with strictly monotonic thresholds b_1 < b_2 < ... < b_{K-1}.
    Enables exact linear token attribution: r_i = w^T h_i.
    """

    def __init__(self, in_dim: int = 4096, num_classes: int = 13):
        super().__init__()
        self.in_dim = in_dim
        self.K = num_classes  # e.g. 13 candidate steps

        # Linear projection to 1D switch score
        self.linear = nn.Linear(in_dim, 1, bias=True)

        # Base threshold b_1
        self.base_threshold = nn.Parameter(torch.tensor(0.0))
        # Strictly positive deltas via softplus: b_k = b_1 + sum(softplus(delta))
        if self.K > 2:
            self.raw_deltas = nn.Parameter(torch.zeros(self.K - 2))
        else:
            self.register_parameter("raw_deltas", None)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.linear.weight, std=0.01)
        nn.init.zeros_(self.linear.bias)
        # Spread initial thresholds evenly
        if self.raw_deltas is not None:
            nn.init.constant_(self.raw_deltas, 0.5)

    def get_monotonic_thresholds(self) -> torch.Tensor:
        """Returns ordered thresholds [b_1, b_2, ..., b_{K-1}]."""
        if self.K <= 1:
            return torch.empty(0, device=self.base_threshold.device)
        if self.K == 2:
            return self.base_threshold.unsqueeze(0)

        deltas = F.softplus(self.raw_deltas)
        cum_deltas = torch.cumsum(deltas, dim=0)
        thresholds = torch.cat([
            self.base_threshold.unsqueeze(0),
            self.base_threshold + cum_deltas,
        ])
        return thresholds

    def forward(self, pooled_t5: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        pooled_t5: [B, 4096]
        returns:
          - switch_score: [B] scalar
          - cumulative_logits: [B, K-1] (b_k - s_p)
          - cumulative_probs: [B, K-1] P(k* <= m)
          - discrete_probs: [B, K] P(k* == m)
          - pred_step_idx: [B]
        """
        # s_p: [B]
        s_p = self.linear(pooled_t5).squeeze(-1)
        thresholds = self.get_monotonic_thresholds()  # [K-1]

        # cumulative logits: b_k - s_p => P(k* <= k) = sigma(b_k - s_p)
        # Shape: [B, K-1]
        cum_logits = thresholds.unsqueeze(0) - s_p.unsqueeze(1)
        cum_probs = torch.sigmoid(cum_logits)

        # Compute discrete class probabilities:
        # P(0) = P(k* <= 0)
        # P(m) = P(k* <= m) - P(k* <= m-1)
        # P(K-1) = 1 - P(k* <= K-2)
        p_first = cum_probs[:, :1]
        p_mid = cum_probs[:, 1:] - cum_probs[:, :-1]
        p_last = 1.0 - cum_probs[:, -1:]
        discrete_probs = torch.cat([p_first, p_mid, p_last], dim=1).clamp(min=1e-7, max=1.0)
        discrete_probs = discrete_probs / discrete_probs.sum(dim=-1, keepdim=True)

        pred_idx = torch.argmax(discrete_probs, dim=-1)

        return {
            "switch_score": s_p,
            "thresholds": thresholds,
            "cumulative_logits": cum_logits,
            "cumulative_probs": cum_probs,
            "discrete_probs": discrete_probs,
            "pred_step_idx": pred_idx,
        }

    @torch.no_grad()
    def get_token_attributions(self, seq_embedding: torch.Tensor) -> torch.Tensor:
        """
        Calculates exact token contribution r_i = w^T h_i.
        seq_embedding: [B, L, 4096] or [L, 4096]
        returns: [B, L] or [L] scalar attributions
        """
        w = self.linear.weight  # [1, 4096]
        if seq_embedding.ndim == 2:
            return F.linear(seq_embedding, w).squeeze(-1)  # [L]
        elif seq_embedding.ndim == 3:
            return F.linear(seq_embedding, w).squeeze(-1)  # [B, L]
        else:
            raise ValueError(f"Unexpected shape for seq_embedding: {seq_embedding.shape}")


# ─── 2. Linear Probe Router (Baseline B1) ───────────────────────────────────────

class LinearProbeRouter(nn.Module):
    """Simple linear probe from T5 embedding directly to K class logits."""

    def __init__(self, in_dim: int = 4096, num_classes: int = 13):
        super().__init__()
        self.classifier = nn.Linear(in_dim, num_classes)

    def forward(self, pooled_t5: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.classifier(pooled_t5)  # [B, K]
        probs = F.softmax(logits, dim=-1)
        pred_idx = torch.argmax(probs, dim=-1)
        return {
            "logits": logits,
            "discrete_probs": probs,
            "pred_step_idx": pred_idx,
        }


# ─── 3. Soft Distillation MLP Router (Baseline B4) ──────────────────────────────

class SoftDistillationMLPRouter(nn.Module):
    """
    Non-linear multi-layer perceptron with LayerNorm and Dropout,
    optimized for soft utility distribution matching.
    """

    def __init__(
        self,
        in_dim: int = 4096,
        hidden_dims: list[int] = [256, 128],
        num_classes: int = 13,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.LayerNorm(h),
                nn.SiLU(),
                nn.Dropout(dropout),
            ])
            prev = h
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(prev, num_classes)

    def forward(self, pooled_t5: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.mlp(pooled_t5)
        logits = self.head(feat)  # [B, K]
        probs = F.softmax(logits, dim=-1)
        pred_idx = torch.argmax(probs, dim=-1)
        return {
            "logits": logits,
            "discrete_probs": probs,
            "pred_step_idx": pred_idx,
        }


class RelativeQualityCurveMLPRouter(SoftDistillationMLPRouter):
    """B4-compatible MLP that regresses quality relative to the final candidate."""

    def forward(self, pooled_t5: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.mlp(pooled_t5)
        quality_deltas = self.head(feat)
        return {"quality_deltas": quality_deltas}


# ─── 4. Specialized Loss Functions ─────────────────────────────────────────────

class OrdinalLoss(nn.Module):
    """
    Ordinal binary cross-entropy loss over K-1 cumulative thresholds.
    y_k = 1 if opt_idx > k else 0.
    """

    def __init__(self):
        super().__init__()

    def forward(self, cumulative_logits: torch.Tensor, ordinal_targets: torch.Tensor) -> torch.Tensor:
        # cumulative_logits: [B, K-1], ordinal_targets: [B, K-1]
        # P(opt_idx > k) = 1 - P(opt_idx <= k) = sigma(s_p - b_k) = sigma(-cum_logits)
        logits_greater = -cumulative_logits
        return F.binary_cross_entropy_with_logits(logits_greater, ordinal_targets)


class SoftUtilityKLLoss(nn.Module):
    """KL Divergence between soft utility targets and predicted probabilities."""

    def __init__(self):
        super().__init__()

    def forward(self, pred_logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(pred_logits, dim=-1)
        # KL(target || model)
        return F.kl_div(log_probs, soft_targets, reduction="batchmean")


class Wasserstein1Loss(nn.Module):
    """Earth Mover's Distance / 1D Wasserstein distance on cumulative distributions."""

    def __init__(self):
        super().__init__()

    def forward(self, pred_probs: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
        pred_cdf = torch.cumsum(pred_probs, dim=-1)
        target_cdf = torch.cumsum(target_probs, dim=-1)
        return torch.mean(torch.abs(pred_cdf - target_cdf))
