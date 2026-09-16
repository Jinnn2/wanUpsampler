from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import json
import math
import random
import sys
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from UNIV_adaptor.combined_v3 import QUALITY_DIMENSIONS, load_json, verify_file  # noqa: E402
from UNIV_adaptor.data_protocol import (  # noqa: E402
    canonical_sha256,
    sha256_file,
    write_json_atomic,
)
from UNIV_adaptor.scripts.router.train_combined_v3_b4_control import (  # noqa: E402
    FIXED_MODEL,
    VARIABLE_MODEL,
    b4_ensemble_probabilities,
    paired_bootstrap,
    soft_utility_targets,
)
from UNIV_adaptor.scripts.router.train_combined_v3_budget_prior import (  # noqa: E402
    load_samples,
    train_latency_profile,
    validate_embedding_manifest,
    validate_merged_index,
    write_csv,
)


POOLED = "bounded_residual_pooled"
ATTENTION = "bounded_residual_token_attention"
BIAS_ONLY = "train_quality_bias_only"
SHUFFLED = "shuffled_prompt_control"
ARCHITECTURES = (POOLED, ATTENTION)
ACTION_FEATURE_NAMES = (
    "spatial_ratio",
    "temporal_ratio",
    "lr_nfe_ratio",
    "switch_ratio",
    "true_lr_steps_ratio",
    "hr_steps_ratio",
    "renoise_sigma",
    "proxy_compute_density",
    "normalized_measured_cost",
    "has_hr_refinement",
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def action_feature_matrix(
    action_catalog: list[dict[str, Any]], normalized_cost: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    if len(action_catalog) != len(normalized_cost):
        raise ValueError("action catalog and cost length differ")
    raw = []
    for item, cost in zip(action_catalog, normalized_cost, strict=True):
        execution = item.get("execution_action")
        requested = item.get("requested_action")
        if isinstance(execution, dict):
            spatial = float(execution["spatial_ratio"])
            temporal = float(execution["temporal_ratio"])
            lr_nfe = 1.0
            switch = 1.0
            true_lr = float(execution["true_lr_steps"]) / 50.0
            hr_steps = float(execution["hr_steps"]) / 50.0
            renoise = float(execution["renoise_sigma"])
        elif isinstance(requested, dict):
            spatial = float(requested["spatial_ratio"])
            temporal = float(requested["temporal_ratio"])
            lr_nfe = float(requested["lr_nfe_ratio"])
            switch = float(requested["switch_ratio"])
            true_lr = lr_nfe * switch
            hr_steps = 0.0
            renoise = 0.0
        else:
            raise ValueError(
                f"action {item.get('artifact_id')} has no supported descriptor"
            )
        proxy = float(item.get("proxy_compute_density", cost))
        values = (
            spatial,
            temporal,
            lr_nfe,
            switch,
            true_lr,
            hr_steps,
            renoise,
            proxy,
            float(cost),
            float(hr_steps > 0.0),
        )
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite action feature: {item.get('artifact_id')}")
        raw.append(values)
    raw_array = np.asarray(raw, dtype=np.float32)
    mean = raw_array.mean(axis=0)
    scale = raw_array.std(axis=0)
    scale = np.where(scale > 1e-6, scale, 1.0).astype(np.float32)
    normalized = (raw_array - mean) / scale
    metadata = {
        "feature_names": list(ACTION_FEATURE_NAMES),
        "raw_features": raw_array.tolist(),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
    }
    return normalized.astype(np.float32), metadata


class StructuredQualityPrior(nn.Module):
    """Bounded prompt-conditioned residual around a fixed train quality prior."""

    def __init__(
        self,
        prompt_dim: int,
        action_features: torch.Tensor,
        global_quality: torch.Tensor,
        *,
        architecture: str,
        hidden_dim: int = 64,
        action_hidden_dim: int = 32,
        dropout: float = 0.1,
        max_residual: float = 0.03,
    ) -> None:
        super().__init__()
        if architecture not in ARCHITECTURES:
            raise ValueError(f"unsupported architecture: {architecture}")
        if action_features.ndim != 2 or global_quality.ndim != 1:
            raise ValueError("invalid action feature or quality shape")
        if action_features.shape[0] != global_quality.numel():
            raise ValueError("action count mismatch")
        self.architecture = architecture
        self.max_residual = float(max_residual)
        self.prompt_norm = nn.LayerNorm(prompt_dim)
        self.prompt_projection = nn.Linear(prompt_dim, hidden_dim)
        self.prompt_dropout = nn.Dropout(dropout)
        if architecture == ATTENTION:
            self.attention_query = nn.Parameter(torch.zeros(hidden_dim))
        else:
            self.register_parameter("attention_query", None)
        self.action_encoder = nn.Sequential(
            nn.LayerNorm(action_features.shape[1]),
            nn.Linear(action_features.shape[1], action_hidden_dim),
            nn.SiLU(),
            nn.Linear(action_hidden_dim, hidden_dim),
        )
        self.interaction = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.zeros_(self.interaction.weight)
        self.register_buffer("action_features", action_features.detach().clone())
        self.register_buffer("global_quality", global_quality.detach().clone())

    def prompt_representation(
        self, prompt_embedding: torch.Tensor, attention_mask: torch.Tensor | None
    ) -> torch.Tensor:
        if self.architecture == POOLED:
            if prompt_embedding.ndim != 2 or attention_mask is not None:
                raise ValueError("pooled model expects [batch, dim] without mask")
            return self.prompt_dropout(
                F.silu(self.prompt_projection(self.prompt_norm(prompt_embedding)))
            )
        if prompt_embedding.ndim != 3 or attention_mask is None:
            raise ValueError("attention model expects [batch, tokens, dim] and mask")
        token_state = F.silu(self.prompt_projection(self.prompt_norm(prompt_embedding)))
        token_state = self.prompt_dropout(token_state)
        logits = torch.einsum("bth,h->bt", token_state, self.attention_query)
        logits = logits / math.sqrt(token_state.shape[-1])
        logits = logits.masked_fill(~attention_mask.bool(), -torch.inf)
        weights = torch.softmax(logits, dim=1)
        return torch.einsum("bt,bth->bh", weights, token_state)

    def forward(
        self, prompt_embedding: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        prompt_state = self.prompt_representation(prompt_embedding, attention_mask)
        action_state = self.action_encoder(self.action_features)
        interaction = torch.einsum(
            "bh,ah->ba", self.interaction(prompt_state), action_state
        ) / math.sqrt(action_state.shape[-1])
        residual = self.max_residual * torch.tanh(interaction)
        return self.global_quality.unsqueeze(0) + residual


def utility_aligned_loss(
    predicted_quality: torch.Tensor,
    true_quality: torch.Tensor,
    normalized_cost: torch.Tensor,
    lambdas: list[float],
    *,
    soft_target_tau: float,
    emd_weight: float,
    pairwise_weight: float,
    quality_weight: float,
    pairwise_tau: float,
    quality_huber_beta: float,
    cost_order: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    if predicted_quality.shape != true_quality.shape or predicted_quality.ndim != 2:
        raise ValueError("quality tensors must share [batch, action] shape")
    if not lambdas:
        raise ValueError("at least one lambda is required")
    device = predicted_quality.device
    cost = normalized_cost.to(device)
    order = cost_order.to(device)
    upper = torch.triu_indices(
        predicted_quality.shape[1], predicted_quality.shape[1], offset=1, device=device
    )
    total_kl = predicted_quality.new_zeros(())
    total_emd = predicted_quality.new_zeros(())
    total_pairwise = predicted_quality.new_zeros(())
    for lambda_value in lambdas:
        pred_utility = predicted_quality - lambda_value * cost.unsqueeze(0)
        true_utility = true_quality - lambda_value * cost.unsqueeze(0)
        target = soft_utility_targets(
            true_quality, cost, lambda_value, temperature=soft_target_tau
        )
        log_prob = F.log_softmax(pred_utility / soft_target_tau, dim=1)
        probability = torch.softmax(pred_utility / soft_target_tau, dim=1)
        total_kl = total_kl + F.kl_div(log_prob, target, reduction="batchmean")
        ordered_probability = probability.index_select(1, order)
        ordered_target = target.index_select(1, order)
        total_emd = total_emd + torch.mean(
            torch.abs(
                torch.cumsum(ordered_probability, dim=1)
                - torch.cumsum(ordered_target, dim=1)
            )
        )
        pred_margin = (
            pred_utility[:, upper[0]] - pred_utility[:, upper[1]]
        ) / pairwise_tau
        target_margin = torch.sigmoid(
            (true_utility[:, upper[0]] - true_utility[:, upper[1]]) / pairwise_tau
        )
        total_pairwise = total_pairwise + F.binary_cross_entropy_with_logits(
            pred_margin, target_margin
        )
    count = float(len(lambdas))
    kl = total_kl / count
    emd = total_emd / count
    pairwise = total_pairwise / count
    quality = F.smooth_l1_loss(predicted_quality, true_quality, beta=quality_huber_beta)
    loss = kl + emd_weight * emd + pairwise_weight * pairwise + quality_weight * quality
    return loss, {
        "kl": float(kl.detach()),
        "emd": float(emd.detach()),
        "pairwise": float(pairwise.detach()),
        "quality": float(quality.detach()),
    }


def deterministic_folds(
    prompt_ids: list[int], fold_count: int, seed: int
) -> list[np.ndarray]:
    if fold_count < 2 or fold_count > len(prompt_ids):
        raise ValueError("invalid fold count")
    order = np.arange(len(prompt_ids), dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(order)
    return [
        np.asarray(value, dtype=np.int64) for value in np.array_split(order, fold_count)
    ]


def deranged_indices(indices: np.ndarray, seed: int) -> np.ndarray:
    if indices.size < 2:
        raise ValueError("derangement requires at least two prompts")
    rng = np.random.default_rng(seed)
    result = indices.copy()
    for _ in range(1000):
        rng.shuffle(result)
        if np.all(result != indices):
            return result.copy()
    return np.roll(indices, 1)


class PromptStore:
    def __init__(
        self,
        samples: list[dict[str, Any]],
        sequence_paths: dict[int, Path],
    ) -> None:
        self.samples = samples
        self.pooled = torch.from_numpy(
            np.stack([row["embedding"] for row in samples])
        ).float()
        self.sequence_paths = [
            sequence_paths[int(row["global_prompt_id"])] for row in samples
        ]
        self._sequence_cache: dict[int, torch.Tensor] = {}

    def _sequence(self, index: int) -> torch.Tensor:
        cached = self._sequence_cache.get(index)
        if cached is not None:
            return cached
        with np.load(self.sequence_paths[index], allow_pickle=False) as payload:
            value = np.asarray(payload["seq_embedding"], dtype=np.float32)
            mask = np.asarray(payload["attention_mask"], dtype=np.int64)
        if value.ndim != 2 or value.shape[1] != self.pooled.shape[1]:
            raise ValueError(f"invalid token embedding: {self.sequence_paths[index]}")
        valid = int(np.count_nonzero(mask))
        if valid < 1 or valid > value.shape[0]:
            raise ValueError(f"invalid token mask: {self.sequence_paths[index]}")
        tensor = torch.from_numpy(value[:valid]).to(torch.float16)
        self._sequence_cache[index] = tensor
        return tensor

    def batch(
        self, feature_indices: np.ndarray, architecture: str, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if architecture == POOLED:
            return self.pooled[torch.from_numpy(feature_indices)].to(device), None
        sequences = [self._sequence(int(index)) for index in feature_indices]
        length = max(value.shape[0] for value in sequences)
        batch = torch.zeros(
            (len(sequences), length, self.pooled.shape[1]), dtype=torch.float16
        )
        mask = torch.zeros((len(sequences), length), dtype=torch.bool)
        for row, value in enumerate(sequences):
            batch[row, : value.shape[0]] = value
            mask[row, : value.shape[0]] = True
        return batch.to(device=device, dtype=torch.float32), mask.to(device)


def embedding_sequence_paths(
    dataset_root: Path, embedding_manifest: dict[str, Any]
) -> dict[int, Path]:
    root = (dataset_root / "t5_embeddings").resolve()
    paths = {}
    for item in embedding_manifest["prompts"]:
        prompt_id = int(item["prompt_id"])
        path = verify_file(item["npz_file"], item["npz_sha256"], label="T5 embedding")
        if path.parent != root:
            raise ValueError("T5 sequence path escapes embedding root")
        paths[prompt_id] = path
    return paths


def batch_indices(
    indices: np.ndarray, batch_size: int, seed: int
) -> Iterable[np.ndarray]:
    order = indices.copy()
    np.random.default_rng(seed).shuffle(order)
    for start in range(0, len(order), batch_size):
        yield order[start : start + batch_size]


@torch.no_grad()
def predict_quality(
    model: StructuredQualityPrior,
    store: PromptStore,
    indices: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    rows = []
    for start in range(0, len(indices), batch_size):
        current = indices[start : start + batch_size]
        embedding, mask = store.batch(current, model.architecture, device)
        rows.append(model(embedding, mask).cpu().numpy())
    return np.concatenate(rows, axis=0)


def mean_policy_regret(
    predicted_quality: np.ndarray,
    true_quality: np.ndarray,
    normalized_cost: np.ndarray,
    lambdas: list[float],
) -> float:
    values = []
    for lambda_value in lambdas:
        true_utility = true_quality - lambda_value * normalized_cost[None, :]
        choices = (predicted_quality - lambda_value * normalized_cost[None, :]).argmax(
            axis=1
        )
        realized = true_utility[np.arange(len(choices)), choices]
        values.append(float(np.mean(true_utility.max(axis=1) - realized)))
    return float(np.mean(values))


def make_model(
    *,
    prompt_dim: int,
    action_features: torch.Tensor,
    global_quality: torch.Tensor,
    architecture: str,
    args: argparse.Namespace,
    device: torch.device,
) -> StructuredQualityPrior:
    return StructuredQualityPrior(
        prompt_dim,
        action_features,
        global_quality,
        architecture=architecture,
        hidden_dim=args.hidden_dim,
        action_hidden_dim=args.action_hidden_dim,
        dropout=args.dropout,
        max_residual=args.max_residual,
    ).to(device)


def train_epochs(
    *,
    model: StructuredQualityPrior,
    store: PromptStore,
    quality: torch.Tensor,
    train_indices: np.ndarray,
    feature_indices: np.ndarray,
    epochs: int,
    args: argparse.Namespace,
    normalized_cost: torch.Tensor,
    cost_order: torch.Tensor,
    device: torch.device,
    eval_indices: np.ndarray | None = None,
    eval_quality: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    if len(train_indices) != len(feature_indices):
        raise ValueError("label and prompt-feature indices differ")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-10,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(epochs, int(getattr(args, "max_epochs", epochs))),
        eta_min=args.min_lr,
    )
    feature_by_label = {
        int(label): int(feature)
        for label, feature in zip(train_indices, feature_indices, strict=True)
    }
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        totals: dict[str, float] = defaultdict(float)
        seen = 0
        for labels in batch_indices(
            train_indices, args.batch_size, args.data_seed + epoch
        ):
            features = np.asarray(
                [feature_by_label[int(i)] for i in labels], dtype=np.int64
            )
            embedding, mask = store.batch(features, model.architecture, device)
            target = quality[torch.from_numpy(labels)].to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(embedding, mask)
            loss, components = utility_aligned_loss(
                prediction,
                target,
                normalized_cost,
                args.lambdas,
                soft_target_tau=args.soft_target_tau,
                emd_weight=args.emd_weight,
                pairwise_weight=args.pairwise_weight,
                quality_weight=args.quality_weight,
                pairwise_tau=args.pairwise_tau,
                quality_huber_beta=args.quality_huber_beta,
                cost_order=cost_order,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch}")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count = len(labels)
            totals["loss"] += float(loss.detach()) * count
            totals["gradient_norm"] += float(gradient_norm) * count
            for name, value in components.items():
                totals[name] += value * count
            seen += count
        scheduler.step()
        row: dict[str, Any] = {
            "epoch": epoch,
            **{f"train_{name}": value / seen for name, value in totals.items()},
            "learning_rate": scheduler.get_last_lr()[0],
        }
        if eval_indices is not None:
            if eval_quality is None:
                raise ValueError("eval quality is required with eval indices")
            predicted = predict_quality(
                model,
                store,
                eval_indices,
                batch_size=args.batch_size,
                device=device,
            )
            row["calibration_policy_regret"] = mean_policy_regret(
                predicted, eval_quality, normalized_cost.cpu().numpy(), args.lambdas
            )
        history.append(row)
        if epoch == 1 or epoch == epochs or epoch % 5 == 0:
            progress = {
                "architecture": model.architecture,
                "epoch": epoch,
                "epochs": epochs,
                "train_loss": row["train_loss"],
            }
            if "calibration_policy_regret" in row:
                progress["calibration_policy_regret"] = row["calibration_policy_regret"]
            print(json.dumps(progress), flush=True)
    return history


def cross_validate(
    *,
    architecture: str,
    train_seed: int,
    store: PromptStore,
    quality: torch.Tensor,
    action_features: torch.Tensor,
    normalized_cost: torch.Tensor,
    cost_order: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    shuffled: bool,
    out_root: Path,
    provenance: dict[str, Any],
) -> tuple[int, list[dict[str, Any]]]:
    prompt_ids = [int(row["global_prompt_id"]) for row in store.samples]
    folds = deterministic_folds(prompt_ids, args.cv_folds, args.fold_seed)
    fold_histories = []
    all_indices = np.arange(len(store.samples), dtype=np.int64)
    for fold_index, calibration_indices in enumerate(folds):
        train_indices = np.setdiff1d(
            all_indices, calibration_indices, assume_unique=True
        )
        feature_indices = train_indices.copy()
        if shuffled:
            feature_indices = deranged_indices(
                feature_indices, train_seed * 1009 + fold_index
            )
        control = SHUFFLED if shuffled else "real_prompt"
        fold_path = (
            out_root
            / "cv"
            / control
            / architecture
            / f"seed_{train_seed}"
            / f"fold_{fold_index}.json"
        )
        fold_config = {
            "architecture": architecture,
            "control": control,
            "train_seed": train_seed,
            "fold": fold_index,
            "train_prompt_ids": [prompt_ids[int(index)] for index in train_indices],
            "feature_prompt_ids": [prompt_ids[int(index)] for index in feature_indices],
            "calibration_prompt_ids": [
                prompt_ids[int(index)] for index in calibration_indices
            ],
            "max_epochs": args.max_epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "min_lr": args.min_lr,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "hidden_dim": args.hidden_dim,
            "action_hidden_dim": args.action_hidden_dim,
            "max_residual": args.max_residual,
            "lambdas": args.lambdas,
            "soft_target_tau": args.soft_target_tau,
            "emd_weight": args.emd_weight,
            "pairwise_weight": args.pairwise_weight,
            "quality_weight": args.quality_weight,
            "pairwise_tau": args.pairwise_tau,
            "quality_huber_beta": args.quality_huber_beta,
            "fold_seed": args.fold_seed,
            "data_seed": args.data_seed,
        }
        if fold_path.is_file():
            saved = load_json(fold_path)
            saved_body = {
                key: value
                for key, value in saved.items()
                if key not in {"schema", "fold_sha256"}
            }
            if (
                saved.get("schema") != "univ_combined_v3_prompt_prior_v2_cv_fold_v1"
                or canonical_sha256(saved_body) != saved.get("fold_sha256")
                or saved.get("provenance") != provenance
                or saved.get("config") != fold_config
            ):
                raise RuntimeError(f"incompatible completed CV fold: {fold_path}")
            fold_history = saved["history"]
            print(f"Reusing completed CV fold: {fold_path}", flush=True)
        else:
            print(
                f"Training CV fold {fold_index + 1}/{args.cv_folds}: "
                f"{control}, {architecture}, seed={train_seed}",
                flush=True,
            )
            fold_global_quality = quality[torch.from_numpy(train_indices)].mean(dim=0)
            seed_everything(train_seed * 1009 + fold_index)
            model = make_model(
                prompt_dim=store.pooled.shape[1],
                action_features=action_features,
                global_quality=fold_global_quality,
                architecture=architecture,
                args=args,
                device=device,
            )
            fold_history = train_epochs(
                model=model,
                store=store,
                quality=quality,
                train_indices=train_indices,
                feature_indices=feature_indices,
                epochs=args.max_epochs,
                args=args,
                normalized_cost=normalized_cost,
                cost_order=cost_order,
                device=device,
                eval_indices=calibration_indices,
                eval_quality=quality[torch.from_numpy(calibration_indices)].numpy(),
            )
            fold_body = {
                "provenance": provenance,
                "config": fold_config,
                "history": fold_history,
            }
            write_json_atomic(
                fold_path,
                {
                    "schema": "univ_combined_v3_prompt_prior_v2_cv_fold_v1",
                    "fold_sha256": canonical_sha256(fold_body),
                    **fold_body,
                },
            )
        for row in fold_history:
            fold_histories.append({"fold": fold_index, **row})
    by_epoch: dict[int, list[float]] = defaultdict(list)
    for row in fold_histories:
        by_epoch[int(row["epoch"])].append(float(row["calibration_policy_regret"]))
    means = {epoch: float(np.mean(values)) for epoch, values in by_epoch.items()}
    best_epoch = min(means, key=lambda epoch: (means[epoch], epoch))
    return int(best_epoch), fold_histories


def run_name(architecture: str, train_seed: int, shuffled: bool) -> str:
    prefix = SHUFFLED if shuffled else architecture
    return f"{prefix}/seed_{train_seed}"


def train_final_run(
    *,
    architecture: str,
    train_seed: int,
    selected_epoch: int,
    shuffled: bool,
    store: PromptStore,
    quality: torch.Tensor,
    action_features: torch.Tensor,
    normalized_cost: torch.Tensor,
    cost_order: torch.Tensor,
    action_ids: list[str],
    action_feature_metadata: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    provenance: dict[str, Any],
    out_root: Path,
) -> dict[str, Any]:
    run_dir = out_root / "runs" / run_name(architecture, train_seed, shuffled)
    summary_path = run_dir / "run_summary.json"
    config = {
        "architecture": architecture,
        "control": SHUFFLED if shuffled else "real_prompt",
        "train_seed": train_seed,
        "selected_epoch": selected_epoch,
        "hidden_dim": args.hidden_dim,
        "action_hidden_dim": args.action_hidden_dim,
        "dropout": args.dropout,
        "max_residual": args.max_residual,
        "lambdas": args.lambdas,
        "loss": {
            "soft_target_tau": args.soft_target_tau,
            "emd_weight": args.emd_weight,
            "pairwise_weight": args.pairwise_weight,
            "quality_weight": args.quality_weight,
            "pairwise_tau": args.pairwise_tau,
            "quality_huber_beta": args.quality_huber_beta,
        },
        "optimizer": "AdamW_beta0.9_0.95_eps1e-10_cosine",
        "lr": args.lr,
        "min_lr": args.min_lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "checkpoint_selection": "train_only_5fold_cv_epoch",
    }
    if summary_path.is_file():
        summary = load_json(summary_path)
        body = {
            key: value
            for key, value in summary.items()
            if key not in {"schema", "run_sha256"}
        }
        if (
            summary.get("schema") != "univ_combined_v3_prompt_prior_v2_run_v1"
            or canonical_sha256(body) != summary.get("run_sha256")
            or summary.get("provenance") != provenance
            or summary.get("training_config") != config
        ):
            raise RuntimeError(f"incompatible completed V2 run: {summary_path}")
        result = summary["result"]
        verify_file(
            result["checkpoint"], result["checkpoint_sha256"], label="V2 checkpoint"
        )
        print(f"Reusing completed V2 run: {run_dir}", flush=True)
        return result
    if run_dir.exists():
        raise RuntimeError(
            f"incomplete V2 run exists; use a new output root: {run_dir}"
        )
    seed_everything(train_seed)
    print(
        f"Training final V2 run: {run_name(architecture, train_seed, shuffled)}, "
        f"epochs={selected_epoch}",
        flush=True,
    )
    global_quality = quality.mean(dim=0)
    model = make_model(
        prompt_dim=store.pooled.shape[1],
        action_features=action_features,
        global_quality=global_quality,
        architecture=architecture,
        args=args,
        device=device,
    )
    indices = np.arange(len(store.samples), dtype=np.int64)
    features = (
        deranged_indices(indices, train_seed * 1009 + 9973)
        if shuffled
        else indices.copy()
    )
    history = train_epochs(
        model=model,
        store=store,
        quality=quality,
        train_indices=indices,
        feature_indices=features,
        epochs=selected_epoch,
        args=args,
        normalized_cost=normalized_cost,
        cost_order=cost_order,
        device=device,
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = run_dir / "prompt_prior_v2.pt"
    checkpoint = {
        "schema": "univ_combined_v3_prompt_prior_v2_checkpoint_v1",
        "architecture": architecture,
        "control": SHUFFLED if shuffled else "real_prompt",
        "state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "prompt_dim": store.pooled.shape[1],
        "hidden_dim": args.hidden_dim,
        "action_hidden_dim": args.action_hidden_dim,
        "dropout": args.dropout,
        "max_residual": args.max_residual,
        "action_ids": action_ids,
        "action_feature_metadata": action_feature_metadata,
        "train_seed": train_seed,
        "selected_epoch": selected_epoch,
        "provenance": provenance,
        "training_config": config,
    }
    torch.save(checkpoint, checkpoint_path)
    write_csv(run_dir / "training_history.csv", history)
    result = {
        "architecture": architecture,
        "method": SHUFFLED if shuffled else architecture,
        "train_seed": train_seed,
        "selected_epoch": selected_epoch,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }
    body = {"provenance": provenance, "training_config": config, "result": result}
    write_json_atomic(
        summary_path,
        {
            "schema": "univ_combined_v3_prompt_prior_v2_run_v1",
            "run_sha256": canonical_sha256(body),
            **body,
        },
    )
    return result


def load_model(
    path: Path, device: torch.device
) -> tuple[StructuredQualityPrior, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "univ_combined_v3_prompt_prior_v2_checkpoint_v1":
        raise ValueError(f"unsupported V2 checkpoint: {path}")
    features = torch.tensor(
        payload["action_feature_metadata"]["raw_features"], dtype=torch.float32
    )
    mean = torch.tensor(payload["action_feature_metadata"]["mean"], dtype=torch.float32)
    scale = torch.tensor(
        payload["action_feature_metadata"]["scale"], dtype=torch.float32
    )
    features = (features - mean) / scale
    global_quality = payload["state_dict"]["global_quality"]
    model = StructuredQualityPrior(
        int(payload["prompt_dim"]),
        features,
        global_quality,
        architecture=str(payload["architecture"]),
        hidden_dim=int(payload["hidden_dim"]),
        action_hidden_dim=int(payload["action_hidden_dim"]),
        dropout=float(payload["dropout"]),
        max_residual=float(payload["max_residual"]),
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device).eval(), payload


def ensemble_quality(
    checkpoints: list[Path],
    store: PromptStore,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, list[tuple[int, np.ndarray]]]:
    indices = np.arange(len(store.samples), dtype=np.int64)
    per_seed = []
    for path in checkpoints:
        model, payload = load_model(path, device)
        prediction = predict_quality(
            model, store, indices, batch_size=batch_size, device=device
        )
        per_seed.append((int(payload["train_seed"]), prediction))
    return np.mean([value for _, value in per_seed], axis=0), per_seed


def load_b4_control_choices(
    b4_root: Path | None,
    *,
    dataset_sha256: str,
    embedding_manifest_sha256: str,
    latency_profile_sha256: str,
    action_ids: list[str],
    lambdas: list[float],
    validation_embeddings: torch.Tensor,
    device: torch.device,
) -> tuple[dict[str, dict[float, np.ndarray]], dict[str, Any] | None]:
    if b4_root is None:
        return {}, None
    summary_path = b4_root / "selection_summary.json"
    summary = load_json(summary_path)
    summary_body = {
        key: value
        for key, value in summary.items()
        if key not in {"schema", "selection_sha256"}
    }
    if (
        summary.get("schema") != "univ_combined_v3_b4_control_selection_v1"
        or canonical_sha256(summary_body) != summary.get("selection_sha256")
        or summary.get("test_accessed")
        or summary.get("dataset", {}).get("dataset_sha256") != dataset_sha256
        or summary.get("embedding_manifest_sha256") != embedding_manifest_sha256
        or summary.get("latency_profile", {}).get("profile_sha256")
        != latency_profile_sha256
        or summary.get("action_ids") != action_ids
        or [float(value) for value in summary.get("lambdas", [])] != lambdas
    ):
        raise ValueError("B4 control is incompatible with this V2 experiment")
    fixed_paths: dict[float, list[Path]] = {value: [] for value in lambdas}
    variable_paths = []
    checkpoint_hashes = []
    for run in summary.get("runs", []):
        model_type = str(run.get("model_type"))
        if model_type not in {FIXED_MODEL, VARIABLE_MODEL}:
            continue
        checkpoint = verify_file(
            run["checkpoint"], run["checkpoint_sha256"], label="B4 checkpoint"
        )
        checkpoint_hashes.append(run["checkpoint_sha256"])
        if model_type == FIXED_MODEL:
            fixed_paths[float(run["lambda"])].append(checkpoint)
        else:
            variable_paths.append(checkpoint)
    if any(not fixed_paths[value] for value in lambdas) or not variable_paths:
        raise ValueError("B4 control has incomplete checkpoint coverage")
    choices: dict[str, dict[float, np.ndarray]] = {
        FIXED_MODEL: {},
        VARIABLE_MODEL: {},
    }
    for lambda_value in lambdas:
        fixed_probability, _ = b4_ensemble_probabilities(
            fixed_paths[lambda_value],
            validation_embeddings,
            lambda_value,
            device=device,
        )
        variable_probability, _ = b4_ensemble_probabilities(
            variable_paths,
            validation_embeddings,
            lambda_value,
            device=device,
        )
        choices[FIXED_MODEL][lambda_value] = fixed_probability.argmax(axis=1)
        choices[VARIABLE_MODEL][lambda_value] = variable_probability.argmax(axis=1)
    return choices, {
        "path": str(summary_path.resolve()),
        "file_sha256": sha256_file(summary_path),
        "selection_sha256": summary["selection_sha256"],
        "checkpoint_set_sha256": canonical_sha256(sorted(checkpoint_hashes)),
        "methods": [FIXED_MODEL, VARIABLE_MODEL],
    }


def best_cost_matched_mixture(
    train_quality: np.ndarray, normalized_cost: np.ndarray, target_cost: float
) -> dict[str, float | int]:
    mean_quality = train_quality.mean(axis=0)
    candidates = []
    for left in range(len(normalized_cost)):
        for right in range(left, len(normalized_cost)):
            c0, c1 = float(normalized_cost[left]), float(normalized_cost[right])
            if left == right:
                if not math.isclose(c0, target_cost, abs_tol=1e-10):
                    continue
                weight_right = 0.0
            else:
                lower, upper = sorted((c0, c1))
                if target_cost < lower - 1e-10 or target_cost > upper + 1e-10:
                    continue
                weight_right = (target_cost - c0) / (c1 - c0)
            quality = (1.0 - weight_right) * mean_quality[
                left
            ] + weight_right * mean_quality[right]
            candidates.append((float(quality), left, right, float(weight_right)))
    if not candidates:
        nearest = int(np.abs(normalized_cost - target_cost).argmin())
        return {
            "left": nearest,
            "right": nearest,
            "right_weight": 0.0,
            "cost": float(normalized_cost[nearest]),
        }
    _, left, right, weight_right = max(
        candidates, key=lambda value: (value[0], -value[1], -value[2])
    )
    return {
        "left": left,
        "right": right,
        "right_weight": weight_right,
        "cost": target_cost,
    }


def evaluation_rows(
    *,
    train_samples: list[dict[str, Any]],
    validation_samples: list[dict[str, Any]],
    action_ids: list[str],
    normalized_cost: np.ndarray,
    lambdas: list[float],
    predictions: dict[str, np.ndarray],
    primary_method: str,
    per_seed_predictions: dict[str, list[tuple[int, np.ndarray]]],
    soft_target_tau: float,
    external_choices: dict[str, dict[float, np.ndarray]],
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]
]:
    train_quality = np.stack([row["candidate_quality"] for row in train_samples])
    val_quality = np.stack([row["candidate_quality"] for row in validation_samples])
    dimensions = {
        name: np.stack([row["dimensions"][name] for row in validation_samples])
        for name in QUALITY_DIMENSIONS
    }
    methods = {
        BIAS_ONLY: np.broadcast_to(train_quality.mean(axis=0), val_quality.shape)
    }
    methods.update(predictions)
    rows = []
    summaries = []
    score_rows = []
    mixture_spec: dict[str, Any] = {}
    for lambda_value in lambdas:
        true_utility = val_quality - lambda_value * normalized_cost[None, :]
        oracle = true_utility.argmax(axis=1)
        oracle_utility = true_utility[np.arange(len(oracle)), oracle]
        choices_by_method = {
            method: (quality - lambda_value * normalized_cost[None, :]).argmax(axis=1)
            for method, quality in methods.items()
        }
        for method, choices_by_lambda in external_choices.items():
            choices = np.asarray(choices_by_lambda[lambda_value], dtype=np.int64)
            if choices.shape != (len(validation_samples),):
                raise ValueError(f"invalid external choices for {method}")
            choices_by_method[method] = choices
        primary_choices = choices_by_method[primary_method]
        target_cost = float(np.mean(normalized_cost[primary_choices]))
        mixture = best_cost_matched_mixture(train_quality, normalized_cost, target_cost)
        mixture_name = f"cost_matched_mixture_for_{primary_method}"
        mixture_spec[f"{lambda_value:.2f}"] = {
            "left_action_id": action_ids[int(mixture["left"])],
            "right_action_id": action_ids[int(mixture["right"])],
            "right_weight": float(mixture["right_weight"]),
            "target_normalized_cost": target_cost,
        }
        choices_by_method["prompt_oracle_upper_bound"] = oracle
        for method, choices in choices_by_method.items():
            realized_quality = val_quality[np.arange(len(choices)), choices]
            realized_utility = true_utility[np.arange(len(choices)), choices]
            summaries.append(
                {
                    "lambda": lambda_value,
                    "method": method,
                    "mean_policy_regret": float(
                        np.mean(oracle_utility - realized_utility)
                    ),
                    "mean_realized_utility": float(np.mean(realized_utility)),
                    "mean_realized_vbench5": float(np.mean(realized_quality)),
                    "mean_normalized_cost": float(np.mean(normalized_cost[choices])),
                    "oracle_exact_action_rate": float(np.mean(choices == oracle)),
                }
            )
            for index, sample in enumerate(validation_samples):
                choice = int(choices[index])
                row = {
                    "global_prompt_id": sample["global_prompt_id"],
                    "prompt_sha256": sample["prompt_sha256"],
                    "seed_count": sample["seed_count"],
                    "lambda": lambda_value,
                    "method": method,
                    "chosen_action_id": action_ids[choice],
                    "realized_vbench5": float(val_quality[index, choice]),
                    "normalized_cost": float(normalized_cost[choice]),
                    "realized_utility": float(realized_utility[index]),
                    "oracle_utility": float(oracle_utility[index]),
                    "policy_regret": float(
                        oracle_utility[index] - realized_utility[index]
                    ),
                }
                for dimension, values in dimensions.items():
                    row[dimension] = float(values[index, choice])
                rows.append(row)
        left, right = int(mixture["left"]), int(mixture["right"])
        weight = float(mixture["right_weight"])
        expected_quality = (1.0 - weight) * val_quality[:, left] + weight * val_quality[
            :, right
        ]
        expected_utility = expected_quality - lambda_value * float(mixture["cost"])
        summaries.append(
            {
                "lambda": lambda_value,
                "method": mixture_name,
                "mean_policy_regret": float(np.mean(oracle_utility - expected_utility)),
                "mean_realized_utility": float(np.mean(expected_utility)),
                "mean_realized_vbench5": float(np.mean(expected_quality)),
                "mean_normalized_cost": float(mixture["cost"]),
                "oracle_exact_action_rate": "",
            }
        )
        for index, sample in enumerate(validation_samples):
            rows.append(
                {
                    "global_prompt_id": sample["global_prompt_id"],
                    "prompt_sha256": sample["prompt_sha256"],
                    "seed_count": sample["seed_count"],
                    "lambda": lambda_value,
                    "method": mixture_name,
                    "chosen_action_id": f"mixture:{action_ids[left]}:{action_ids[right]}:{weight:.8f}",
                    "realized_vbench5": float(expected_quality[index]),
                    "normalized_cost": float(mixture["cost"]),
                    "realized_utility": float(expected_utility[index]),
                    "oracle_utility": float(oracle_utility[index]),
                    "policy_regret": float(
                        oracle_utility[index] - expected_utility[index]
                    ),
                }
            )
        for method, quality in predictions.items():
            for index, sample in enumerate(validation_samples):
                target = torch.softmax(
                    torch.from_numpy(true_utility[index]).float() / soft_target_tau,
                    dim=0,
                ).numpy()
                for action_index, action_id in enumerate(action_ids):
                    score_rows.append(
                        {
                            "global_prompt_id": sample["global_prompt_id"],
                            "lambda": lambda_value,
                            "method": method,
                            "action_id": action_id,
                            "true_quality": float(val_quality[index, action_index]),
                            "predicted_quality": float(quality[index, action_index]),
                            "true_utility": float(true_utility[index, action_index]),
                            "predicted_utility": float(
                                quality[index, action_index]
                                - lambda_value * normalized_cost[action_index]
                            ),
                            "soft_utility_target": float(target[action_index]),
                        }
                    )
        for method, seed_values in per_seed_predictions.items():
            for train_seed, quality in seed_values:
                choices = (quality - lambda_value * normalized_cost[None, :]).argmax(
                    axis=1
                )
                realized = true_utility[np.arange(len(choices)), choices]
                summaries.append(
                    {
                        "lambda": lambda_value,
                        "method": f"{method}_seed_{train_seed}",
                        "mean_policy_regret": float(np.mean(oracle_utility - realized)),
                        "mean_realized_utility": float(np.mean(realized)),
                        "mean_realized_vbench5": float(
                            np.mean(val_quality[np.arange(len(choices)), choices])
                        ),
                        "mean_normalized_cost": float(
                            np.mean(normalized_cost[choices])
                        ),
                        "oracle_exact_action_rate": float(np.mean(choices == oracle)),
                    }
                )
    return summaries, rows, score_rows, mixture_spec


def write_csv_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def material_harm_summary(
    rows: list[dict[str, Any]], candidate_methods: list[str], epsilon: float
) -> list[dict[str, Any]]:
    index = {
        (str(row["method"]), float(row["lambda"]), int(row["global_prompt_id"])): row
        for row in rows
    }
    prompt_ids = sorted({int(row["global_prompt_id"]) for row in rows})
    lambdas = sorted({float(row["lambda"]) for row in rows})
    result = []
    for method in candidate_methods:
        macro = []
        for lambda_value in lambdas:
            deltas = np.asarray(
                [
                    float(index[(method, lambda_value, prompt_id)]["realized_utility"])
                    - float(
                        index[(BIAS_ONLY, lambda_value, prompt_id)]["realized_utility"]
                    )
                    for prompt_id in prompt_ids
                ]
            )
            rate = float(np.mean(deltas < -epsilon))
            macro.extend(deltas < -epsilon)
            result.append(
                {
                    "method": method,
                    "lambda": lambda_value,
                    "harm_epsilon": epsilon,
                    "material_harm_rate": rate,
                }
            )
        result.append(
            {
                "method": method,
                "lambda": "macro",
                "harm_epsilon": epsilon,
                "material_harm_rate": float(np.mean(macro)),
            }
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--b4-root", default=None)
    parser.add_argument(
        "--train-seeds", nargs="+", type=int, default=[42, 100, 2024, 31415, 27182]
    )
    parser.add_argument(
        "--lambdas",
        nargs="+",
        type=float,
        default=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10],
    )
    parser.add_argument(
        "--architectures", nargs="+", choices=ARCHITECTURES, default=list(ARCHITECTURES)
    )
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--action-hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-residual", type=float, default=0.03)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--soft-target-tau", type=float, default=0.02)
    parser.add_argument("--emd-weight", type=float, default=0.5)
    parser.add_argument("--pairwise-weight", type=float, default=0.25)
    parser.add_argument("--quality-weight", type=float, default=1.0)
    parser.add_argument("--pairwise-tau", type=float, default=0.02)
    parser.add_argument("--quality-huber-beta", type=float, default=0.02)
    parser.add_argument("--hardware-label", default="unspecified_generation_device")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=2027)
    parser.add_argument("--fold-seed", type=int, default=1729)
    parser.add_argument("--data-seed", type=int, default=811)
    parser.add_argument("--harm-epsilon", type=float, default=0.001)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    args.lambdas = sorted(set(float(value) for value in args.lambdas))
    if len(args.train_seeds) < 5 or len(args.train_seeds) != len(set(args.train_seeds)):
        parser.error("train-seeds must contain at least five unique seeds")
    if len(args.architectures) != len(set(args.architectures)):
        parser.error("architectures must be unique")
    if not args.lambdas or any(value < 0.0 for value in args.lambdas):
        parser.error("lambdas must be non-empty and non-negative")
    positive = (
        args.cv_folds,
        args.max_epochs,
        args.batch_size,
        args.hidden_dim,
        args.action_hidden_dim,
        args.max_residual,
        args.lr,
        args.min_lr,
        args.soft_target_tau,
        args.pairwise_tau,
        args.quality_huber_beta,
        args.bootstrap_samples,
    )
    if any(value <= 0 for value in positive):
        parser.error("positive training arguments must be positive")
    if args.weight_decay < 0 or not 0 <= args.dropout < 1 or args.harm_epsilon < 0:
        parser.error("invalid regularization or harm threshold")
    return args


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    dataset_path = dataset_root / "dataset_index.json"
    dataset = validate_merged_index(load_json(dataset_path))
    embedding_manifest, embeddings = validate_embedding_manifest(dataset_root, dataset)
    samples, action_ids = load_samples(dataset, embeddings)
    latency = train_latency_profile(
        dataset, action_ids, hardware_label=args.hardware_label
    )
    normalized_cost_np = np.asarray(
        [latency["normalized_action_cost"][action_id] for action_id in action_ids],
        dtype=np.float64,
    )
    action_features_np, action_feature_metadata = action_feature_matrix(
        dataset["action_catalog"], normalized_cost_np
    )
    action_features = torch.from_numpy(action_features_np).float()
    normalized_cost = torch.from_numpy(normalized_cost_np).float()
    cost_order = torch.argsort(normalized_cost)
    sequence_paths = embedding_sequence_paths(dataset_root, embedding_manifest)
    train_store = PromptStore(samples["train"], sequence_paths)
    train_quality = torch.from_numpy(
        np.stack([row["candidate_quality"] for row in samples["train"]])
    ).float()
    out_root = Path(args.out_root).resolve()
    summary_path = out_root / "selection_summary.json"
    if summary_path.exists():
        raise FileExistsError(f"refusing to overwrite V2 selection: {summary_path}")
    out_root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out_root / "latency_profile.json", latency)
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    loader_path = Path(
        sys.modules[
            "UNIV_adaptor.scripts.router.train_combined_v3_budget_prior"
        ].__file__
    ).resolve()
    provenance = {
        "dataset_sha256": dataset["dataset_sha256"],
        "embedding_manifest_sha256": embedding_manifest["manifest_sha256"],
        "text_encoder_checkpoint_sha256": embedding_manifest.get(
            "text_encoder_checkpoint_sha256"
        ),
        "latency_profile_sha256": latency["profile_sha256"],
        "evaluation_stage": "validation_once_after_train_only_cv",
        "test_accessed": False,
        "trainer_sha256": sha256_file(Path(__file__).resolve()),
        "combined_v3_loader_sha256": sha256_file(loader_path),
    }
    cv_rows = []
    selected_epochs: dict[str, dict[int, int]] = defaultdict(dict)
    architecture_cv_regret: dict[str, float] = {}
    for architecture in args.architectures:
        for train_seed in args.train_seeds:
            best_epoch, fold_rows = cross_validate(
                architecture=architecture,
                train_seed=train_seed,
                store=train_store,
                quality=train_quality,
                action_features=action_features,
                normalized_cost=normalized_cost,
                cost_order=cost_order,
                args=args,
                device=device,
                shuffled=False,
                out_root=out_root,
                provenance=provenance,
            )
            selected_epochs[architecture][train_seed] = best_epoch
            for row in fold_rows:
                cv_rows.append(
                    {
                        "architecture": architecture,
                        "control": "real_prompt",
                        "train_seed": train_seed,
                        **row,
                    }
                )
    for architecture in args.architectures:
        values = []
        for train_seed in args.train_seeds:
            epoch = selected_epochs[architecture][train_seed]
            values.extend(
                float(row["calibration_policy_regret"])
                for row in cv_rows
                if row["architecture"] == architecture
                and row["control"] == "real_prompt"
                and int(row["train_seed"]) == train_seed
                and int(row["epoch"]) == epoch
            )
        architecture_cv_regret[architecture] = float(np.mean(values))
    primary = min(
        architecture_cv_regret, key=lambda name: (architecture_cv_regret[name], name)
    )
    shuffled_epochs: dict[int, int] = {}
    for train_seed in args.train_seeds:
        best_epoch, fold_rows = cross_validate(
            architecture=primary,
            train_seed=train_seed,
            store=train_store,
            quality=train_quality,
            action_features=action_features,
            normalized_cost=normalized_cost,
            cost_order=cost_order,
            args=args,
            device=device,
            shuffled=True,
            out_root=out_root,
            provenance=provenance,
        )
        shuffled_epochs[train_seed] = best_epoch
        for row in fold_rows:
            cv_rows.append(
                {
                    "architecture": primary,
                    "control": SHUFFLED,
                    "train_seed": train_seed,
                    **row,
                }
            )
    write_csv_gz(out_root / "train_cv_history.csv.gz", cv_rows)
    runs = []
    checkpoint_groups: dict[str, list[Path]] = defaultdict(list)
    for architecture in args.architectures:
        for train_seed in args.train_seeds:
            result = train_final_run(
                architecture=architecture,
                train_seed=train_seed,
                selected_epoch=selected_epochs[architecture][train_seed],
                shuffled=False,
                store=train_store,
                quality=train_quality,
                action_features=action_features,
                normalized_cost=normalized_cost,
                cost_order=cost_order,
                action_ids=action_ids,
                action_feature_metadata=action_feature_metadata,
                args=args,
                device=device,
                provenance=provenance,
                out_root=out_root,
            )
            runs.append(result)
            checkpoint_groups[architecture].append(Path(result["checkpoint"]))
    for train_seed in args.train_seeds:
        result = train_final_run(
            architecture=primary,
            train_seed=train_seed,
            selected_epoch=shuffled_epochs[train_seed],
            shuffled=True,
            store=train_store,
            quality=train_quality,
            action_features=action_features,
            normalized_cost=normalized_cost,
            cost_order=cost_order,
            action_ids=action_ids,
            action_feature_metadata=action_feature_metadata,
            args=args,
            device=device,
            provenance=provenance,
            out_root=out_root,
        )
        runs.append(result)
        checkpoint_groups[SHUFFLED].append(Path(result["checkpoint"]))
    # Validation labels are consumed only after architecture and epochs are frozen by train CV.
    validation_store = PromptStore(samples["validation"], sequence_paths)
    b4_root = Path(args.b4_root).resolve() if args.b4_root else None
    b4_choices, b4_control = load_b4_control_choices(
        b4_root,
        dataset_sha256=dataset["dataset_sha256"],
        embedding_manifest_sha256=embedding_manifest["manifest_sha256"],
        latency_profile_sha256=latency["profile_sha256"],
        action_ids=action_ids,
        lambdas=args.lambdas,
        validation_embeddings=validation_store.pooled,
        device=device,
    )
    ensemble_predictions: dict[str, np.ndarray] = {}
    per_seed_predictions: dict[str, list[tuple[int, np.ndarray]]] = {}
    for method, checkpoints in checkpoint_groups.items():
        ensemble, per_seed = ensemble_quality(
            checkpoints, validation_store, batch_size=args.batch_size, device=device
        )
        ensemble_predictions[method] = ensemble
        per_seed_predictions[method] = per_seed
    summaries, rows, score_rows, mixture_spec = evaluation_rows(
        train_samples=samples["train"],
        validation_samples=samples["validation"],
        action_ids=action_ids,
        normalized_cost=normalized_cost_np,
        lambdas=args.lambdas,
        predictions=ensemble_predictions,
        primary_method=primary,
        per_seed_predictions=per_seed_predictions,
        soft_target_tau=args.soft_target_tau,
        external_choices=b4_choices,
    )
    mixture_method = f"cost_matched_mixture_for_{primary}"
    candidate_methods = [*ensemble_predictions, *b4_choices]
    per_lambda, macro = paired_bootstrap(
        rows,
        reference_method=BIAS_ONLY,
        candidate_methods=candidate_methods,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    mixture_per_lambda, mixture_macro = paired_bootstrap(
        rows,
        reference_method=mixture_method,
        candidate_methods=[primary],
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed + 1,
    )
    per_lambda.extend(mixture_per_lambda)
    macro.extend(mixture_macro)
    if FIXED_MODEL in b4_choices:
        b4_per_lambda, b4_macro = paired_bootstrap(
            rows,
            reference_method=FIXED_MODEL,
            candidate_methods=[primary],
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed + 2,
        )
        per_lambda.extend(b4_per_lambda)
        macro.extend(b4_macro)
    harm = material_harm_summary(rows, candidate_methods, args.harm_epsilon)
    write_csv(out_root / "validation_results.csv", summaries)
    write_csv_gz(out_root / "validation_predictions.csv.gz", rows)
    write_csv_gz(out_root / "validation_action_scores.csv.gz", score_rows)
    write_csv(out_root / "validation_paired_bootstrap.csv", per_lambda)
    write_csv(out_root / "validation_macro_paired_bootstrap.csv", macro)
    write_csv(out_root / "validation_material_harm.csv", harm)
    macro_regret = {
        method: float(
            np.mean(
                [
                    float(row["mean_policy_regret"])
                    for row in summaries
                    if row["method"] == method
                ]
            )
        )
        for method in [BIAS_ONLY, *ensemble_predictions, *b4_choices, mixture_method]
    }
    bias_regret = macro_regret[BIAS_ONLY]
    oracle_closure = {
        method: (bias_regret - regret) / bias_regret if bias_regret > 0 else 0.0
        for method, regret in macro_regret.items()
    }
    utility_bootstrap = {
        (row["reference_method"], row["candidate_method"]): row
        for row in macro
        if row["metric"] == "realized_utility"
    }
    primary_seed_positive = 0
    bias_by_lambda = {
        float(row["lambda"]): float(row["mean_policy_regret"])
        for row in summaries
        if row["method"] == BIAS_ONLY
    }
    for train_seed in args.train_seeds:
        seed_method = f"{primary}_seed_{train_seed}"
        seed_macro = float(
            np.mean(
                [
                    float(row["mean_policy_regret"])
                    for row in summaries
                    if row["method"] == seed_method
                ]
            )
        )
        primary_seed_positive += int(seed_macro <= bias_regret)
    nonnegative_lambdas = sum(
        float(row["mean_policy_regret"]) <= bias_by_lambda[float(row["lambda"])] + 1e-12
        for row in summaries
        if row["method"] == primary
    )
    primary_harm = next(
        float(row["material_harm_rate"])
        for row in harm
        if row["method"] == primary and row["lambda"] == "macro"
    )
    mixture_ci = utility_bootstrap[(mixture_method, primary)]
    shuffle_vs_bias = utility_bootstrap[(BIAS_ONLY, SHUFFLED)]
    b4_utility_comparison = (
        utility_bootstrap[(FIXED_MODEL, primary)] if FIXED_MODEL in b4_choices else None
    )
    gates = {
        "macro_oracle_gap_closure_at_least_0.20": oracle_closure[primary] >= 0.20,
        "cost_matched_mixture_utility_ci95_positive": float(mixture_ci["ci95_low"])
        > 0.0,
        "at_least_4_of_5_train_seeds_nonworse_than_bias": primary_seed_positive >= 4,
        "at_least_7_of_10_lambdas_nonworse_than_bias": nonnegative_lambdas >= 7,
        "shuffle_has_no_positive_ci_vs_bias": float(shuffle_vs_bias["ci95_low"]) <= 0.0,
        "material_harm_rate_at_most_0.05": primary_harm <= 0.05,
    }
    artifacts = {
        "train_cv_history": "train_cv_history.csv.gz",
        "validation_results": "validation_results.csv",
        "validation_predictions": "validation_predictions.csv.gz",
        "validation_action_scores": "validation_action_scores.csv.gz",
        "validation_paired_bootstrap": "validation_paired_bootstrap.csv",
        "validation_macro_paired_bootstrap": "validation_macro_paired_bootstrap.csv",
        "validation_material_harm": "validation_material_harm.csv",
    }
    body = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "evaluation_stage": "selection",
        "evaluation_split": "validation",
        "test_accessed": False,
        "experiment_role": "structured_utility_aligned_prompt_prior_v2",
        "selection_protocol": "architecture_and_epoch_selected_by_train_only_5fold_cv_before_one_validation_pass",
        "dataset": {
            "path": str(dataset_path),
            "file_sha256": sha256_file(dataset_path),
            "dataset_sha256": dataset["dataset_sha256"],
        },
        "embedding_manifest_sha256": embedding_manifest["manifest_sha256"],
        "b4_control": b4_control,
        "latency_profile": latency,
        "action_ids": action_ids,
        "action_feature_metadata": action_feature_metadata,
        "lambdas": args.lambdas,
        "train_prompt_count": len(samples["train"]),
        "validation_prompt_count": len(samples["validation"]),
        "train_seeds": args.train_seeds,
        "architectures": args.architectures,
        "train_cv_macro_regret": architecture_cv_regret,
        "primary_architecture_selected_by_train_cv": primary,
        "selected_epochs": {
            name: {str(seed): epoch for seed, epoch in values.items()}
            for name, values in selected_epochs.items()
        },
        "shuffled_control_selected_epochs": {
            str(seed): epoch for seed, epoch in shuffled_epochs.items()
        },
        "runs": runs,
        "cost_matched_mixture": mixture_spec,
        "macro_policy_regret": macro_regret,
        "oracle_gap_closure": oracle_closure,
        "success_gate_details": {
            "primary_seed_nonworse_count": primary_seed_positive,
            "primary_lambda_nonworse_count": nonnegative_lambdas,
            "primary_macro_material_harm_rate": primary_harm,
            "cost_matched_mixture_utility_ci95_low": float(mixture_ci["ci95_low"]),
            "shuffle_utility_ci95_low_vs_bias": float(shuffle_vs_bias["ci95_low"]),
            "utility_vs_fixed_lambda_b4": b4_utility_comparison,
        },
        "success_gates": gates,
        "all_success_gates_passed": all(gates.values()),
        "artifacts": artifacts,
    }
    summary = {
        "schema": "univ_combined_v3_prompt_prior_v2_selection_v1",
        "selection_sha256": canonical_sha256(body),
        **body,
    }
    write_json_atomic(summary_path, summary)
    portable_files = [
        summary_path,
        out_root / "latency_profile.json",
        *[out_root / value for value in artifacts.values()],
    ]
    manifest_rows = []
    for path in portable_files:
        manifest_rows.append(
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    write_csv(out_root / "portable_manifest.csv", manifest_rows)
    archive_path = out_root / "prompt_prior_v2_portable_results.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for path in [*portable_files, out_root / "portable_manifest.csv"]:
            archive.add(path, arcname=path.name)
    print(
        json.dumps(
            {
                "primary_architecture": primary,
                "macro_policy_regret": macro_regret,
                "oracle_gap_closure": oracle_closure,
                "success_gates": gates,
                "portable_results": str(archive_path),
                "selection_summary": str(summary_path),
                "test_accessed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
