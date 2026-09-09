from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from UNIV_adaptor.combined_v3 import QUALITY_DIMENSIONS, load_json, verify_file  # noqa: E402
from UNIV_adaptor.data_protocol import (  # noqa: E402
    canonical_sha256,
    sha256_file,
    write_json_atomic,
)
from UNIV_adaptor.scripts.router.train_combined_v3_budget_prior import (  # noqa: E402
    RelativeQualityPrior,
    load_samples,
    train_latency_profile,
    validate_embedding_manifest,
    validate_merged_index,
    write_csv,
)


FIXED_MODEL = "b4_fixed_lambda_bank"
VARIABLE_MODEL = "b4_variable_lambda"
B4_METHODS = (FIXED_MODEL, VARIABLE_MODEL)


class B4SoftUtilityRouter(nn.Module):
    """Original B4 hidden backbone with an optional scalar lambda input."""

    def __init__(
        self,
        in_dim: int,
        action_count: int,
        *,
        lambda_conditioned: bool,
        hidden_dims: tuple[int, ...] = (256, 128),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.lambda_conditioned = bool(lambda_conditioned)
        previous = in_dim + int(self.lambda_conditioned)
        layers: list[nn.Module] = []
        for hidden in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous, hidden),
                    nn.LayerNorm(hidden),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                ]
            )
            previous = hidden
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(previous, action_count)

    def forward(
        self, prompt_embedding: torch.Tensor, lambda_feature: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.lambda_conditioned:
            if lambda_feature is None:
                raise ValueError("lambda-conditioned B4 requires lambda_feature")
            if lambda_feature.ndim == 1:
                lambda_feature = lambda_feature.unsqueeze(1)
            if lambda_feature.shape != (prompt_embedding.shape[0], 1):
                raise ValueError("lambda_feature must have shape [batch] or [batch, 1]")
            prompt_embedding = torch.cat(
                [prompt_embedding, lambda_feature.to(prompt_embedding.dtype)], dim=1
            )
        elif lambda_feature is not None:
            raise ValueError("fixed-lambda B4 does not accept lambda_feature")
        return self.head(self.mlp(prompt_embedding))


def normalize_lambda(
    value: torch.Tensor | float, *, lambda_min: float, lambda_max: float
) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if math.isclose(lambda_min, lambda_max, rel_tol=0.0, abs_tol=1e-12):
        return torch.zeros_like(tensor)
    return 2.0 * (tensor - lambda_min) / (lambda_max - lambda_min) - 1.0


def soft_utility_targets(
    quality: torch.Tensor,
    normalized_cost: torch.Tensor,
    lambda_value: torch.Tensor | float,
    *,
    temperature: float,
) -> torch.Tensor:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if quality.ndim != 2 or normalized_cost.ndim != 1:
        raise ValueError("quality must be [batch, action] and cost must be [action]")
    if quality.shape[1] != normalized_cost.numel():
        raise ValueError("quality and cost action counts differ")
    lam = torch.as_tensor(lambda_value, dtype=quality.dtype, device=quality.device)
    if lam.ndim == 0:
        lam = lam.expand(quality.shape[0])
    if lam.shape != (quality.shape[0],):
        raise ValueError("lambda_value must be scalar or [batch]")
    utility = quality - lam.unsqueeze(1) * normalized_cost.to(quality.device)
    return torch.softmax(
        (utility - utility.max(dim=1, keepdim=True).values) / temperature,
        dim=1,
    )


def b4_distillation_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    emd_weight: float,
    cost_order: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if emd_weight < 0.0:
        raise ValueError("emd_weight must be non-negative")
    if logits.shape != targets.shape or logits.ndim != 2:
        raise ValueError("logits and targets must share [batch, action] shape")
    if cost_order.shape != (logits.shape[1],):
        raise ValueError("cost_order must contain one index per action")
    log_probs = F.log_softmax(logits, dim=1)
    probabilities = torch.softmax(logits, dim=1)
    kl = F.kl_div(log_probs, targets, reduction="batchmean")
    ordered_probs = probabilities.index_select(1, cost_order.to(logits.device))
    ordered_targets = targets.index_select(1, cost_order.to(targets.device))
    emd = torch.mean(
        torch.abs(
            torch.cumsum(ordered_probs, dim=1) - torch.cumsum(ordered_targets, dim=1)
        )
    )
    return kl + emd_weight * emd, kl.detach(), emd.detach()


def sample_arrays(
    samples: list[dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, dict[str, np.ndarray]]:
    embeddings = torch.from_numpy(
        np.stack([sample["embedding"] for sample in samples])
    ).float()
    quality_np = np.stack([sample["candidate_quality"] for sample in samples])
    quality = torch.from_numpy(quality_np).float()
    dimensions = {
        name: np.stack([sample["dimensions"][name] for sample in samples])
        for name in QUALITY_DIMENSIONS
    }
    return embeddings, quality, quality_np, dimensions


@torch.no_grad()
def validation_metrics(
    model: B4SoftUtilityRouter,
    embeddings: torch.Tensor,
    quality: torch.Tensor,
    normalized_cost: torch.Tensor,
    lambdas: list[float],
    *,
    temperature: float,
    emd_weight: float,
    cost_order: torch.Tensor,
    lambda_conditioned: bool,
    lambda_min: float,
    lambda_max: float,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    regrets = []
    losses = []
    x = embeddings.to(device)
    q = quality.to(device)
    cost = normalized_cost.to(device)
    for lambda_value in lambdas:
        lambda_feature = None
        if lambda_conditioned:
            lambda_feature = normalize_lambda(
                torch.full((x.shape[0],), lambda_value),
                lambda_min=lambda_min,
                lambda_max=lambda_max,
            ).to(device)
        logits = model(x, lambda_feature)
        targets = soft_utility_targets(q, cost, lambda_value, temperature=temperature)
        loss, _, _ = b4_distillation_loss(
            logits, targets, emd_weight=emd_weight, cost_order=cost_order
        )
        utility = q - lambda_value * cost.unsqueeze(0)
        choices = logits.argmax(dim=1)
        realized = utility[torch.arange(choices.numel(), device=device), choices]
        regrets.append(float((utility.max(dim=1).values - realized).mean().cpu()))
        losses.append(float(loss.cpu()))
    return float(np.mean(regrets)), float(np.mean(losses))


def training_tensors(
    embeddings: torch.Tensor,
    quality: torch.Tensor,
    normalized_cost: torch.Tensor,
    lambdas: list[float],
    *,
    temperature: float,
    lambda_conditioned: bool,
    lambda_min: float,
    lambda_max: float,
) -> TensorDataset:
    if lambda_conditioned:
        return TensorDataset(embeddings, quality)
    if len(lambdas) != 1:
        raise ValueError("fixed-lambda training requires exactly one lambda")
    targets = soft_utility_targets(
        quality, normalized_cost, lambdas[0], temperature=temperature
    )
    return TensorDataset(embeddings, targets)


def expand_variable_lambda_batch(
    embeddings: torch.Tensor,
    quality: torch.Tensor,
    normalized_cost: torch.Tensor,
    lambdas: list[float],
    *,
    temperature: float,
    lambda_min: float,
    lambda_max: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    repeated_embeddings = embeddings.repeat_interleave(len(lambdas), dim=0)
    repeated_quality = quality.repeat_interleave(len(lambdas), dim=0)
    lambda_values = torch.tensor(
        lambdas, dtype=quality.dtype, device=quality.device
    ).repeat(embeddings.shape[0])
    targets = soft_utility_targets(
        repeated_quality,
        normalized_cost,
        lambda_values,
        temperature=temperature,
    )
    features = normalize_lambda(
        lambda_values, lambda_min=lambda_min, lambda_max=lambda_max
    ).to(quality.device)
    return repeated_embeddings, features, targets


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_slug(model_type: str, seed: int, lambda_value: float | None) -> str:
    if model_type == VARIABLE_MODEL:
        return f"{VARIABLE_MODEL}/seed_{seed}"
    if lambda_value is None:
        raise ValueError("fixed-lambda run requires lambda_value")
    return f"{FIXED_MODEL}/lambda_{lambda_value:.2f}/seed_{seed}"


def train_one_run(
    *,
    model_type: str,
    lambda_value: float | None,
    seed: int,
    train_embeddings: torch.Tensor,
    train_quality: torch.Tensor,
    validation_embeddings: torch.Tensor,
    validation_quality: torch.Tensor,
    normalized_cost: torch.Tensor,
    lambdas: list[float],
    action_ids: list[str],
    args: argparse.Namespace,
    device: torch.device,
    cost_order: torch.Tensor,
    provenance: dict[str, Any],
    out_root: Path,
) -> dict[str, Any]:
    lambda_conditioned = model_type == VARIABLE_MODEL
    run_lambdas = lambdas if lambda_conditioned else [float(lambda_value)]
    lambda_min, lambda_max = min(lambdas), max(lambdas)
    run_dir = out_root / run_slug(model_type, seed, lambda_value)
    summary_path = run_dir / "run_summary.json"
    training_config = {
        "model_type": model_type,
        "lambda": lambda_value,
        "lambdas": run_lambdas,
        "input_dim": 4096 + int(lambda_conditioned),
        "hidden_dims": [256, 128],
        "normalization": "layer_norm_after_each_hidden_linear",
        "dropout": args.dropout,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "optimizer": "AdamW",
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "schedule": "cosine_to_1e-5",
        "gradient_clip_norm": 1.0,
        "soft_target_temperature": args.soft_target_tau,
        "loss": "KL_target_to_model_plus_cost_ordered_wasserstein",
        "emd_weight": args.emd_weight,
        "checkpoint_selection": "validation_policy_regret_then_distillation_loss",
    }
    if summary_path.is_file():
        summary = load_json(summary_path)
        body = {
            key: value
            for key, value in summary.items()
            if key not in {"schema", "run_sha256"}
        }
        if (
            summary.get("schema") != "univ_combined_v3_b4_seed_run_v1"
            or canonical_sha256(body) != summary.get("run_sha256")
            or summary.get("provenance") != provenance
            or summary.get("training_config") != training_config
        ):
            raise RuntimeError(f"incompatible completed B4 run: {summary_path}")
        result = summary["result"]
        verify_file(
            result["checkpoint"], result["checkpoint_sha256"], label="B4 checkpoint"
        )
        print(f"Reusing completed B4 run: {run_dir}")
        return result
    if run_dir.exists():
        raise RuntimeError(
            f"incomplete B4 run exists; use a new output root: {run_dir}"
        )

    seed_everything(seed)
    model = B4SoftUtilityRouter(
        4096,
        len(action_ids),
        lambda_conditioned=lambda_conditioned,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )
    train_data = training_tensors(
        train_embeddings,
        train_quality,
        normalized_cost,
        run_lambdas,
        temperature=args.soft_target_tau,
        lambda_conditioned=lambda_conditioned,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
    )
    loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=torch.Generator().manual_seed(seed),
    )
    best_key = (float("inf"), float("inf"))
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_kl = 0.0
        total_emd = 0.0
        count = 0
        for batch in loader:
            if lambda_conditioned:
                prompt_embedding, batch_quality = batch
                prompt_count = int(prompt_embedding.shape[0])
                embedding, lambda_feature, targets = expand_variable_lambda_batch(
                    prompt_embedding.to(device),
                    batch_quality.to(device),
                    normalized_cost.to(device),
                    run_lambdas,
                    temperature=args.soft_target_tau,
                    lambda_min=lambda_min,
                    lambda_max=lambda_max,
                )
            else:
                embedding, targets = batch
                lambda_feature = None
                prompt_count = int(embedding.shape[0])
                embedding = embedding.to(device)
                targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(embedding, lambda_feature)
            loss, kl, emd = b4_distillation_loss(
                logits,
                targets,
                emd_weight=args.emd_weight,
                cost_order=cost_order,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite B4 loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * prompt_count
            total_kl += float(kl) * prompt_count
            total_emd += float(emd) * prompt_count
            count += prompt_count
        scheduler.step()
        regret, val_loss = validation_metrics(
            model,
            validation_embeddings,
            validation_quality,
            normalized_cost,
            run_lambdas,
            temperature=args.soft_target_tau,
            emd_weight=args.emd_weight,
            cost_order=cost_order,
            lambda_conditioned=lambda_conditioned,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            device=device,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(count, 1),
                "train_kl": total_kl / max(count, 1),
                "train_emd": total_emd / max(count, 1),
                "validation_policy_regret": regret,
                "validation_distillation_loss": val_loss,
                "learning_rate": scheduler.get_last_lr()[0],
            }
        )
        key = (regret, val_loss)
        if key < best_key:
            best_key = key
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("no B4 validation checkpoint was selected")
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = {
        "schema": "univ_combined_v3_b4_checkpoint_v1",
        "model_type": model_type,
        "lambda": lambda_value,
        "lambdas": run_lambdas,
        "lambda_min": lambda_min,
        "lambda_max": lambda_max,
        "lambda_conditioned": lambda_conditioned,
        "state_dict": best_state,
        "input_dim": 4096,
        "hidden_dims": [256, 128],
        "dropout": args.dropout,
        "action_ids": action_ids,
        "normalized_action_cost": normalized_cost.tolist(),
        "cost_order": cost_order.tolist(),
        "best_epoch": best_epoch,
        "train_seed": seed,
        "provenance": provenance,
        "training_config": training_config,
    }
    checkpoint_path = run_dir / "b4_router.pt"
    torch.save(checkpoint, checkpoint_path)
    write_csv(run_dir / "training_history.csv", history)
    result = {
        "model_type": model_type,
        "lambda": lambda_value,
        "train_seed": seed,
        "best_epoch": best_epoch,
        "validation_policy_regret": best_key[0],
        "validation_distillation_loss": best_key[1],
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }
    body = {
        "provenance": provenance,
        "training_config": training_config,
        "result": result,
    }
    write_json_atomic(
        summary_path,
        {
            "schema": "univ_combined_v3_b4_seed_run_v1",
            "run_sha256": canonical_sha256(body),
            **body,
        },
    )
    return result


def load_b4_model(
    path: Path, device: torch.device
) -> tuple[B4SoftUtilityRouter, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "univ_combined_v3_b4_checkpoint_v1":
        raise ValueError(f"unsupported B4 checkpoint: {path}")
    model = B4SoftUtilityRouter(
        int(payload["input_dim"]),
        len(payload["action_ids"]),
        lambda_conditioned=bool(payload["lambda_conditioned"]),
        hidden_dims=tuple(int(value) for value in payload["hidden_dims"]),
        dropout=float(payload["dropout"]),
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device).eval(), payload


@torch.no_grad()
def b4_ensemble_probabilities(
    checkpoints: list[Path],
    embeddings: torch.Tensor,
    lambda_value: float,
    *,
    device: torch.device,
) -> tuple[np.ndarray, list[tuple[int, np.ndarray]]]:
    seed_probabilities = []
    for checkpoint in checkpoints:
        model, payload = load_b4_model(checkpoint, device)
        lambda_feature = None
        if payload["lambda_conditioned"]:
            lambda_feature = normalize_lambda(
                torch.full((embeddings.shape[0],), lambda_value),
                lambda_min=float(payload["lambda_min"]),
                lambda_max=float(payload["lambda_max"]),
            ).to(device)
        logits = model(embeddings.to(device), lambda_feature)
        seed_probabilities.append(
            (
                int(payload["train_seed"]),
                torch.softmax(logits, dim=1).cpu().numpy(),
            )
        )
    return (
        np.mean([probabilities for _, probabilities in seed_probabilities], axis=0),
        seed_probabilities,
    )


@torch.no_grad()
def load_quality_curve_predictions(
    quality_curve_root: Path | None,
    *,
    dataset_sha256: str,
    embedding_manifest_sha256: str,
    latency_profile_sha256: str,
    action_ids: list[str],
    embeddings: torch.Tensor,
    device: torch.device,
) -> dict[str, np.ndarray]:
    if quality_curve_root is None:
        return {}
    summary_path = quality_curve_root / "selection_summary.json"
    summary = load_json(summary_path)
    if (
        summary.get("schema") != "univ_prompt_action_quality_prior_selection_v1"
        or summary.get("test_accessed") is not False
        or summary.get("dataset", {}).get("dataset_sha256") != dataset_sha256
        or summary.get("embedding_manifest", {}).get("manifest_sha256")
        != embedding_manifest_sha256
        or summary.get("latency_profile", {}).get("profile_sha256")
        != latency_profile_sha256
        or summary.get("action_ids") != action_ids
    ):
        raise ValueError("quality-curve root is incompatible with this B4 control")

    def predict(checkpoint: Path) -> np.ndarray:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("action_ids") != action_ids:
            raise ValueError(f"quality-curve action ids differ: {checkpoint}")
        model = RelativeQualityPrior(
            4096, len(action_ids), dropout=float(payload["dropout"])
        )
        model.load_state_dict(payload["state_dict"], strict=True)
        return model.to(device).eval()(embeddings.to(device)).detach().cpu().numpy()

    selected_meta = summary["selected_checkpoint"]
    selected_path = verify_file(
        selected_meta["path"], selected_meta["sha256"], label="selected quality curve"
    )
    selected = predict(selected_path)
    seed_predictions = []
    for run in summary["runs"]:
        path = verify_file(
            run["checkpoint"], run["checkpoint_sha256"], label="quality curve seed"
        )
        seed_predictions.append(predict(path))
    return {
        "quality_curve_selected": selected,
        "quality_curve_ensemble": np.mean(seed_predictions, axis=0),
    }


def evaluate_methods(
    *,
    train_samples: list[dict[str, Any]],
    validation_samples: list[dict[str, Any]],
    action_ids: list[str],
    normalized_cost: np.ndarray,
    lambdas: list[float],
    fixed_checkpoints: dict[float, list[Path]],
    variable_checkpoints: list[Path],
    quality_curve_predictions: dict[str, np.ndarray],
    soft_target_tau: float,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    train_quality = np.stack([row["candidate_quality"] for row in train_samples])
    val_embeddings, _, val_quality, dimensions = sample_arrays(validation_samples)
    summaries = []
    rows = []
    score_rows = []
    for lambda_value in lambdas:
        true_utility = val_quality - lambda_value * normalized_cost[None, :]
        oracle_choices = true_utility.argmax(axis=1)
        train_utility = train_quality - lambda_value * normalized_cost[None, :]
        fixed_index = int(train_utility.mean(axis=0).argmax())
        fixed_choices = np.full(len(validation_samples), fixed_index, dtype=np.int64)
        fixed_probs, fixed_seed_probs = b4_ensemble_probabilities(
            fixed_checkpoints[lambda_value],
            val_embeddings,
            lambda_value,
            device=device,
        )
        variable_probs, variable_seed_probs = b4_ensemble_probabilities(
            variable_checkpoints, val_embeddings, lambda_value, device=device
        )
        methods: dict[str, tuple[np.ndarray, np.ndarray | None]] = {
            "prompt_oracle_upper_bound": (oracle_choices, None),
            "train_selected_fixed": (fixed_choices, None),
            FIXED_MODEL: (fixed_probs.argmax(axis=1), fixed_probs),
            VARIABLE_MODEL: (variable_probs.argmax(axis=1), variable_probs),
        }
        for name, prediction in quality_curve_predictions.items():
            methods[name] = (
                (prediction - lambda_value * normalized_cost[None, :]).argmax(axis=1),
                None,
            )
        oracle_utility = true_utility[
            np.arange(len(validation_samples)), oracle_choices
        ]
        for method, (choices, probabilities) in methods.items():
            realized_utility = true_utility[np.arange(len(choices)), choices]
            realized_quality = val_quality[np.arange(len(choices)), choices]
            summary: dict[str, Any] = {
                "lambda": lambda_value,
                "method": method,
                "mean_policy_regret": float(np.mean(oracle_utility - realized_utility)),
                "mean_realized_utility": float(np.mean(realized_utility)),
                "mean_realized_vbench5": float(np.mean(realized_quality)),
                "mean_normalized_cost": float(np.mean(normalized_cost[choices])),
                "fixed_action_id": action_ids[fixed_index]
                if method == "train_selected_fixed"
                else "",
                "oracle_exact_action_rate": float(np.mean(choices == oracle_choices)),
            }
            for dimension, values in dimensions.items():
                summary[f"mean_{dimension}"] = float(
                    np.mean(values[np.arange(len(choices)), choices])
                )
            summaries.append(summary)
            for index, sample in enumerate(validation_samples):
                choice = int(choices[index])
                row: dict[str, Any] = {
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
                if probabilities is not None:
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
                                "true_utility": float(
                                    true_utility[index, action_index]
                                ),
                                "soft_utility_target": float(target[action_index]),
                                "predicted_probability": float(
                                    probabilities[index, action_index]
                                ),
                            }
                        )
        for method, seed_probs in (
            (FIXED_MODEL, fixed_seed_probs),
            (VARIABLE_MODEL, variable_seed_probs),
        ):
            for train_seed, probabilities in seed_probs:
                choices = probabilities.argmax(axis=1)
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
                        "fixed_action_id": "",
                        "oracle_exact_action_rate": float(
                            np.mean(choices == oracle_choices)
                        ),
                        **{
                            f"mean_{dimension}": float(
                                np.mean(values[np.arange(len(choices)), choices])
                            )
                            for dimension, values in dimensions.items()
                        },
                    }
                )
    return summaries, rows, score_rows


def metric_delta(candidate: float, reference: float, metric: str) -> float:
    if metric in {"policy_regret", "normalized_cost"}:
        return reference - candidate
    return candidate - reference


def paired_bootstrap(
    rows: list[dict[str, Any]],
    *,
    reference_method: str,
    candidate_methods: list[str],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metrics = (
        "policy_regret",
        "realized_utility",
        "realized_vbench5",
        "normalized_cost",
    )
    rng = np.random.default_rng(bootstrap_seed)
    per_lambda = []
    macro = []
    lambdas = sorted({float(row["lambda"]) for row in rows})
    index = {
        (str(row["method"]), float(row["lambda"]), int(row["global_prompt_id"])): row
        for row in rows
    }
    prompt_ids = sorted({int(row["global_prompt_id"]) for row in rows})
    for candidate_method in candidate_methods:
        for metric in metrics:
            macro_by_prompt: dict[int, list[float]] = defaultdict(list)
            for lambda_value in lambdas:
                values = []
                for prompt_id in prompt_ids:
                    candidate = index[(candidate_method, lambda_value, prompt_id)]
                    reference = index[(reference_method, lambda_value, prompt_id)]
                    delta = metric_delta(
                        float(candidate[metric]), float(reference[metric]), metric
                    )
                    values.append(delta)
                    macro_by_prompt[prompt_id].append(delta)
                array = np.asarray(values, dtype=np.float64)
                draws = array[
                    rng.integers(
                        0,
                        array.size,
                        size=(bootstrap_samples, array.size),
                    )
                ].mean(axis=1)
                low, high = np.quantile(draws, [0.025, 0.975])
                per_lambda.append(
                    {
                        "reference_method": reference_method,
                        "candidate_method": candidate_method,
                        "lambda": lambda_value,
                        "metric": metric,
                        "orientation": "positive_means_candidate_better",
                        "improvement_mean": float(array.mean()),
                        "ci95_low": float(low),
                        "ci95_high": float(high),
                        "prompt_count": len(prompt_ids),
                        "bootstrap_samples": bootstrap_samples,
                        "bootstrap_seed": bootstrap_seed,
                    }
                )
            array = np.asarray(
                [np.mean(macro_by_prompt[prompt_id]) for prompt_id in prompt_ids],
                dtype=np.float64,
            )
            draws = array[
                rng.integers(
                    0,
                    array.size,
                    size=(bootstrap_samples, array.size),
                )
            ].mean(axis=1)
            low, high = np.quantile(draws, [0.025, 0.975])
            macro.append(
                {
                    "reference_method": reference_method,
                    "candidate_method": candidate_method,
                    "metric": metric,
                    "orientation": "positive_means_candidate_better",
                    "macro_improvement_mean": float(array.mean()),
                    "ci95_low": float(low),
                    "ci95_high": float(high),
                    "prompt_count": len(prompt_ids),
                    "lambda_count": len(lambdas),
                    "bootstrap_samples": bootstrap_samples,
                    "bootstrap_seed": bootstrap_seed,
                }
            )
    return per_lambda, macro


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--quality-curve-root", default=None)
    parser.add_argument("--train-seeds", nargs="+", type=int, default=[42, 100, 2024])
    parser.add_argument(
        "--lambdas",
        nargs="+",
        type=float,
        default=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10],
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--soft-target-tau", type=float, default=0.02)
    parser.add_argument("--emd-weight", type=float, default=0.5)
    parser.add_argument("--hardware-label", default="unspecified_generation_device")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=2027)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        parser.error("epochs/batch-size must be positive and num-workers non-negative")
    if args.lr <= 0.0 or args.weight_decay < 0.0 or not 0.0 <= args.dropout < 1.0:
        parser.error("invalid optimizer or dropout configuration")
    if args.soft_target_tau <= 0.0 or args.emd_weight < 0.0:
        parser.error("soft-target-tau must be positive and emd-weight non-negative")
    if len(args.train_seeds) != len(set(args.train_seeds)) or len(args.train_seeds) < 3:
        parser.error("train-seeds must contain at least three unique values")
    if args.bootstrap_samples < 1 or not args.hardware_label.strip():
        parser.error("invalid bootstrap-samples or hardware-label")
    return args


def main() -> None:
    args = parse_args()
    lambdas = sorted(set(float(value) for value in args.lambdas))
    if not lambdas or any(value < 0.0 for value in lambdas):
        raise ValueError("lambdas must be non-empty and non-negative")
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
    normalized_cost = torch.from_numpy(normalized_cost_np).float()
    cost_order = torch.argsort(normalized_cost)
    train_embeddings, train_quality, _, _ = sample_arrays(samples["train"])
    validation_embeddings, validation_quality, _, _ = sample_arrays(
        samples["validation"]
    )
    out_root = Path(args.out_root).resolve()
    final_path = out_root / "selection_summary.json"
    if final_path.exists():
        raise FileExistsError(f"refusing to overwrite B4 selection: {final_path}")
    out_root.mkdir(parents=True, exist_ok=True)
    latency_path = out_root / "latency_profile.json"
    if latency_path.is_file():
        if load_json(latency_path).get("profile_sha256") != latency["profile_sha256"]:
            raise RuntimeError("output root contains another latency profile")
    else:
        write_json_atomic(latency_path, latency)
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
        "evaluation_stage": "validation_only_selection",
        "test_accessed": False,
        "trainer_sha256": sha256_file(Path(__file__).resolve()),
        "combined_v3_loader_sha256": sha256_file(loader_path),
    }
    runs = []
    fixed_paths: dict[float, list[Path]] = {value: [] for value in lambdas}
    variable_paths = []
    for lambda_value in lambdas:
        for seed in args.train_seeds:
            result = train_one_run(
                model_type=FIXED_MODEL,
                lambda_value=lambda_value,
                seed=seed,
                train_embeddings=train_embeddings,
                train_quality=train_quality,
                validation_embeddings=validation_embeddings,
                validation_quality=validation_quality,
                normalized_cost=normalized_cost,
                lambdas=lambdas,
                action_ids=action_ids,
                args=args,
                device=device,
                cost_order=cost_order,
                provenance=provenance,
                out_root=out_root,
            )
            runs.append(result)
            fixed_paths[lambda_value].append(Path(result["checkpoint"]))
    for seed in args.train_seeds:
        result = train_one_run(
            model_type=VARIABLE_MODEL,
            lambda_value=None,
            seed=seed,
            train_embeddings=train_embeddings,
            train_quality=train_quality,
            validation_embeddings=validation_embeddings,
            validation_quality=validation_quality,
            normalized_cost=normalized_cost,
            lambdas=lambdas,
            action_ids=action_ids,
            args=args,
            device=device,
            cost_order=cost_order,
            provenance=provenance,
            out_root=out_root,
        )
        runs.append(result)
        variable_paths.append(Path(result["checkpoint"]))
    quality_curve_root = (
        Path(args.quality_curve_root).resolve() if args.quality_curve_root else None
    )
    quality_predictions = load_quality_curve_predictions(
        quality_curve_root,
        dataset_sha256=dataset["dataset_sha256"],
        embedding_manifest_sha256=embedding_manifest["manifest_sha256"],
        latency_profile_sha256=latency["profile_sha256"],
        action_ids=action_ids,
        embeddings=validation_embeddings,
        device=device,
    )
    summaries, rows, score_rows = evaluate_methods(
        train_samples=samples["train"],
        validation_samples=samples["validation"],
        action_ids=action_ids,
        normalized_cost=normalized_cost_np,
        lambdas=lambdas,
        fixed_checkpoints=fixed_paths,
        variable_checkpoints=variable_paths,
        quality_curve_predictions=quality_predictions,
        soft_target_tau=args.soft_target_tau,
        device=device,
    )
    ensemble_methods = [*B4_METHODS, *sorted(quality_predictions)]
    per_lambda, macro = paired_bootstrap(
        rows,
        reference_method="train_selected_fixed",
        candidate_methods=ensemble_methods,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    curve_reference = "quality_curve_ensemble"
    if curve_reference in quality_predictions:
        against_curve, macro_against_curve = paired_bootstrap(
            rows,
            reference_method=curve_reference,
            candidate_methods=list(B4_METHODS),
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed + 1,
        )
        per_lambda.extend(against_curve)
        macro.extend(macro_against_curve)
    write_csv(out_root / "validation_results.csv", summaries)
    write_csv(out_root / "validation_predictions.csv", rows)
    write_csv(out_root / "validation_action_scores.csv", score_rows)
    write_csv(out_root / "validation_paired_bootstrap.csv", per_lambda)
    write_csv(out_root / "validation_macro_paired_bootstrap.csv", macro)
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
        for method in ["train_selected_fixed", *ensemble_methods]
    }
    best_b4 = min(B4_METHODS, key=lambda method: (macro_regret[method], method))
    body = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "evaluation_stage": "selection",
        "evaluation_split": "validation",
        "test_accessed": False,
        "experiment_role": "true_b4_soft_utility_control",
        "model_roles": {
            FIXED_MODEL: "original_style_per_lambda_control_bank",
            VARIABLE_MODEL: "shared_variable_lambda_controller_candidate",
        },
        "selection_caveat": (
            "checkpoints and the comparison between B4 variants are selected on "
            "validation; only a later frozen test run may provide confirmation"
        ),
        "dataset": {
            "path": str(dataset_path),
            "file_sha256": sha256_file(dataset_path),
            "dataset_sha256": dataset["dataset_sha256"],
        },
        "embedding_manifest_sha256": embedding_manifest["manifest_sha256"],
        "latency_profile": latency,
        "action_ids": action_ids,
        "emd_action_order": [action_ids[int(index)] for index in cost_order],
        "lambdas": lambdas,
        "train_prompt_count": len(samples["train"]),
        "validation_prompt_count": len(samples["validation"]),
        "train_seeds": args.train_seeds,
        "soft_target_tau": args.soft_target_tau,
        "emd_weight": args.emd_weight,
        "runs": runs,
        "quality_curve_root": str(quality_curve_root) if quality_curve_root else None,
        "macro_policy_regret": macro_regret,
        "best_b4_by_validation_macro_regret": best_b4,
        "artifacts": {
            "validation_results": "validation_results.csv",
            "validation_predictions": "validation_predictions.csv",
            "validation_action_scores": "validation_action_scores.csv",
            "validation_paired_bootstrap": "validation_paired_bootstrap.csv",
            "validation_macro_paired_bootstrap": "validation_macro_paired_bootstrap.csv",
        },
    }
    summary = {
        "schema": "univ_combined_v3_b4_control_selection_v1",
        "selection_sha256": canonical_sha256(body),
        **body,
    }
    write_json_atomic(final_path, summary)
    print(
        json.dumps(
            {
                "best_b4_by_validation_macro_regret": best_b4,
                "macro_policy_regret": macro_regret,
                "selection_summary": str(final_path),
                "test_accessed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
