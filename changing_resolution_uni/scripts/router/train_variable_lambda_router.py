#!/usr/bin/env python3
"""Train prompt-only and latent-state sequential routers across utility lambdas."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


DATASET_SCHEMA = "variable_lambda_online_state_dataset_v1"
MODEL_LABELS = {
    "prompt_only": "Variable-Lambda Prompt Router",
    "prompt_state": "Variable-Lambda Prompt+State Router",
}
SCHEDULE_FEATURE_NAMES = [
    "step_fraction",
    "candidate_fraction",
    "sigma",
    "log_snr",
    "lambda",
    "normalized_cost_profile",
    "lambda_cost_profile",
    "remaining_lr_fraction",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--model-type",
        choices=["prompt_only", "prompt_state", "both"],
        default="both",
    )
    parser.add_argument(
        "--feature-groups",
        nargs="+",
        default=[
            "x0_global",
            "residual_global",
            "x0_channel",
            "residual_channel",
            "local_energy",
            "trajectory_delta",
        ],
    )
    parser.add_argument(
        "--train-lambdas",
        type=float,
        nargs="+",
        default=[0.01, 0.02, 0.04, 0.06, 0.08, 0.10],
    )
    parser.add_argument(
        "--eval-lambdas",
        type=float,
        nargs="+",
        default=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10],
    )
    parser.add_argument("--primary-lambda", type=float, default=0.08)
    parser.add_argument("--harm-epsilon", type=float, default=0.001)
    parser.add_argument("--risk-threshold", type=float, default=0.5)
    parser.add_argument("--regret-scale", type=float, default=100.0)
    parser.add_argument("--regret-loss-weight", type=float, default=1.0)
    parser.add_argument("--harm-loss-weight", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-batch-trajectories", type=int, default=64)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    args.train_lambdas = sorted(set(float(value) for value in args.train_lambdas))
    args.eval_lambdas = sorted(set(float(value) for value in args.eval_lambdas))
    for name in ("train_lambdas", "eval_lambdas"):
        values = getattr(args, name)
        if not values or any(value < 0 or not math.isfinite(value) for value in values):
            parser.error(
                f"{name.replace('_', '-')} must contain finite non-negative values"
            )
    if args.primary_lambda not in args.eval_lambdas:
        parser.error("primary-lambda must be present in eval-lambdas")
    if args.harm_epsilon < 0 or args.regret_scale <= 0:
        parser.error("harm-epsilon must be non-negative and regret-scale positive")
    if not 0 <= args.risk_threshold <= 1:
        parser.error("risk-threshold must be in [0, 1]")
    if args.regret_loss_weight < 0 or args.harm_loss_weight < 0:
        parser.error("loss weights must be non-negative")
    if (
        args.epochs < 1
        or args.batch_size < 1
        or args.num_workers < 0
        or args.eval_batch_trajectories < 1
    ):
        parser.error(
            "epochs/batch-size/eval-batch-trajectories must be positive and "
            "num-workers non-negative"
        )
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return rows


def load_dataset_manifest(dataset_dir: Path) -> dict[str, Any]:
    path = dataset_dir / "dataset_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != DATASET_SCHEMA
        or manifest.get("is_complete") is not True
    ):
        raise ValueError(f"Invalid variable-lambda dataset manifest: {path}")
    if manifest.get("test_accessed"):
        raise ValueError(
            "Selection training refuses a state dataset that accessed test"
        )
    if set(manifest.get("selected_splits", [])) != {"train", "validation"}:
        raise ValueError(
            "Selection requires exactly train and validation state features"
        )
    return manifest


def load_pooled_t5(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        pooled = np.asarray(payload["pooled_embedding"], dtype=np.float32)
    if pooled.shape != (4096,) or not np.isfinite(pooled).all():
        raise ValueError(f"Invalid pooled T5 embedding: {path}, shape={pooled.shape}")
    return pooled


def load_trajectories(
    dataset_dir: Path,
    manifest: dict[str, Any],
    split: str,
    selected_feature_indices: np.ndarray,
) -> list[dict[str, Any]]:
    split_meta = manifest["splits"][split]
    index_path = dataset_dir / split_meta["index_file"]
    if sha256_file(index_path) != split_meta["index_sha256"]:
        raise ValueError(f"Trajectory index SHA256 mismatch: {index_path}")
    rows = read_jsonl(index_path)
    prompt_cache: dict[int, np.ndarray] = {}
    trajectories = []
    candidate_steps = np.asarray(manifest["candidate_steps"], dtype=np.int64)
    for row in rows:
        prompt_id = int(row["prompt_id"])
        feature_path = dataset_dir / row["feature_file"]
        with np.load(feature_path, allow_pickle=False) as payload:
            observed_steps = np.asarray(payload["candidate_steps"], dtype=np.int64)
            features = np.asarray(payload["features"], dtype=np.float32)
            sigmas = np.asarray(payload["sigmas"], dtype=np.float32)
            qualities = np.asarray(payload["qualities"], dtype=np.float32)
            latencies = np.asarray(payload["latencies"], dtype=np.float32)
            native_latency = float(payload["native_latency"])
            dimensions = np.asarray(payload["dimensions"], dtype=np.float32)
        candidate_count = len(candidate_steps)
        if not np.array_equal(observed_steps, candidate_steps):
            raise ValueError(f"Candidate steps mismatch: {feature_path}")
        expected_shapes = {
            "features": (candidate_count, int(manifest["feature_count"])),
            "sigmas": (candidate_count,),
            "qualities": (candidate_count,),
            "latencies": (candidate_count,),
            "dimensions": (candidate_count, len(manifest["quality_dimensions"])),
        }
        observed_shapes = {
            "features": features.shape,
            "sigmas": sigmas.shape,
            "qualities": qualities.shape,
            "latencies": latencies.shape,
            "dimensions": dimensions.shape,
        }
        if observed_shapes != expected_shapes:
            raise ValueError(
                f"State array shapes mismatch in {feature_path}: "
                f"expected={expected_shapes}, observed={observed_shapes}"
            )
        if native_latency <= 0 or not all(
            np.isfinite(value).all()
            for value in (features, sigmas, qualities, latencies, dimensions)
        ):
            raise ValueError(f"Non-finite or invalid state arrays: {feature_path}")
        if prompt_id not in prompt_cache:
            prompt_cache[prompt_id] = load_pooled_t5(Path(row["t5_embedding_path"]))
        trajectories.append(
            {
                "split": split,
                "prompt_id": prompt_id,
                "seed": int(row["seed"]),
                "pooled_t5": prompt_cache[prompt_id],
                "features": features[:, selected_feature_indices],
                "sigmas": sigmas,
                "qualities": qualities,
                "costs": latencies / native_latency,
                "latencies": latencies,
                "native_latency": native_latency,
                "dimensions": dimensions,
            }
        )
    expected_count = int(split_meta["trajectory_count"])
    if len(trajectories) != expected_count:
        raise ValueError(
            f"{split}: loaded {len(trajectories)}, expected {expected_count}"
        )
    return trajectories


def select_feature_indices(
    manifest: dict[str, Any], requested_groups: list[str]
) -> tuple[np.ndarray, list[str]]:
    groups = manifest["feature_groups"]
    missing = [group for group in requested_groups if group not in groups]
    if missing:
        raise ValueError(
            f"Unknown feature groups {missing}; available={sorted(groups)}"
        )
    indices = sorted(
        {int(index) for group in requested_groups for index in groups[group]}
    )
    if not indices:
        raise ValueError("Feature group selection is empty")
    names = [manifest["feature_names"][index] for index in indices]
    return np.asarray(indices, dtype=np.int64), names


def fit_state_normalizer(
    trajectories: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.concatenate(
        [trajectory["features"] for trajectory in trajectories], axis=0
    )
    mean = matrix.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = matrix.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-6)
    return mean, std


def fit_cost_profile(trajectories: list[dict[str, Any]]) -> np.ndarray:
    costs = np.stack([trajectory["costs"] for trajectory in trajectories])
    return costs.mean(axis=0, dtype=np.float64).astype(np.float32)


def true_stop_regret(
    qualities: np.ndarray,
    costs: np.ndarray,
    lambda_value: float,
) -> np.ndarray:
    utilities = qualities - float(lambda_value) * costs
    future_best = np.maximum.accumulate(utilities[::-1])[::-1]
    return np.maximum(future_best - utilities, 0.0).astype(np.float32)


def schedule_features(
    candidate_steps: np.ndarray,
    sigmas: np.ndarray,
    lambda_value: float,
    cost_profile: np.ndarray,
) -> np.ndarray:
    sigma = np.clip(sigmas.astype(np.float64), 1e-6, 1.0 - 1e-6)
    log_snr = 2.0 * np.log1p(-sigma) - 2.0 * np.log(sigma)
    count = len(candidate_steps)
    result = np.stack(
        [
            candidate_steps / 50.0,
            np.arange(count, dtype=np.float64) / max(count - 1, 1),
            sigma,
            np.clip(log_snr, -20.0, 20.0),
            np.full(count, lambda_value, dtype=np.float64),
            cost_profile,
            float(lambda_value) * cost_profile,
            (50.0 - candidate_steps) / 50.0,
        ],
        axis=1,
    )
    return result.astype(np.float32)


class LambdaStateDataset(Dataset):
    def __init__(
        self,
        trajectories: list[dict[str, Any]],
        lambdas: list[float],
        candidate_steps: np.ndarray,
        state_mean: np.ndarray,
        state_std: np.ndarray,
        cost_profile: np.ndarray,
        harm_epsilon: float,
        regret_scale: float,
    ):
        self.trajectories = trajectories
        self.lambdas = lambdas
        self.candidate_steps = candidate_steps
        self.state_mean = state_mean
        self.state_std = state_std
        self.cost_profile = cost_profile
        self.harm_epsilon = harm_epsilon
        self.regret_scale = regret_scale
        self.candidate_count = len(candidate_steps)

    def __len__(self) -> int:
        return len(self.trajectories) * len(self.lambdas) * self.candidate_count

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        state_index = index % self.candidate_count
        pair_index = index // self.candidate_count
        lambda_index = pair_index % len(self.lambdas)
        trajectory_index = pair_index // len(self.lambdas)
        trajectory = self.trajectories[trajectory_index]
        lambda_value = self.lambdas[lambda_index]
        regrets = true_stop_regret(
            trajectory["qualities"], trajectory["costs"], lambda_value
        )
        schedule = schedule_features(
            self.candidate_steps,
            trajectory["sigmas"],
            lambda_value,
            self.cost_profile,
        )
        state = (trajectory["features"][state_index] - self.state_mean) / self.state_std
        regret = float(regrets[state_index])
        return {
            "pooled_t5": torch.from_numpy(trajectory["pooled_t5"]),
            "state": torch.from_numpy(state.astype(np.float32)),
            "schedule": torch.from_numpy(schedule[state_index]),
            "regret_target": torch.tensor(
                regret * self.regret_scale, dtype=torch.float32
            ),
            "harm_target": torch.tensor(
                float(regret > self.harm_epsilon), dtype=torch.float32
            ),
        }


class VariableLambdaRouter(nn.Module):
    def __init__(self, state_dim: int, use_state: bool, dropout: float):
        super().__init__()
        self.use_state = use_state
        self.prompt_encoder = nn.Sequential(
            nn.Linear(4096, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
        )
        self.schedule_encoder = nn.Sequential(
            nn.Linear(len(SCHEDULE_FEATURE_NAMES), 32),
            nn.LayerNorm(32),
            nn.SiLU(),
        )
        if use_state:
            self.state_encoder = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.LayerNorm(128),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.LayerNorm(64),
                nn.SiLU(),
            )
            fusion_dim = 128 + 32 + 64
        else:
            self.state_encoder = None
            fusion_dim = 128 + 32
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.SiLU(),
        )
        self.regret_head = nn.Linear(64, 1)
        self.harm_head = nn.Linear(64, 1)

    def forward(
        self,
        pooled_t5: torch.Tensor,
        state: torch.Tensor,
        schedule: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        parts = [self.prompt_encoder(pooled_t5), self.schedule_encoder(schedule)]
        if self.use_state:
            parts.append(self.state_encoder(state))
        fused = self.fusion(torch.cat(parts, dim=-1))
        return {
            "scaled_regret": self.regret_head(fused).squeeze(-1),
            "harm_logit": self.harm_head(fused).squeeze(-1),
        }


@torch.no_grad()
def predict_trajectory(
    model: VariableLambdaRouter,
    trajectory: dict[str, Any],
    lambda_value: float,
    candidate_steps: np.ndarray,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    cost_profile: np.ndarray,
    regret_scale: float,
    harm_epsilon: float,
    risk_threshold: float,
    device: torch.device,
) -> tuple[int, np.ndarray, np.ndarray]:
    model.eval()
    state = (trajectory["features"] - state_mean) / state_std
    schedule = schedule_features(
        candidate_steps, trajectory["sigmas"], lambda_value, cost_profile
    )
    count = len(candidate_steps)
    output = model(
        torch.from_numpy(trajectory["pooled_t5"])
        .to(device)
        .unsqueeze(0)
        .expand(count, -1),
        torch.from_numpy(state.astype(np.float32)).to(device),
        torch.from_numpy(schedule).to(device),
    )
    predicted_regret = output["scaled_regret"].clamp(min=0).cpu().numpy() / regret_scale
    harm_probability = torch.sigmoid(output["harm_logit"]).cpu().numpy()
    eligible = np.flatnonzero(
        (predicted_regret <= harm_epsilon) & (harm_probability <= risk_threshold)
    )
    chosen_index = int(eligible[0]) if eligible.size else count - 1
    return chosen_index, predicted_regret, harm_probability


def best_fixed_steps(
    trajectories: list[dict[str, Any]], lambdas: list[float]
) -> dict[float, int]:
    result = {}
    for lambda_value in lambdas:
        utilities = np.stack(
            [
                trajectory["qualities"] - lambda_value * trajectory["costs"]
                for trajectory in trajectories
            ]
        )
        result[lambda_value] = int(np.argmax(utilities.mean(axis=0)))
    return result


@torch.no_grad()
def evaluate_model(
    model: VariableLambdaRouter,
    model_type: str,
    trajectories: list[dict[str, Any]],
    lambdas: list[float],
    candidate_steps: np.ndarray,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    cost_profile: np.ndarray,
    regret_scale: float,
    harm_epsilon: float,
    risk_threshold: float,
    fixed_steps: dict[float, int],
    quality_dimensions: list[str],
    device: torch.device,
    eval_batch_trajectories: int,
    emit_rows: bool,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    lambda_regrets = []
    for lambda_value in lambdas:
        fixed_index = fixed_steps[lambda_value]
        for start in range(0, len(trajectories), eval_batch_trajectories):
            chunk = trajectories[start : start + eval_batch_trajectories]
            batch_size = len(chunk)
            candidate_count = len(candidate_steps)
            pooled = np.repeat(
                np.stack([trajectory["pooled_t5"] for trajectory in chunk])[:, None, :],
                candidate_count,
                axis=1,
            ).reshape(batch_size * candidate_count, -1)
            state = np.stack(
                [
                    (trajectory["features"] - state_mean) / state_std
                    for trajectory in chunk
                ]
            ).reshape(batch_size * candidate_count, -1)
            schedule = np.stack(
                [
                    schedule_features(
                        candidate_steps,
                        trajectory["sigmas"],
                        lambda_value,
                        cost_profile,
                    )
                    for trajectory in chunk
                ]
            ).reshape(batch_size * candidate_count, -1)
            output = model(
                torch.from_numpy(pooled).to(device),
                torch.from_numpy(state.astype(np.float32)).to(device),
                torch.from_numpy(schedule).to(device),
            )
            predicted_regret = (
                output["scaled_regret"]
                .clamp(min=0)
                .reshape(batch_size, candidate_count)
                .cpu()
                .numpy()
                / regret_scale
            )
            harm_probability = (
                torch.sigmoid(output["harm_logit"])
                .reshape(batch_size, candidate_count)
                .cpu()
                .numpy()
            )
            eligible = (predicted_regret <= harm_epsilon) & (
                harm_probability <= risk_threshold
            )
            has_eligible = eligible.any(axis=1)
            chosen_indices = np.where(
                has_eligible, eligible.argmax(axis=1), candidate_count - 1
            )
            for row_index, trajectory in enumerate(chunk):
                chosen = int(chosen_indices[row_index])
                utility = trajectory["qualities"] - lambda_value * trajectory["costs"]
                oracle_index = int(np.argmax(utility))
                oracle_utility = float(utility[oracle_index])
                realized_utility = float(utility[chosen])
                regret = max(0.0, oracle_utility - realized_utility)
                fixed_regret = max(0.0, oracle_utility - float(utility[fixed_index]))
                lambda_regrets.append(regret)
                if emit_rows:
                    row = {
                        "split": trajectory["split"],
                        "model_type": model_type,
                        "Method": MODEL_LABELS[model_type],
                        "prompt_id": trajectory["prompt_id"],
                        "seed": trajectory["seed"],
                        "lambda": lambda_value,
                        "chosen_step": int(candidate_steps[chosen]),
                        "oracle_step": int(candidate_steps[oracle_index]),
                        "best_fixed_step": int(candidate_steps[fixed_index]),
                        "policy_regret": regret,
                        "best_fixed_regret": fixed_regret,
                        "realized_utility": realized_utility,
                        "oracle_utility": oracle_utility,
                        "realized_vbench5": float(trajectory["qualities"][chosen]),
                        "realized_latency_sec": float(trajectory["latencies"][chosen]),
                        "speedup_vs_native": float(
                            trajectory["native_latency"]
                            / trajectory["latencies"][chosen]
                        ),
                        "harmful_stop": int(regret > harm_epsilon),
                        "predicted_stop_regret": float(
                            predicted_regret[row_index, chosen]
                        ),
                        "predicted_harm_probability": float(
                            harm_probability[row_index, chosen]
                        ),
                    }
                    for dimension_index, dimension in enumerate(quality_dimensions):
                        row[f"realized_{dimension}"] = float(
                            trajectory["dimensions"][chosen, dimension_index]
                        )
                    rows.append(row)
    return {"macro_policy_regret": float(np.mean(lambda_regrets))}, rows


def train_model(
    model_type: str,
    train_loader: DataLoader,
    validation_trajectories: list[dict[str, Any]],
    eval_lambdas: list[float],
    candidate_steps: np.ndarray,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    cost_profile: np.ndarray,
    fixed_steps: dict[float, int],
    quality_dimensions: list[str],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[VariableLambdaRouter, int, dict[str, float]]:
    use_state = model_type == "prompt_state"
    model = VariableLambdaRouter(
        state_dim=len(state_mean), use_state=use_state, dropout=args.dropout
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.05
    )
    best_regret = float("inf")
    best_epoch = 0
    best_weights = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            output = model(
                batch["pooled_t5"].to(device),
                batch["state"].to(device),
                batch["schedule"].to(device),
            )
            regret_loss = F.smooth_l1_loss(
                output["scaled_regret"], batch["regret_target"].to(device)
            )
            harm_loss = F.binary_cross_entropy_with_logits(
                output["harm_logit"], batch["harm_target"].to(device)
            )
            loss = (
                args.regret_loss_weight * regret_loss
                + args.harm_loss_weight * harm_loss
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss for {model_type}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        metrics, _ = evaluate_model(
            model,
            model_type,
            validation_trajectories,
            eval_lambdas,
            candidate_steps,
            state_mean,
            state_std,
            cost_profile,
            args.regret_scale,
            args.harm_epsilon,
            args.risk_threshold,
            fixed_steps,
            quality_dimensions,
            device,
            args.eval_batch_trajectories,
            emit_rows=False,
        )
        if metrics["macro_policy_regret"] < best_regret:
            best_regret = metrics["macro_policy_regret"]
            best_epoch = epoch
            best_weights = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(
                f"{model_type} epoch={epoch:02d}/{args.epochs} "
                f"loss={np.mean(losses):.6f} val_macro_regret="
                f"{metrics['macro_policy_regret']:.6f}"
            )
    if best_weights is None:
        raise RuntimeError(f"No checkpoint selected for {model_type}")
    model.load_state_dict(best_weights)
    final_metrics, _ = evaluate_model(
        model,
        model_type,
        validation_trajectories,
        eval_lambdas,
        candidate_steps,
        state_mean,
        state_std,
        cost_profile,
        args.regret_scale,
        args.harm_epsilon,
        args.risk_threshold,
        fixed_steps,
        quality_dimensions,
        device,
        args.eval_batch_trajectories,
        emit_rows=False,
    )
    return model, best_epoch, final_metrics


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    dataset_dir = Path(args.dataset_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists():
        raise FileExistsError(f"Output directory already exists: {out_dir}")
    out_dir.mkdir(parents=True)
    manifest = load_dataset_manifest(dataset_dir)
    selected_indices, selected_names = select_feature_indices(
        manifest, args.feature_groups
    )
    train_trajectories = load_trajectories(
        dataset_dir, manifest, "train", selected_indices
    )
    validation_trajectories = load_trajectories(
        dataset_dir, manifest, "validation", selected_indices
    )
    train_prompts = {trajectory["prompt_id"] for trajectory in train_trajectories}
    validation_prompts = {
        trajectory["prompt_id"] for trajectory in validation_trajectories
    }
    if train_prompts & validation_prompts:
        raise ValueError("Train and validation prompts overlap")
    candidate_steps = np.asarray(manifest["candidate_steps"], dtype=np.int64)
    state_mean, state_std = fit_state_normalizer(train_trajectories)
    cost_profile = fit_cost_profile(train_trajectories)
    fixed_steps = best_fixed_steps(train_trajectories, args.eval_lambdas)
    train_dataset = LambdaStateDataset(
        train_trajectories,
        args.train_lambdas,
        candidate_steps,
        state_mean,
        state_std,
        cost_profile,
        args.harm_epsilon,
        args.regret_scale,
    )
    device = torch.device(args.device)
    models_to_train = (
        [args.model_type]
        if args.model_type != "both"
        else ["prompt_only", "prompt_state"]
    )
    summary_rows = []
    prediction_rows = []
    for model_type in models_to_train:
        seed_everything(args.seed)
        generator = torch.Generator().manual_seed(args.seed)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
        )
        model, best_epoch, metrics = train_model(
            model_type,
            train_loader,
            validation_trajectories,
            args.eval_lambdas,
            candidate_steps,
            state_mean,
            state_std,
            cost_profile,
            fixed_steps,
            manifest["quality_dimensions"],
            args,
            device,
        )
        _, rows = evaluate_model(
            model,
            model_type,
            validation_trajectories,
            args.eval_lambdas,
            candidate_steps,
            state_mean,
            state_std,
            cost_profile,
            args.regret_scale,
            args.harm_epsilon,
            args.risk_threshold,
            fixed_steps,
            manifest["quality_dimensions"],
            device,
            args.eval_batch_trajectories,
            emit_rows=True,
        )
        prediction_rows.extend(rows)
        summary_rows.append(
            {
                "model_type": model_type,
                "Method": MODEL_LABELS[model_type],
                "best_epoch": best_epoch,
                **metrics,
            }
        )
        torch.save(
            {
                "schema": "variable_lambda_router_checkpoint_v1",
                "model_type": model_type,
                "state_dict": model.state_dict(),
                "state_dim": len(state_mean),
                "dropout": args.dropout,
                "selected_feature_names": selected_names,
                "schedule_feature_names": SCHEDULE_FEATURE_NAMES,
                "state_mean": state_mean,
                "state_std": state_std,
                "cost_profile": cost_profile,
                "candidate_steps": candidate_steps,
                "train_lambdas": args.train_lambdas,
                "eval_lambdas": args.eval_lambdas,
                "harm_epsilon": args.harm_epsilon,
                "risk_threshold": args.risk_threshold,
                "regret_scale": args.regret_scale,
                "best_epoch": best_epoch,
                "validation_metrics": metrics,
                "dataset_manifest": str(dataset_dir / "dataset_manifest.json"),
                "dataset_manifest_sha256": sha256_file(
                    dataset_dir / "dataset_manifest.json"
                ),
            },
            out_dir / f"{model_type}_router.pt",
        )

    write_csv(out_dir / "validation_predictions.csv", prediction_rows)
    write_csv(out_dir / "validation_model_summary.csv", summary_rows)
    selected_model = min(summary_rows, key=lambda row: row["macro_policy_regret"])
    run_summary = {
        "schema": "variable_lambda_router_selection_run_v1",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "evaluation_split": "validation",
        "test_accessed": False,
        "train_seed": args.seed,
        "model_type_request": args.model_type,
        "selected_model_type": selected_model["model_type"],
        "selection_rule": "minimum validation macro policy regret across eval lambdas",
        "train_lambdas": args.train_lambdas,
        "eval_lambdas": args.eval_lambdas,
        "primary_lambda": args.primary_lambda,
        "harm_epsilon": args.harm_epsilon,
        "risk_threshold": args.risk_threshold,
        "feature_groups": args.feature_groups,
        "selected_feature_count": len(selected_names),
        "dataset_manifest": str(dataset_dir / "dataset_manifest.json"),
        "dataset_manifest_sha256": sha256_file(dataset_dir / "dataset_manifest.json"),
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "regret_scale": args.regret_scale,
            "regret_loss_weight": args.regret_loss_weight,
            "harm_loss_weight": args.harm_loss_weight,
            "eval_batch_trajectories": args.eval_batch_trajectories,
        },
        "train_prompts": len(train_prompts),
        "validation_prompts": len(validation_prompts),
        "train_trajectories": len(train_trajectories),
        "validation_trajectories": len(validation_trajectories),
        "cost_profile": cost_profile.tolist(),
        "fixed_steps": {
            f"{lambda_value:.6f}": int(candidate_steps[index])
            for lambda_value, index in fixed_steps.items()
        },
        "models": summary_rows,
        "artifacts": {
            "predictions": "validation_predictions.csv",
            "model_summary": "validation_model_summary.csv",
        },
    }
    (out_dir / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(run_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
