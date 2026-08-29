#!/usr/bin/env python3
"""Train prompt-only and latent-state sequential routers across utility lambdas."""

from __future__ import annotations

import argparse
import copy
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
LATENCY_PROFILE_SCHEMA = "train_calibrated_latency_profile_v1"
MODEL_LABELS = {
    "prompt_only": "Variable-Lambda Prompt Router",
    "prompt_state": "Variable-Lambda Prompt+State Router",
    "b4_offline": "Variable-Lambda Offline B4 Router",
    "b4_prompt_state": "Variable-Lambda B4-Prior+State Router",
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
        choices=[
            "prompt_only",
            "prompt_state",
            "b4_offline",
            "b4_prompt_state",
            "both",
            "b4_pair",
            "all",
        ],
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
    parser.add_argument(
        "--b4-temperature",
        type=float,
        default=0.02,
        help="Temperature for variable-lambda B4 soft utility targets.",
    )
    parser.add_argument(
        "--b4-emd-weight",
        type=float,
        default=0.5,
        help="Weight on ordered-distribution Wasserstein/EMD loss for B4.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-batch-trajectories", type=int, default=64)
    parser.add_argument(
        "--expected-latency-profile-sha256",
        default=None,
        help="Optional operator-pinned SHA256 for the locked train cost profile.",
    )
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
    if args.b4_temperature <= 0 or not math.isfinite(args.b4_temperature):
        parser.error("b4-temperature must be finite and positive")
    if args.b4_emd_weight < 0 or not math.isfinite(args.b4_emd_weight):
        parser.error("b4-emd-weight must be finite and non-negative")
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


def load_locked_latency_profile(
    manifest: dict[str, Any],
    candidate_steps: np.ndarray,
    expected_sha256: str | None,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
    metadata = manifest.get("latency_profile")
    if not isinstance(metadata, dict):
        raise ValueError("State dataset has no locked latency profile")
    profile_path = Path(str(metadata.get("path", ""))).resolve()
    observed_sha256 = sha256_file(profile_path)
    recorded_sha256 = str(metadata.get("sha256", ""))
    if observed_sha256 != recorded_sha256:
        raise ValueError(f"Latency profile SHA256 mismatch: {profile_path}")
    if expected_sha256 and observed_sha256 != expected_sha256.lower():
        raise ValueError(
            "Latency profile differs from --expected-latency-profile-sha256: "
            f"{profile_path}"
        )
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    if payload.get("schema") != LATENCY_PROFILE_SCHEMA:
        raise ValueError(f"Unexpected latency profile schema: {profile_path}")
    if payload.get("monotonic_nonincreasing") is not True:
        raise ValueError("Locked latency profile is not monotonic non-increasing")
    if not np.array_equal(
        np.asarray(payload.get("candidate_steps"), dtype=np.int64), candidate_steps
    ):
        raise ValueError("Latency profile candidate steps mismatch")
    costs = np.asarray(
        payload.get("selected_normalized_cost_profile"), dtype=np.float32
    )
    seconds = np.asarray(
        payload.get("calibrated_candidate_latency_seconds"), dtype=np.float32
    )
    native_seconds = float(payload.get("calibrated_native_latency_seconds", 0.0))
    if costs.shape != candidate_steps.shape or seconds.shape != candidate_steps.shape:
        raise ValueError("Latency profile vector shape mismatch")
    if not np.isfinite(costs).all() or not np.isfinite(seconds).all():
        raise ValueError("Latency profile contains non-finite values")
    if np.any(costs <= 0) or np.any(seconds <= 0) or native_seconds <= 0:
        raise ValueError("Latency profile costs must be positive")
    if not np.allclose(seconds / native_seconds, costs, rtol=1e-5, atol=1e-7):
        raise ValueError("Latency profile seconds and normalized costs disagree")
    provenance = {
        "schema": payload["schema"],
        "path": str(profile_path),
        "sha256": observed_sha256,
        "hardware_label": payload["hardware_label"],
        "source_split": payload["source_split"],
        "source_prompt_count": int(payload["source_prompt_count"]),
        "aggregation": payload["aggregation_used_for_selection"],
        "raw_trajectory_latency_role": "diagnostic_only",
    }
    return costs, seconds, native_seconds, provenance


def apply_locked_latency_profile(
    trajectories: list[dict[str, Any]],
    costs: np.ndarray,
    candidate_seconds: np.ndarray,
    native_seconds: float,
) -> None:
    for trajectory in trajectories:
        trajectory["raw_costs_diagnostic"] = trajectory["costs"]
        trajectory["costs"] = costs.copy()
        trajectory["calibrated_latencies"] = candidate_seconds.copy()
        trajectory["calibrated_native_latency"] = native_seconds


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
            "lambda_value": torch.tensor(lambda_value, dtype=torch.float32),
            "candidate_index": torch.tensor(state_index, dtype=torch.long),
            "regret_target": torch.tensor(
                regret * self.regret_scale, dtype=torch.float32
            ),
            "harm_target": torch.tensor(
                float(regret > self.harm_epsilon), dtype=torch.float32
            ),
        }


class B4LambdaDataset(Dataset):
    """Prompt+lambda examples for the generation-independent B4 prior."""

    def __init__(
        self,
        trajectories: list[dict[str, Any]],
        lambdas: list[float],
        temperature: float,
    ):
        self.trajectories = trajectories
        self.lambdas = lambdas
        self.temperature = float(temperature)

    def __len__(self) -> int:
        return len(self.trajectories) * len(self.lambdas)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        lambda_index = index % len(self.lambdas)
        trajectory_index = index // len(self.lambdas)
        trajectory = self.trajectories[trajectory_index]
        lambda_value = self.lambdas[lambda_index]
        utilities = trajectory["qualities"] - lambda_value * trajectory["costs"]
        centered = (utilities - float(np.max(utilities))) / self.temperature
        weights = np.exp(centered.astype(np.float64))
        soft_target = (weights / weights.sum()).astype(np.float32)
        return {
            "pooled_t5": torch.from_numpy(trajectory["pooled_t5"]),
            "lambda_value": torch.tensor(lambda_value, dtype=torch.float32),
            "soft_utility_target": torch.from_numpy(soft_target),
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
        lambda_value: torch.Tensor | None = None,
        candidate_index: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del lambda_value, candidate_index
        parts = [self.prompt_encoder(pooled_t5), self.schedule_encoder(schedule)]
        if self.use_state:
            parts.append(self.state_encoder(state))
        fused = self.fusion(torch.cat(parts, dim=-1))
        return {
            "scaled_regret": self.regret_head(fused).squeeze(-1),
            "harm_logit": self.harm_head(fused).squeeze(-1),
        }


class VariableLambdaB4Prior(nn.Module):
    """Offline prompt+lambda prior over all candidate handoff steps."""

    def __init__(self, candidate_count: int, dropout: float):
        super().__init__()
        self.candidate_count = int(candidate_count)
        self.prompt_encoder = nn.Sequential(
            nn.Linear(4096, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
        )
        self.lambda_encoder = nn.Sequential(
            nn.Linear(1, 16),
            nn.LayerNorm(16),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(128 + 16, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, self.candidate_count),
        )

    def forward(
        self, pooled_t5: torch.Tensor, lambda_value: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        prompt_feature = self.prompt_encoder(pooled_t5)
        scaled_lambda = lambda_value.reshape(-1, 1) / 0.1
        lambda_feature = self.lambda_encoder(scaled_lambda)
        logits = self.head(torch.cat([prompt_feature, lambda_feature], dim=-1))
        return {
            "logits": logits,
            "discrete_probs": torch.softmax(logits, dim=-1),
            "prompt_feature": prompt_feature,
        }


class B4PromptStateRouter(nn.Module):
    """Frozen offline B4 prior with a trainable online latent correction head."""

    def __init__(
        self,
        b4_prior: VariableLambdaB4Prior,
        state_dim: int,
        candidate_steps: np.ndarray,
        dropout: float,
    ):
        super().__init__()
        self.b4_prior = copy.deepcopy(b4_prior)
        for parameter in self.b4_prior.parameters():
            parameter.requires_grad_(False)
        candidate_tensor = torch.as_tensor(candidate_steps, dtype=torch.float32)
        self.register_buffer("candidate_steps", candidate_tensor)
        candidate_count = int(candidate_tensor.numel())
        self.schedule_encoder = nn.Sequential(
            nn.Linear(len(SCHEDULE_FEATURE_NAMES), 32),
            nn.LayerNorm(32),
            nn.SiLU(),
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
        )
        # Full B4 distribution plus expected step, entropy, maximum probability,
        # probability at the current candidate, tail mass, and top-1 margin.
        self.prior_encoder = nn.Sequential(
            nn.Linear(candidate_count + 6, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(128 + 32 + 64 + 32, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.SiLU(),
        )
        self.regret_head = nn.Linear(64, 1)
        self.harm_head = nn.Linear(64, 1)

    def train(self, mode: bool = True) -> "B4PromptStateRouter":
        super().train(mode)
        self.b4_prior.eval()
        return self

    def forward(
        self,
        pooled_t5: torch.Tensor,
        state: torch.Tensor,
        schedule: torch.Tensor,
        lambda_value: torch.Tensor | None = None,
        candidate_index: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if lambda_value is None or candidate_index is None:
            raise ValueError("B4PromptStateRouter requires lambda and candidate index")
        self.b4_prior.eval()
        with torch.no_grad():
            prior = self.b4_prior(pooled_t5, lambda_value)
        probabilities = prior["discrete_probs"]
        normalized_steps = self.candidate_steps / 50.0
        expected_step = (probabilities * normalized_steps.unsqueeze(0)).sum(dim=1)
        entropy = -(
            probabilities.clamp_min(1e-8) * probabilities.clamp_min(1e-8).log()
        ).sum(dim=1) / math.log(probabilities.shape[1])
        top2 = torch.topk(probabilities, k=min(2, probabilities.shape[1]), dim=1).values
        margin = top2[:, 0] - (top2[:, 1] if top2.shape[1] > 1 else 0.0)
        current_probability = probabilities.gather(
            1, candidate_index.reshape(-1, 1)
        ).squeeze(1)
        positions = torch.arange(probabilities.shape[1], device=probabilities.device)
        tail_probability = (
            probabilities * (positions.unsqueeze(0) > candidate_index.unsqueeze(1))
        ).sum(dim=1)
        prior_features = torch.cat(
            [
                probabilities,
                expected_step.unsqueeze(1),
                entropy.unsqueeze(1),
                probabilities.max(dim=1).values.unsqueeze(1),
                current_probability.unsqueeze(1),
                tail_probability.unsqueeze(1),
                margin.unsqueeze(1),
            ],
            dim=1,
        )
        fused = self.fusion(
            torch.cat(
                [
                    prior["prompt_feature"],
                    self.schedule_encoder(schedule),
                    self.state_encoder(state),
                    self.prior_encoder(prior_features),
                ],
                dim=1,
            )
        )
        return {
            "scaled_regret": self.regret_head(fused).squeeze(-1),
            "harm_logit": self.harm_head(fused).squeeze(-1),
        }


@torch.no_grad()
def predict_trajectory(
    model: nn.Module,
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
        torch.full((count,), lambda_value, dtype=torch.float32, device=device),
        torch.arange(count, dtype=torch.long, device=device),
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


def prediction_row(
    *,
    trajectory: dict[str, Any],
    model_type: str,
    lambda_value: float,
    chosen: int,
    oracle_index: int,
    fixed_index: int,
    candidate_steps: np.ndarray,
    regret: float,
    fixed_regret: float,
    realized_utility: float,
    oracle_utility: float,
    harm_epsilon: float,
    quality_dimensions: list[str],
    decision_mode: str,
    predicted_stop_regret: float | str,
    predicted_harm_probability: float | str,
) -> dict[str, Any]:
    row = {
        "split": trajectory["split"],
        "model_type": model_type,
        "Method": MODEL_LABELS[model_type],
        "decision_mode": decision_mode,
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
        "realized_latency_sec": float(trajectory["calibrated_latencies"][chosen]),
        "speedup_vs_native": float(
            trajectory["calibrated_native_latency"]
            / trajectory["calibrated_latencies"][chosen]
        ),
        "normalized_cost": float(trajectory["costs"][chosen]),
        "raw_manifest_latency_sec_diagnostic": float(
            trajectory["latencies"][chosen]
        ),
        "raw_manifest_speedup_diagnostic": float(
            trajectory["native_latency"] / trajectory["latencies"][chosen]
        ),
        "harmful_stop": int(regret > harm_epsilon),
        "predicted_stop_regret": predicted_stop_regret,
        "predicted_harm_probability": predicted_harm_probability,
    }
    for dimension_index, dimension in enumerate(quality_dimensions):
        row[f"realized_{dimension}"] = float(
            trajectory["dimensions"][chosen, dimension_index]
        )
    return row


@torch.no_grad()
def evaluate_b4_offline_model(
    model: VariableLambdaB4Prior,
    trajectories: list[dict[str, Any]],
    lambdas: list[float],
    candidate_steps: np.ndarray,
    fixed_steps: dict[float, int],
    harm_epsilon: float,
    quality_dimensions: list[str],
    device: torch.device,
    eval_batch_trajectories: int,
    emit_rows: bool,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    lambda_regrets = []
    for lambda_value in lambdas:
        fixed_index = fixed_steps[lambda_value]
        for start in range(0, len(trajectories), eval_batch_trajectories):
            chunk = trajectories[start : start + eval_batch_trajectories]
            pooled = torch.from_numpy(
                np.stack([trajectory["pooled_t5"] for trajectory in chunk])
            ).to(device)
            lambda_tensor = torch.full(
                (len(chunk),), lambda_value, dtype=torch.float32, device=device
            )
            probabilities = model(pooled, lambda_tensor)["discrete_probs"]
            chosen_indices = probabilities.argmax(dim=1).cpu().numpy()
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
                    rows.append(
                        prediction_row(
                            trajectory=trajectory,
                            model_type="b4_offline",
                            lambda_value=lambda_value,
                            chosen=chosen,
                            oracle_index=oracle_index,
                            fixed_index=fixed_index,
                            candidate_steps=candidate_steps,
                            regret=regret,
                            fixed_regret=fixed_regret,
                            realized_utility=realized_utility,
                            oracle_utility=oracle_utility,
                            harm_epsilon=harm_epsilon,
                            quality_dimensions=quality_dimensions,
                            decision_mode="offline_argmax",
                            predicted_stop_regret="",
                            predicted_harm_probability="",
                        )
                    )
    return {"macro_policy_regret": float(np.mean(lambda_regrets))}, rows


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
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
    if model_type == "b4_offline":
        if not isinstance(model, VariableLambdaB4Prior):
            raise TypeError("b4_offline evaluation requires VariableLambdaB4Prior")
        return evaluate_b4_offline_model(
            model,
            trajectories,
            lambdas,
            candidate_steps,
            fixed_steps,
            harm_epsilon,
            quality_dimensions,
            device,
            eval_batch_trajectories,
            emit_rows,
        )
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
            lambda_tensor = torch.full(
                (batch_size * candidate_count,),
                lambda_value,
                dtype=torch.float32,
                device=device,
            )
            candidate_index = torch.arange(
                candidate_count, dtype=torch.long, device=device
            ).repeat(batch_size)
            output = model(
                torch.from_numpy(pooled).to(device),
                torch.from_numpy(state.astype(np.float32)).to(device),
                torch.from_numpy(schedule).to(device),
                lambda_tensor,
                candidate_index,
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
                    rows.append(
                        prediction_row(
                            trajectory=trajectory,
                            model_type=model_type,
                            lambda_value=lambda_value,
                            chosen=chosen,
                            oracle_index=oracle_index,
                            fixed_index=fixed_index,
                            candidate_steps=candidate_steps,
                            regret=regret,
                            fixed_regret=fixed_regret,
                            realized_utility=realized_utility,
                            oracle_utility=oracle_utility,
                            harm_epsilon=harm_epsilon,
                            quality_dimensions=quality_dimensions,
                            decision_mode="sequential_stop",
                            predicted_stop_regret=float(
                                predicted_regret[row_index, chosen]
                            ),
                            predicted_harm_probability=float(
                                harm_probability[row_index, chosen]
                            ),
                        )
                    )
    return {"macro_policy_regret": float(np.mean(lambda_regrets))}, rows


def ordered_emd_loss(
    predicted_probabilities: torch.Tensor, target_probabilities: torch.Tensor
) -> torch.Tensor:
    predicted_cdf = predicted_probabilities.cumsum(dim=1)
    target_cdf = target_probabilities.cumsum(dim=1)
    return torch.mean(torch.abs(predicted_cdf - target_cdf))


def train_b4_model(
    train_loader: DataLoader,
    validation_trajectories: list[dict[str, Any]],
    eval_lambdas: list[float],
    candidate_steps: np.ndarray,
    fixed_steps: dict[float, int],
    quality_dimensions: list[str],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[VariableLambdaB4Prior, int, dict[str, float]]:
    model = VariableLambdaB4Prior(
        candidate_count=len(candidate_steps), dropout=args.dropout
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
                batch["lambda_value"].to(device),
            )
            target = batch["soft_utility_target"].to(device)
            kl_loss = F.kl_div(
                F.log_softmax(output["logits"], dim=1),
                target,
                reduction="batchmean",
            )
            emd_loss = ordered_emd_loss(output["discrete_probs"], target)
            loss = kl_loss + args.b4_emd_weight * emd_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss for b4_offline")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        metrics, _ = evaluate_b4_offline_model(
            model,
            validation_trajectories,
            eval_lambdas,
            candidate_steps,
            fixed_steps,
            args.harm_epsilon,
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
                f"b4_offline epoch={epoch:02d}/{args.epochs} "
                f"loss={np.mean(losses):.6f} val_macro_regret="
                f"{metrics['macro_policy_regret']:.6f}"
            )
    if best_weights is None:
        raise RuntimeError("No checkpoint selected for b4_offline")
    model.load_state_dict(best_weights)
    final_metrics, _ = evaluate_b4_offline_model(
        model,
        validation_trajectories,
        eval_lambdas,
        candidate_steps,
        fixed_steps,
        args.harm_epsilon,
        quality_dimensions,
        device,
        args.eval_batch_trajectories,
        emit_rows=False,
    )
    return model, best_epoch, final_metrics


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
    b4_prior: VariableLambdaB4Prior | None = None,
) -> tuple[nn.Module, int, dict[str, float]]:
    if model_type in {"prompt_only", "prompt_state"}:
        model: nn.Module = VariableLambdaRouter(
            state_dim=len(state_mean),
            use_state=model_type == "prompt_state",
            dropout=args.dropout,
        ).to(device)
    elif model_type == "b4_prompt_state":
        if b4_prior is None:
            raise ValueError("b4_prompt_state requires a trained B4 prior")
        model = B4PromptStateRouter(
            b4_prior=b4_prior,
            state_dim=len(state_mean),
            candidate_steps=candidate_steps,
            dropout=args.dropout,
        ).to(device)
    else:
        raise ValueError(f"Unsupported online model type: {model_type}")
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
                batch["lambda_value"].to(device),
                batch["candidate_index"].to(device),
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


def requested_model_types(request: str) -> list[str]:
    if request == "both":
        return ["prompt_only", "prompt_state"]
    if request == "b4_pair":
        return ["b4_offline", "b4_prompt_state"]
    if request == "all":
        return [
            "prompt_only",
            "prompt_state",
            "b4_offline",
            "b4_prompt_state",
        ]
    return [request]


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
    (
        cost_profile,
        calibrated_candidate_seconds,
        calibrated_native_seconds,
        latency_profile_provenance,
    ) = load_locked_latency_profile(
        manifest,
        candidate_steps,
        args.expected_latency_profile_sha256,
    )
    apply_locked_latency_profile(
        train_trajectories,
        cost_profile,
        calibrated_candidate_seconds,
        calibrated_native_seconds,
    )
    apply_locked_latency_profile(
        validation_trajectories,
        cost_profile,
        calibrated_candidate_seconds,
        calibrated_native_seconds,
    )
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
    models_to_train = requested_model_types(args.model_type)
    b4_prior = None
    b4_best_epoch = None
    b4_metrics = None
    if {"b4_offline", "b4_prompt_state"} & set(models_to_train):
        b4_dataset = B4LambdaDataset(
            train_trajectories,
            args.train_lambdas,
            args.b4_temperature,
        )
        seed_everything(args.seed)
        b4_loader = DataLoader(
            b4_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(args.seed),
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
        )
        b4_prior, b4_best_epoch, b4_metrics = train_b4_model(
            b4_loader,
            validation_trajectories,
            args.eval_lambdas,
            candidate_steps,
            fixed_steps,
            manifest["quality_dimensions"],
            args,
            device,
        )
    summary_rows = []
    prediction_rows = []
    checkpoint_artifacts: dict[str, dict[str, str]] = {}
    for model_type in models_to_train:
        if model_type == "b4_offline":
            if b4_prior is None or b4_best_epoch is None or b4_metrics is None:
                raise RuntimeError("B4 prior was not trained")
            model = b4_prior
            best_epoch = b4_best_epoch
            metrics = b4_metrics
        else:
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
                b4_prior=b4_prior,
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
        checkpoint_path = out_dir / f"{model_type}_router.pt"
        torch.save(
            {
                "schema": "variable_lambda_router_checkpoint_v2",
                "model_type": model_type,
                "state_dict": model.state_dict(),
                "state_dim": len(state_mean),
                "dropout": args.dropout,
                "selected_feature_names": selected_names,
                "schedule_feature_names": SCHEDULE_FEATURE_NAMES,
                "state_mean": state_mean,
                "state_std": state_std,
                "cost_profile": cost_profile,
                "calibrated_candidate_latency_seconds": calibrated_candidate_seconds,
                "calibrated_native_latency_seconds": calibrated_native_seconds,
                "latency_profile": latency_profile_provenance,
                "candidate_steps": candidate_steps,
                "train_lambdas": args.train_lambdas,
                "eval_lambdas": args.eval_lambdas,
                "harm_epsilon": args.harm_epsilon,
                "risk_threshold": args.risk_threshold,
                "regret_scale": args.regret_scale,
                "b4_temperature": args.b4_temperature,
                "b4_emd_weight": args.b4_emd_weight,
                "b4_prior_frozen": model_type == "b4_prompt_state",
                "best_epoch": best_epoch,
                "validation_metrics": metrics,
                "dataset_manifest": str(dataset_dir / "dataset_manifest.json"),
                "dataset_manifest_sha256": sha256_file(
                    dataset_dir / "dataset_manifest.json"
                ),
            },
            checkpoint_path,
        )
        checkpoint_artifacts[model_type] = {
            "path": checkpoint_path.name,
            "sha256": sha256_file(checkpoint_path),
        }

    write_csv(out_dir / "validation_predictions.csv", prediction_rows)
    write_csv(out_dir / "validation_model_summary.csv", summary_rows)
    selected_model = min(summary_rows, key=lambda row: row["macro_policy_regret"])
    run_summary = {
        "schema": "variable_lambda_router_selection_run_v2",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "evaluation_split": "validation",
        "test_accessed": False,
        "train_seed": args.seed,
        "model_type_request": args.model_type,
        "model_types": models_to_train,
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
            "b4_temperature": args.b4_temperature,
            "b4_emd_weight": args.b4_emd_weight,
            "b4_prior_frozen_in_hybrid": True,
            "eval_batch_trajectories": args.eval_batch_trajectories,
        },
        "train_prompts": len(train_prompts),
        "validation_prompts": len(validation_prompts),
        "train_trajectories": len(train_trajectories),
        "validation_trajectories": len(validation_trajectories),
        "cost_profile": cost_profile.tolist(),
        "calibrated_candidate_latency_seconds": calibrated_candidate_seconds.tolist(),
        "calibrated_native_latency_seconds": calibrated_native_seconds,
        "latency_profile": latency_profile_provenance,
        "fixed_steps": {
            f"{lambda_value:.6f}": int(candidate_steps[index])
            for lambda_value, index in fixed_steps.items()
        },
        "models": summary_rows,
        "artifacts": {
            "predictions": "validation_predictions.csv",
            "model_summary": "validation_model_summary.csv",
            "checkpoints": checkpoint_artifacts,
        },
    }
    (out_dir / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(run_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
