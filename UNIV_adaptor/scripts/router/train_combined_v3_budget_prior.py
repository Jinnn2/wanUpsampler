from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from UNIV_adaptor.combined_v3 import (  # noqa: E402
    ALLOWED_SPLITS,
    MERGED_DATASET_SCHEMA,
    QUALITY_DIMENSIONS,
    load_json,
    validate_scored_record,
    verify_file,
)
from UNIV_adaptor.data_protocol import (  # noqa: E402
    canonical_sha256,
    sha256_file,
    write_json_atomic,
)


class RelativeQualityPrior(nn.Module):
    def __init__(
        self,
        in_dim: int,
        action_count: int,
        hidden_dims: tuple[int, ...] = (256, 128),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.LayerNorm(in_dim)]
        previous = in_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(previous, hidden), nn.SiLU(), nn.Dropout(dropout)])
            previous = hidden
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(previous, action_count)

    def forward(self, prompt_embedding: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(prompt_embedding))


def validate_merged_index(value: dict[str, Any]) -> dict[str, Any]:
    if value.get("schema") != MERGED_DATASET_SCHEMA:
        raise ValueError(f"unsupported merged dataset: {value.get('schema')}")
    body = {
        key: item
        for key, item in value.items()
        if key not in {"schema", "dataset_sha256"}
    }
    if canonical_sha256(body) != value.get("dataset_sha256"):
        raise ValueError("merged dataset index hash mismatch")
    if value.get("selected_splits") != list(ALLOWED_SPLITS) or value.get(
        "test_accessed"
    ):
        raise ValueError("budget-prior selection requires train/validation only")
    if value.get("quality_dimensions") != list(QUALITY_DIMENSIONS):
        raise ValueError("merged dataset has a non-canonical quality profile")
    return value


def validate_embedding_manifest(
    dataset_root: Path, dataset: dict[str, Any]
) -> tuple[dict[str, Any], dict[int, np.ndarray]]:
    path = dataset_root / "t5_embeddings" / "t5_manifest.json"
    value = load_json(path)
    if value.get("schema") != "prompt_t5_embeddings_manifest_v2":
        raise ValueError("training requires provenance-bound T5 embedding manifest v2")
    if (
        value.get("backend") != "wan_native"
        or value.get("required_backend") != "wan_native"
    ):
        raise ValueError(
            "formal budget-prior training requires the frozen Wan native T5 backend"
        )
    if not value.get("text_encoder_checkpoint_sha256"):
        raise ValueError("T5 manifest does not bind the native encoder checkpoint")
    verify_file(
        value["text_encoder_checkpoint"],
        value["text_encoder_checkpoint_sha256"],
        label="native T5 encoder checkpoint",
    )
    tokenizer_root = Path(value.get("tokenizer_path", "")).resolve()
    expected_tokenizer_root = (
        Path(dataset["model_root"]) / "google" / "umt5-xxl"
    ).resolve()
    if tokenizer_root != expected_tokenizer_root or not tokenizer_root.is_dir():
        raise ValueError("T5 manifest does not use the generation model tokenizer")
    tokenizer_files = value.get("tokenizer_files")
    if not isinstance(tokenizer_files, list) or not tokenizer_files:
        raise ValueError("T5 manifest has no tokenizer file inventory")
    if canonical_sha256(tokenizer_files) != value.get("tokenizer_files_sha256"):
        raise ValueError("T5 tokenizer inventory hash mismatch")
    for item in tokenizer_files:
        relative = Path(str(item["relative_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("T5 tokenizer inventory path escapes tokenizer root")
        verify_file(
            tokenizer_root / relative,
            item["sha256"],
            label="T5 tokenizer file",
        )
    extractor = (
        REPO_ROOT
        / "changing_resolution_uni/scripts/data/extract_prompt_t5_embeddings.py"
    )
    if sha256_file(extractor) != value.get("extractor_sha256"):
        raise ValueError(
            "T5 embeddings were produced by a different extractor revision"
        )
    if (
        Path(value.get("model_path", "")).resolve()
        != Path(dataset["model_root"]).resolve()
    ):
        raise ValueError("T5 embeddings use a different Wan model root than generation")
    if (
        not value.get("complete")
        or value.get("prompt_offset") != 0
        or value.get("limit") is not None
    ):
        raise ValueError("T5 embedding manifest is not a complete global prompt pass")
    body = {
        key: item
        for key, item in value.items()
        if key not in {"schema", "manifest_sha256"}
    }
    if canonical_sha256(body) != value.get("manifest_sha256"):
        raise ValueError("T5 embedding manifest hash mismatch")
    prompts_path = verify_file(
        dataset["prompts_file"], value["prompts_file_sha256"], label="embedding prompts"
    )
    if prompts_path != (dataset_root / "prompts.txt").resolve():
        raise ValueError("embedding manifest belongs to another prompt file")
    entries = value.get("prompts")
    if not isinstance(entries, list) or len(entries) != dataset["prompt_count"]:
        raise ValueError("T5 embedding prompt coverage mismatch")
    expected = {int(row["global_prompt_id"]): row for row in dataset["prompts"]}
    embeddings = {}
    t5_root = (dataset_root / "t5_embeddings").resolve()
    for entry in entries:
        prompt_id = int(entry["prompt_id"])
        prompt = expected.get(prompt_id)
        if prompt is None or entry.get("prompt_text") != prompt["prompt"]:
            raise ValueError(f"T5 prompt identity mismatch for global id {prompt_id}")
        raw_digest = hashlib.sha256(prompt["prompt"].encode("utf-8")).hexdigest()
        if entry.get("prompt_sha256") != raw_digest:
            raise ValueError(f"T5 raw prompt digest mismatch for global id {prompt_id}")
        npz_path = verify_file(
            entry["npz_file"], entry["npz_sha256"], label="T5 embedding"
        )
        metadata_path = verify_file(
            entry["json_file"], entry["json_sha256"], label="T5 metadata"
        )
        if npz_path.parent != t5_root or metadata_path.parent != t5_root:
            raise ValueError("T5 embedding path escapes merged dataset root")
        metadata = load_json(metadata_path)
        if (
            metadata.get("prompt_id") != prompt_id
            or metadata.get("prompt_text") != prompt["prompt"]
            or metadata.get("prompt_sha256") != raw_digest
        ):
            raise ValueError(f"T5 metadata mismatch for global id {prompt_id}")
        with np.load(npz_path, allow_pickle=False) as payload:
            pooled = np.asarray(payload["pooled_embedding"], dtype=np.float32)
        if pooled.shape != (4096,) or not np.isfinite(pooled).all():
            raise ValueError(f"invalid pooled T5 embedding: {npz_path}")
        embeddings[prompt_id] = pooled
    if set(embeddings) != set(expected):
        raise ValueError("T5 embedding ids do not cover merged prompts exactly")
    return value, embeddings


def load_samples(
    dataset: dict[str, Any], embeddings: dict[int, np.ndarray]
) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    action_ids = [str(item["artifact_id"]) for item in dataset["action_catalog"]]
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for index_row in dataset["records"]:
        path = verify_file(
            index_row["record_path"],
            index_row["record_file_sha256"],
            label="training scored record",
        )
        record = load_json(path)
        validate_scored_record(record)
        if record["record_sha256"] != index_row["record_sha256"]:
            raise RuntimeError(f"training record identity mismatch: {path}")
        observed_ids = [item["artifact_id"] for item in record["budget_candidates"]]
        if observed_ids != action_ids:
            raise RuntimeError(f"training action order mismatch: {path}")
        global_id = int(index_row["global_prompt_id"])
        grouped[(record["split"], global_id)].append(record)

    samples: dict[str, list[dict[str, Any]]] = {split: [] for split in ALLOWED_SPLITS}
    for (split, global_id), records in sorted(grouped.items()):
        expected_seed_count = 1 if split == "train" else 3
        if len(records) != expected_seed_count:
            raise RuntimeError(
                f"global prompt {global_id} has {len(records)} {split} records; "
                f"expected {expected_seed_count}"
            )
        candidate_quality = np.asarray(
            [
                [
                    candidate["quality"]["vbench5"]
                    for candidate in record["budget_candidates"]
                ]
                for record in records
            ],
            dtype=np.float64,
        ).mean(axis=0)
        native_quality = float(
            np.mean(
                [record["native_teacher"]["quality"]["vbench5"] for record in records]
            )
        )
        dimensions = {
            name: np.asarray(
                [
                    [
                        candidate["quality"]["dimensions"][name]
                        for candidate in record["budget_candidates"]
                    ]
                    for record in records
                ],
                dtype=np.float64,
            ).mean(axis=0)
            for name in QUALITY_DIMENSIONS
        }
        samples[split].append(
            {
                "global_prompt_id": global_id,
                "prompt_sha256": records[0]["prompt_sha256"],
                "seed_count": len(records),
                "embedding": embeddings[global_id],
                "candidate_quality": candidate_quality.astype(np.float32),
                "relative_quality": (candidate_quality - native_quality).astype(
                    np.float32
                ),
                "native_quality": native_quality,
                "dimensions": dimensions,
            }
        )
    expected_counts = dataset["prompts_by_split"]
    for split in ALLOWED_SPLITS:
        if len(samples[split]) != int(expected_counts[split]):
            raise RuntimeError(f"{split} prompt sample coverage mismatch")
    return samples, action_ids


def train_latency_profile(
    dataset: dict[str, Any], action_ids: list[str], *, hardware_label: str
) -> dict[str, Any]:
    native = []
    values: dict[str, list[float]] = {action_id: [] for action_id in action_ids}
    source_records = []
    for row in dataset["records"]:
        if row["split"] != "train":
            continue
        path = verify_file(
            row["record_path"], row["record_file_sha256"], label="latency source record"
        )
        record = load_json(path)
        validate_scored_record(record)
        if record["record_sha256"] != row["record_sha256"]:
            raise RuntimeError(f"latency source record identity mismatch: {path}")
        native.append(float(record["native_teacher"]["cost"]["pipeline_seconds"]))
        for candidate in record["budget_candidates"]:
            values[candidate["artifact_id"]].append(
                float(candidate["cost"]["pipeline_seconds"])
            )
        source_records.append(row["record_sha256"])
    native_median = float(np.median(np.asarray(native, dtype=np.float64)))
    if not math.isfinite(native_median) or native_median <= 0.0:
        raise ValueError("invalid train Native-HR latency median")
    action_seconds = {
        action_id: float(np.median(np.asarray(values[action_id], dtype=np.float64)))
        for action_id in action_ids
    }
    normalized = {
        action_id: seconds / native_median
        for action_id, seconds in action_seconds.items()
    }
    if any(not math.isfinite(value) or value <= 0.0 for value in normalized.values()):
        raise ValueError("invalid train-normalized action latency")
    body = {
        "source": "train_median_generation_pipeline_seconds",
        "hardware_label": hardware_label,
        "hardware_label_provenance": "operator_declared",
        "evidence_scope": "selection_cost_coordinate_not_cross_hardware_speed_claim",
        "selection_split_accessed": "train",
        "record_count": len(source_records),
        "source_record_set_sha256": canonical_sha256(sorted(source_records)),
        "native_median_seconds": native_median,
        "action_median_seconds": action_seconds,
        "normalized_action_cost": normalized,
    }
    return {
        "schema": "univ_combined_v3_train_latency_profile_v1",
        "profile_sha256": canonical_sha256(body),
        **body,
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_tensors(samples: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.from_numpy(np.stack([row["embedding"] for row in samples])).float(),
        torch.from_numpy(
            np.stack([row["relative_quality"] for row in samples])
        ).float(),
    )


@torch.no_grad()
def validation_metrics(
    model: nn.Module,
    samples: list[dict[str, Any]],
    normalized_cost: np.ndarray,
    lambdas: list[float],
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    embeddings, target = sample_tensors(samples)
    predictions = model(embeddings.to(device)).cpu().numpy()
    quality = np.stack([row["candidate_quality"] for row in samples])
    regrets = []
    for lambda_value in lambdas:
        true_utility = quality - lambda_value * normalized_cost[None, :]
        chosen = np.argmax(
            predictions - lambda_value * normalized_cost[None, :], axis=1
        )
        oracle = np.max(true_utility, axis=1)
        realized = true_utility[np.arange(len(samples)), chosen]
        regrets.append(float(np.mean(np.maximum(0.0, oracle - realized))))
    mae = float(np.mean(np.abs(predictions - target.numpy())))
    return float(np.mean(regrets)), mae


def train_one_seed(
    *,
    seed: int,
    train_samples: list[dict[str, Any]],
    validation_samples: list[dict[str, Any]],
    action_ids: list[str],
    normalized_cost: np.ndarray,
    lambdas: list[float],
    args: argparse.Namespace,
    device: torch.device,
    run_dir: Path,
    provenance: dict[str, Any],
    training_config: dict[str, Any],
) -> dict[str, Any]:
    seed_everything(seed)
    model = RelativeQualityPrior(4096, len(action_ids), dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-10,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
    )
    criterion = nn.SmoothL1Loss(beta=args.huber_beta)
    train_x, train_y = sample_tensors(train_samples)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=generator,
    )
    best_state = None
    best_epoch = -1
    best_key = (float("inf"), float("inf"))
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        count = 0
        for embedding, target in loader:
            embedding = embedding.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(embedding)
            loss = criterion(prediction, target)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite training loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * embedding.shape[0]
            count += int(embedding.shape[0])
        scheduler.step()
        regret, mae = validation_metrics(
            model, validation_samples, normalized_cost, lambdas, device
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / max(count, 1),
                "validation_macro_policy_regret": regret,
                "validation_relative_quality_mae": mae,
                "learning_rate": scheduler.get_last_lr()[0],
            }
        )
        if (regret, mae) < best_key:
            best_key = (regret, mae)
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("no validation checkpoint was selected")
    model.load_state_dict(best_state)
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = {
        "schema": "univ_prompt_action_quality_prior_checkpoint_v1",
        "model_type": "relative_quality_mlp",
        "state_dict": best_state,
        "input_dim": 4096,
        "hidden_dims": [256, 128],
        "dropout": args.dropout,
        "action_ids": action_ids,
        "normalized_action_cost": normalized_cost.tolist(),
        "lambdas": lambdas,
        "best_epoch": best_epoch,
        "train_seed": seed,
        "provenance": provenance,
        "training_config": training_config,
    }
    checkpoint_path = run_dir / "budget_prior.pt"
    torch.save(checkpoint, checkpoint_path)
    write_csv(run_dir / "training_history.csv", history)
    result = {
        "train_seed": seed,
        "best_epoch": best_epoch,
        "validation_macro_policy_regret": best_key[0],
        "validation_relative_quality_mae": best_key[1],
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }
    run_body = {
        "provenance": provenance,
        "training_config": training_config,
        "result": result,
    }
    run_summary = {
        "schema": "univ_prompt_action_quality_prior_seed_run_v1",
        "run_sha256": canonical_sha256(run_body),
        **run_body,
    }
    with (run_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(run_summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return result


def load_completed_seed_run(
    run_dir: Path,
    *,
    seed: int,
    provenance: dict[str, Any],
    training_config: dict[str, Any],
) -> dict[str, Any] | None:
    path = run_dir / "run_summary.json"
    if not path.is_file():
        if run_dir.exists():
            raise RuntimeError(
                f"incomplete seed run directory exists; use a new output root: {run_dir}"
            )
        return None
    value = load_json(path)
    if value.get("schema") != "univ_prompt_action_quality_prior_seed_run_v1":
        raise ValueError(f"unsupported seed run summary: {path}")
    body = {
        key: item for key, item in value.items() if key not in {"schema", "run_sha256"}
    }
    if canonical_sha256(body) != value.get("run_sha256"):
        raise ValueError(f"seed run summary hash mismatch: {path}")
    if (
        value.get("provenance") != provenance
        or value.get("training_config") != training_config
    ):
        raise RuntimeError(
            f"seed run belongs to another training configuration: {path}"
        )
    result = value.get("result")
    if not isinstance(result, dict) or int(result.get("train_seed", -1)) != seed:
        raise ValueError(f"seed run identity mismatch: {path}")
    verify_file(
        result["checkpoint"], result["checkpoint_sha256"], label="seed checkpoint"
    )
    return result


@torch.no_grad()
def evaluate_selected(
    checkpoint_path: Path,
    train_samples: list[dict[str, Any]],
    validation_samples: list[dict[str, Any]],
    action_ids: list[str],
    normalized_cost: np.ndarray,
    lambdas: list[float],
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = RelativeQualityPrior(
        4096, len(action_ids), dropout=float(payload["dropout"])
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval()
    val_x, _ = sample_tensors(validation_samples)
    predictions = model(val_x.to(device)).cpu().numpy()
    train_quality = np.stack([row["candidate_quality"] for row in train_samples])
    val_quality = np.stack([row["candidate_quality"] for row in validation_samples])
    summaries = []
    rows = []
    for lambda_value in lambdas:
        train_utility = train_quality - lambda_value * normalized_cost[None, :]
        fixed_index = int(np.argmax(train_utility.mean(axis=0)))
        true_utility = val_quality - lambda_value * normalized_cost[None, :]
        oracle_choice = np.argmax(true_utility, axis=1)
        learned_choice = np.argmax(
            predictions - lambda_value * normalized_cost[None, :], axis=1
        )
        for method, choices in (
            ("prompt_oracle_upper_bound", oracle_choice),
            ("train_selected_fixed", np.full(len(validation_samples), fixed_index)),
            ("learned_prompt_prior", learned_choice),
        ):
            realized_utility = true_utility[np.arange(len(choices)), choices]
            oracle_utility = true_utility[np.arange(len(choices)), oracle_choice]
            realized_quality = val_quality[np.arange(len(choices)), choices]
            realized_dimensions = {
                name: np.asarray(
                    [
                        sample["dimensions"][name][choices[index]]
                        for index, sample in enumerate(validation_samples)
                    ],
                    dtype=np.float64,
                )
                for name in QUALITY_DIMENSIONS
            }
            summary = {
                "lambda": lambda_value,
                "method": method,
                "mean_policy_regret": float(
                    np.mean(np.maximum(0.0, oracle_utility - realized_utility))
                ),
                "mean_realized_utility": float(np.mean(realized_utility)),
                "mean_realized_vbench5": float(np.mean(realized_quality)),
                "mean_normalized_cost": float(np.mean(normalized_cost[choices])),
                "fixed_action_id": action_ids[fixed_index]
                if method == "train_selected_fixed"
                else "",
            }
            for name, values in realized_dimensions.items():
                summary[f"mean_{name}"] = float(np.mean(values))
            summaries.append(summary)
            for sample_index, sample in enumerate(validation_samples):
                choice = int(choices[sample_index])
                row = {
                    "global_prompt_id": sample["global_prompt_id"],
                    "prompt_sha256": sample["prompt_sha256"],
                    "seed_count": sample["seed_count"],
                    "lambda": lambda_value,
                    "method": method,
                    "chosen_action_id": action_ids[choice],
                    "realized_vbench5": float(val_quality[sample_index, choice]),
                    "normalized_cost": float(normalized_cost[choice]),
                    "realized_utility": float(realized_utility[sample_index]),
                    "oracle_utility": float(oracle_utility[sample_index]),
                    "policy_regret": float(
                        max(
                            0.0,
                            oracle_utility[sample_index]
                            - realized_utility[sample_index],
                        )
                    ),
                }
                for name, values in realized_dimensions.items():
                    row[name] = float(values[sample_index])
                rows.append(row)
    return summaries, rows


def paired_bootstrap(
    rows: list[dict[str, Any]], *, samples: int, seed: int
) -> list[dict[str, Any]]:
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    metrics = (
        ("policy_regret", -1.0),
        ("realized_utility", 1.0),
        ("realized_vbench5", 1.0),
        ("normalized_cost", -1.0),
    )
    output = []
    rng = np.random.default_rng(seed)
    lambda_values = sorted({float(row["lambda"]) for row in rows})
    for lambda_value in lambda_values:
        selected = [row for row in rows if float(row["lambda"]) == lambda_value]
        by_method = {
            method: {
                int(row["global_prompt_id"]): row
                for row in selected
                if row["method"] == method
            }
            for method in ("train_selected_fixed", "learned_prompt_prior")
        }
        prompt_ids = sorted(by_method["train_selected_fixed"])
        if set(prompt_ids) != set(by_method["learned_prompt_prior"]):
            raise RuntimeError(
                "paired validation methods have different prompt coverage"
            )
        for metric, direction in metrics:
            improvement = np.asarray(
                [
                    direction
                    * (
                        float(by_method["learned_prompt_prior"][prompt_id][metric])
                        - float(by_method["train_selected_fixed"][prompt_id][metric])
                    )
                    for prompt_id in prompt_ids
                ],
                dtype=np.float64,
            )
            draws = improvement[
                rng.integers(0, improvement.size, size=(samples, improvement.size))
            ].mean(axis=1)
            low, high = np.quantile(draws, [0.025, 0.975])
            output.append(
                {
                    "lambda": lambda_value,
                    "metric": metric,
                    "orientation": "positive_means_learned_better",
                    "improvement_mean": float(improvement.mean()),
                    "ci95_low": float(low),
                    "ci95_high": float(high),
                    "prompt_count": len(prompt_ids),
                    "bootstrap_samples": samples,
                    "bootstrap_seed": seed,
                }
            )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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
    normalized_cost = np.asarray(
        [latency["normalized_action_cost"][action_id] for action_id in action_ids],
        dtype=np.float64,
    )
    lambdas = sorted(set(float(value) for value in args.lambdas))
    if not lambdas or any(value < 0.0 for value in lambdas):
        raise ValueError("lambdas must be non-negative")
    out_root = Path(args.out_root).resolve()
    final_path = out_root / "selection_summary.json"
    if final_path.exists():
        raise FileExistsError(
            f"refusing to overwrite an existing selection: {final_path}"
        )
    out_root.mkdir(parents=True, exist_ok=True)
    latency_path = out_root / "latency_profile.json"
    if latency_path.is_file():
        previous_latency = load_json(latency_path)
        if previous_latency.get("profile_sha256") != latency["profile_sha256"]:
            raise RuntimeError(
                f"output root contains another latency profile: {latency_path}"
            )
    else:
        write_json_atomic(latency_path, latency)
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    runs = []
    training_config = {
        "model_type": "relative_quality_mlp",
        "input_dim": 4096,
        "hidden_dims": [256, 128],
        "dropout": args.dropout,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "optimizer": "AdamW",
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "betas": [0.9, 0.95],
        "eps": 1e-10,
        "schedule": "cosine_to_0.1x",
        "gradient_clip_norm": 1.0,
        "loss": "SmoothL1",
        "huber_beta": args.huber_beta,
        "checkpoint_selection": "validation_macro_policy_regret_then_quality_mae",
    }
    checkpoint_provenance = {
        "dataset_sha256": dataset["dataset_sha256"],
        "embedding_manifest_sha256": embedding_manifest["manifest_sha256"],
        "text_encoder_checkpoint_sha256": embedding_manifest.get(
            "text_encoder_checkpoint_sha256"
        ),
        "latency_profile_sha256": latency["profile_sha256"],
        "evaluation_stage": "validation_only_selection",
        "test_accessed": False,
        "trainer_sha256": sha256_file(Path(__file__).resolve()),
    }
    for seed in args.train_seeds:
        run_dir = out_root / f"seed_{seed}"
        completed = load_completed_seed_run(
            run_dir,
            seed=seed,
            provenance=checkpoint_provenance,
            training_config=training_config,
        )
        if completed is not None:
            print(f"Reusing completed training seed {seed}: {run_dir}")
            runs.append(completed)
        else:
            runs.append(
                train_one_seed(
                    seed=seed,
                    train_samples=samples["train"],
                    validation_samples=samples["validation"],
                    action_ids=action_ids,
                    normalized_cost=normalized_cost,
                    lambdas=lambdas,
                    args=args,
                    device=device,
                    run_dir=run_dir,
                    provenance=checkpoint_provenance,
                    training_config=training_config,
                )
            )
    selected = min(
        runs,
        key=lambda row: (
            row["validation_macro_policy_regret"],
            row["validation_relative_quality_mae"],
            row["train_seed"],
        ),
    )
    selected_checkpoint = Path(selected["checkpoint"])
    frozen_checkpoint = out_root / "selected_budget_prior.pt"
    if frozen_checkpoint.is_file():
        if sha256_file(frozen_checkpoint) != selected["checkpoint_sha256"]:
            raise RuntimeError(
                f"output root contains another selected checkpoint: {frozen_checkpoint}"
            )
    else:
        shutil.copy2(selected_checkpoint, frozen_checkpoint)
    summaries, predictions = evaluate_selected(
        frozen_checkpoint,
        samples["train"],
        samples["validation"],
        action_ids,
        normalized_cost,
        lambdas,
        device,
    )
    write_csv(out_root / "validation_results.csv", summaries)
    write_csv(out_root / "validation_predictions.csv", predictions)
    bootstrap = paired_bootstrap(
        predictions, samples=args.bootstrap_samples, seed=args.bootstrap_seed
    )
    write_csv(out_root / "validation_paired_bootstrap.csv", bootstrap)
    body = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "evaluation_stage": "selection",
        "evaluation_split": "validation",
        "test_accessed": False,
        "model_role": "prompt_only_action_quality_prior",
        "dataset": {
            "path": str(dataset_path),
            "file_sha256": sha256_file(dataset_path),
            "dataset_sha256": dataset["dataset_sha256"],
        },
        "embedding_manifest": {
            "path": str((dataset_root / "t5_embeddings/t5_manifest.json").resolve()),
            "manifest_sha256": embedding_manifest["manifest_sha256"],
            "text_encoder_checkpoint_sha256": embedding_manifest.get(
                "text_encoder_checkpoint_sha256"
            ),
        },
        "latency_profile": latency,
        "action_ids": action_ids,
        "lambdas": lambdas,
        "train_prompt_count": len(samples["train"]),
        "validation_prompt_count": len(samples["validation"]),
        "train_seeds": args.train_seeds,
        "training_config": training_config,
        "runs": runs,
        "selected_run": selected,
        "selected_checkpoint": {
            "path": str(frozen_checkpoint),
            "sha256": sha256_file(frozen_checkpoint),
        },
        "validation_results": summaries,
        "validation_paired_bootstrap": bootstrap,
    }
    summary = {
        "schema": "univ_prompt_action_quality_prior_selection_v1",
        "selection_sha256": canonical_sha256(body),
        **body,
    }
    with final_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                "selected_seed": selected["train_seed"],
                "validation_macro_policy_regret": selected[
                    "validation_macro_policy_regret"
                ],
                "selected_checkpoint": str(frozen_checkpoint),
                "test_accessed": False,
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--train-seeds", nargs="+", type=int, default=[42, 100, 2024])
    parser.add_argument(
        "--lambdas",
        nargs="+",
        type=float,
        default=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10],
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--huber-beta", type=float, default=0.02)
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
    if args.huber_beta <= 0.0 or len(args.train_seeds) != len(set(args.train_seeds)):
        parser.error("huber-beta must be positive and train-seeds unique")
    if args.bootstrap_samples < 1:
        parser.error("bootstrap-samples must be positive")
    if not args.hardware_label.strip():
        parser.error("hardware-label must be non-empty")
    return args


if __name__ == "__main__":
    main()
