#!/usr/bin/env python3
"""
Prompt-Conditioned Router Dataset & DataLoader with Prompt-Disjoint Splitting.
Loads precomputed T5 embeddings and oracle utility records.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from changing_resolution_uni.scripts.data.oracle_record_schema import (
    FORMAL_STEPS,
    aggregate_prompt_records,
)


class RouterDataset(Dataset):
    """
    Dataset for training prompt-conditioned switching router.
    Features:
      - pooled_t5: [4096] float32
      - target_step_idx: int (0 to K-1)
      - target_step: int (e.g. 47)
      - soft_utility_target: [K] float32
      - utilities: [K] float32
      - latencies: [K] float32
      - vbench5: [K] float32
      - ordinal_labels: [K-1] float32 (binary 1 if k* > threshold_k)
    """

    def __init__(
        self,
        samples: list[dict[str, Any]],
        t5_dir: Path,
        candidate_steps: list[int],
        primary_lambda: float = 0.01,
        tau: float = 0.02,
    ):
        self.samples = samples
        self.t5_dir = Path(t5_dir)
        self.candidate_steps = candidate_steps
        self.K = len(candidate_steps)
        self.step_to_idx = {s: i for i, s in enumerate(candidate_steps)}
        self.primary_lambda = primary_lambda
        self.tau = tau

        # Pre-cache T5 pooled embeddings for fast training
        self.pooled_cache: dict[int, np.ndarray] = {}
        self._load_t5_cache()

    def _load_t5_cache(self) -> None:
        unique_pids = {int(s["prompt_id"]) for s in self.samples}
        errors = []
        for pid in unique_pids:
            npz_path = self.t5_dir / f"prompt_{pid:06d}.npz"
            if not npz_path.is_file():
                errors.append(f"prompt {pid}: missing {npz_path}")
                continue
            try:
                with np.load(npz_path) as data:
                    if "pooled_embedding" not in data:
                        raise ValueError("missing pooled_embedding")
                    pooled = np.asarray(data["pooled_embedding"], dtype=np.float32)
                if pooled.shape != (4096,):
                    raise ValueError(f"expected pooled_embedding shape (4096,), got {pooled.shape}")
                if not np.isfinite(pooled).all():
                    raise ValueError("pooled_embedding contains non-finite values")
                self.pooled_cache[pid] = pooled
            except Exception as exc:
                errors.append(f"prompt {pid}: invalid {npz_path}: {exc}")
        if errors:
            preview = "\n".join(f"  - {item}" for item in errors[:20])
            suffix = "" if len(errors) <= 20 else f"\n  ... and {len(errors) - 20} more"
            raise ValueError(f"T5 embedding coverage check failed:\n{preview}{suffix}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | Any]:
        item = self.samples[idx]
        pid = item["prompt_id"]

        pooled = self.pooled_cache[pid]
        u_arr = np.asarray(item["utilities"], dtype=np.float32)
        vbench_arr = np.asarray(item["vbench5"], dtype=np.float32)
        latency_arr = np.asarray(item["latencies"], dtype=np.float32)
        for name, values in (
            ("utilities", u_arr),
            ("vbench5", vbench_arr),
            ("latencies", latency_arr),
        ):
            if values.shape != (self.K,) or not np.isfinite(values).all():
                raise ValueError(
                    f"prompt {pid}: {name} must have shape ({self.K},) with finite values"
                )

        # 3. Target optimal step
        opt_idx = int(np.argmax(u_arr))
        opt_step = self.candidate_steps[opt_idx]

        # 4. Soft utility target (KL distillation target)
        u_max = np.max(u_arr)
        exp_u = np.exp((u_arr - u_max) / max(self.tau, 1e-4))
        soft_target = exp_u / np.sum(exp_u)

        # 5. Ordinal binary targets: y_k = 1 if opt_idx > k else 0 for k in [0, K-2]
        ordinal_targets = (opt_idx > np.arange(self.K - 1)).astype(np.float32)

        native_lat = float(item["native_latency_seconds"])

        return {
            "prompt_id": pid,
            "seed_count": item["seed_count"],
            "prompt_text": item.get("prompt_text", ""),
            "pooled_t5": torch.from_numpy(pooled).float(),  # [4096]
            "target_step_idx": torch.tensor(opt_idx, dtype=torch.long),
            "target_step": torch.tensor(opt_step, dtype=torch.long),
            "soft_utility_target": torch.from_numpy(soft_target.astype(np.float32)),  # [K]
            "utilities": torch.from_numpy(u_arr),  # [K]
            "vbench5": torch.from_numpy(vbench_arr),  # [K]
            "latencies": torch.from_numpy(latency_arr),  # [K]
            "native_latency": torch.tensor(native_lat, dtype=torch.float32),
            "ordinal_targets": torch.from_numpy(ordinal_targets),  # [K-1]
            "seed_oracle_utility": torch.tensor(
                float(item["seed_oracle_utility"]), dtype=torch.float32
            ),
        }


def create_prompt_disjoint_splits(
    dataset_dir: str | Path,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
    primary_lambda: float = 0.01,
    tau: float = 0.02,
    candidate_steps: list[int] | None = None,
) -> tuple[RouterDataset, RouterDataset, RouterDataset, dict[str, Any]]:
    """
    Partition prompts strictly by Prompt ID into Train, Val, and Test sets.
    """
    root = Path(dataset_dir)
    records_dir = root / "records"
    t5_dir = root / "t5_embeddings"

    if not records_dir.is_dir():
        raise FileNotFoundError(f"Records directory not found at {records_dir}")

    expected_steps = list(candidate_steps or FORMAL_STEPS)
    dataset_manifest = root / "dataset_manifest.json"
    manifest_data: dict[str, Any] = {}
    if dataset_manifest.is_file():
        manifest_data = json.loads(dataset_manifest.read_text(encoding="utf-8"))
        if manifest_data.get("is_complete") is False:
            raise ValueError(f"Dataset manifest is incomplete: {dataset_manifest}")
    if manifest_data.get("record_files"):
        record_files = [records_dir / str(name) for name in manifest_data["record_files"]]
    else:
        record_files = sorted(records_dir.glob("*.json"))

    records_by_prompt_seed: dict[int, dict[int, dict[str, Any]]] = {}
    record_sources: dict[tuple[int, int], Path] = {}
    read_errors: list[str] = []
    for r_file in record_files:
        try:
            data = json.loads(r_file.read_text(encoding="utf-8"))
            pid = int(data["prompt_id"])
            seed_val = int(data.get("seed", 42))
            key = (pid, seed_val)
            if key in record_sources:
                raise ValueError(
                    f"duplicate prompt/seed also present in {record_sources[key].name}"
                )
            record_sources[key] = r_file
            records_by_prompt_seed.setdefault(pid, {})[seed_val] = data
        except Exception as exc:
            read_errors.append(f"{r_file.name}: {exc}")
    if read_errors:
        preview = "\n".join(f"  - {item}" for item in read_errors[:30])
        raise ValueError(f"Failed to index oracle records:\n{preview}")

    records_by_prompt = {
        pid: list(seed_records.values())
        for pid, seed_records in records_by_prompt_seed.items()
    }
    manifest_expected_seeds = None
    if "expected_seeds" in manifest_data:
        manifest_expected_seeds = [int(seed) for seed in manifest_data["expected_seeds"]]
    if manifest_expected_seeds is None:
        manifest_expected_seeds = [42, 100, 2024]
    prompt_samples, expected_seeds = aggregate_prompt_records(
        records_by_prompt,
        candidate_steps=expected_steps,
        primary_lambda=primary_lambda,
        expected_seeds=manifest_expected_seeds,
    )
    expected_prompts = manifest_data.get("expected_prompts")
    if expected_prompts is not None and len(prompt_samples) != int(expected_prompts):
        raise ValueError(
            f"Prompt coverage mismatch: manifest expects {expected_prompts}, "
            f"loaded {len(prompt_samples)}"
        )

    all_pids = sorted(prompt_samples)
    if not all_pids:
        raise ValueError(f"No valid trajectory records found in {records_dir}")

    # Deterministic prompt shuffling
    rng = np.random.RandomState(seed)
    shuffled_pids = rng.permutation(all_pids).tolist()

    n_total = len(shuffled_pids)
    if n_total < 3:
        raise ValueError("Prompt-disjoint train/val/test split requires at least 3 prompts")
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1")
    n_val = max(1, int(n_total * val_ratio))
    n_test = max(1, int(n_total * test_ratio))
    n_train = n_total - n_val - n_test
    if n_train < 1:
        raise ValueError("Prompt-disjoint split left no prompts for training")

    train_pids = set(shuffled_pids[:n_train])
    val_pids = set(shuffled_pids[n_train : n_train + n_val])
    test_pids = set(shuffled_pids[n_train + n_val : n_train + n_val + n_test])

    train_samples = [prompt_samples[pid] for pid in sorted(train_pids)]
    val_samples = [prompt_samples[pid] for pid in sorted(val_pids)]
    test_samples = [prompt_samples[pid] for pid in sorted(test_pids)]

    train_ds = RouterDataset(train_samples, t5_dir, expected_steps, primary_lambda, tau)
    val_ds = RouterDataset(val_samples, t5_dir, expected_steps, primary_lambda, tau)
    test_ds = RouterDataset(test_samples, t5_dir, expected_steps, primary_lambda, tau)

    meta = {
        "total_prompts": n_total,
        "train_prompts": len(train_pids),
        "val_prompts": len(val_pids),
        "test_prompts": len(test_pids),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
        "train_trajectories": len(train_samples) * len(expected_seeds),
        "val_trajectories": len(val_samples) * len(expected_seeds),
        "test_trajectories": len(test_samples) * len(expected_seeds),
        "seeds_per_prompt": expected_seeds,
        "label_granularity": "prompt_mean_utility_across_seeds",
        "candidate_steps": expected_steps,
        "primary_lambda": primary_lambda,
    }

    return train_ds, val_ds, test_ds, meta


def get_dataloaders(
    dataset_dir: str | Path,
    batch_size: int = 32,
    num_workers: int = 2,
    seed: int = 42,
    primary_lambda: float = 0.01,
    tau: float = 0.02,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, Any]]:
    train_ds, val_ds, test_ds, meta = create_prompt_disjoint_splits(
        dataset_dir=dataset_dir,
        seed=seed,
        primary_lambda=primary_lambda,
        tau=tau,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader, meta
