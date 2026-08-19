#!/usr/bin/env python3
"""
Prompt-Conditioned Router Dataset & DataLoader with Prompt-Disjoint Splitting.
Loads precomputed T5 embeddings and oracle utility records.
"""
from __future__ import annotations

import json
import os
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
        primary_lambda: float = 0.05,
        tau: float = 0.02,
    ):
        self.samples = samples
        self.t5_dir = Path(t5_dir)
        self.candidate_steps = candidate_steps
        self.K = len(candidate_steps)
        self.step_to_idx = {s: i for i, s in enumerate(candidate_steps)}
        self.primary_lambda = primary_lambda
        self.u_key = f"u_{primary_lambda:.2f}"
        self.tau = tau

        # Pre-cache T5 pooled embeddings for fast training
        self.pooled_cache: dict[int, np.ndarray] = {}
        self._load_t5_cache()

    def _load_t5_cache(self) -> None:
        unique_pids = {s["prompt_id"] for s in self.samples}
        for pid in unique_pids:
            npz_path = self.t5_dir / f"prompt_{pid:06d}.npz"
            if npz_path.is_file():
                try:
                    data = np.load(npz_path)
                    if "pooled_embedding" in data:
                        self.pooled_cache[pid] = data["pooled_embedding"].astype(np.float32)
                except Exception:
                    pass

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | Any]:
        item = self.samples[idx]
        pid = item["prompt_id"]

        # 1. Pooled T5 representation
        if pid in self.pooled_cache:
            pooled = self.pooled_cache[pid]
        else:
            # Fallback zero vector if embedding missing
            pooled = np.zeros(4096, dtype=np.float32)

        # 2. Extract utilities, latencies, and VBench scores
        cands = item.get("candidates", [])
        if not cands and "manifest" in item:
            # Reconstruct from manifest branches if needed
            branches = item["manifest"].get("branches", [])
            cands = [
                {
                    "step": int(b.get("candidate_step", 0)),
                    "vbench5": float(b.get("vbench5", b.get("quality", 0.8))),
                    "latency_seconds": float(b.get("latency_seconds", b.get("estimated_warm_pipeline_seconds", 100.0))),
                }
                for b in branches
            ]

        u_arr = np.zeros(self.K, dtype=np.float32)
        vbench_arr = np.zeros(self.K, dtype=np.float32)
        latency_arr = np.zeros(self.K, dtype=np.float32)

        for c in cands:
            s = int(c["step"])
            if s in self.step_to_idx:
                k_idx = self.step_to_idx[s]
                vb = float(c.get("vbench5", 0.0))
                lat = float(c.get("latency_seconds", 100.0))
                u = float(c.get("utilities", {}).get(self.u_key, vb - self.primary_lambda * (lat / 180.0)))
                u_arr[k_idx] = u
                vbench_arr[k_idx] = vb
                latency_arr[k_idx] = lat

        # 3. Target optimal step
        opt_idx = int(np.argmax(u_arr))
        opt_step = self.candidate_steps[opt_idx]

        # 4. Soft utility target (KL distillation target)
        u_max = np.max(u_arr)
        exp_u = np.exp((u_arr - u_max) / max(self.tau, 1e-4))
        soft_target = exp_u / np.sum(exp_u)

        # 5. Ordinal binary targets: y_k = 1 if opt_idx > k else 0 for k in [0, K-2]
        ordinal_targets = (opt_idx > np.arange(self.K - 1)).astype(np.float32)

        return {
            "prompt_id": pid,
            "seed": item.get("seed", 42),
            "prompt_text": item.get("prompt_text", ""),
            "pooled_t5": torch.from_numpy(pooled),  # [4096]
            "target_step_idx": torch.tensor(opt_idx, dtype=torch.long),
            "target_step": torch.tensor(opt_step, dtype=torch.long),
            "soft_utility_target": torch.from_numpy(soft_target.astype(np.float32)),  # [K]
            "utilities": torch.from_numpy(u_arr),  # [K]
            "vbench5": torch.from_numpy(vbench_arr),  # [K]
            "latencies": torch.from_numpy(latency_arr),  # [K]
            "ordinal_targets": torch.from_numpy(ordinal_targets),  # [K-1]
        }


def create_prompt_disjoint_splits(
    dataset_dir: str | Path,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
    primary_lambda: float = 0.05,
    tau: float = 0.02,
) -> tuple[RouterDataset, RouterDataset, RouterDataset, dict[str, Any]]:
    """
    Partition 1000 prompts strictly by Prompt ID into Train, Val, and Test sets.
    """
    root = Path(dataset_dir)
    records_dir = root / "records"
    t5_dir = root / "t5_embeddings"

    if not records_dir.is_dir():
        raise FileNotFoundError(f"Records directory not found at {records_dir}")

    # Gather all trajectory records
    records_by_prompt: dict[int, list[dict[str, Any]]] = {}
    for r_file in sorted(records_dir.glob("*.json")):
        try:
            data = json.loads(r_file.read_text(encoding="utf-8"))
            pid = int(data["prompt_id"])
            records_by_prompt.setdefault(pid, []).append(data)
        except Exception:
            pass

    all_pids = sorted(list(records_by_prompt.keys()))
    if not all_pids:
        raise ValueError(f"No valid trajectory records found in {records_dir}")

    # Detect candidate steps from first record
    first_record = records_by_prompt[all_pids[0]][0]
    cand_steps = [int(c["step"]) for c in first_record.get("candidates", [])]
    if not cand_steps and "manifest" in first_record:
        cand_steps = [int(b["candidate_step"]) for b in first_record["manifest"].get("branches", [])]
    if not cand_steps:
        cand_steps = [30, 35, *range(40, 51)]

    # Deterministic prompt shuffling
    rng = np.random.RandomState(seed)
    shuffled_pids = rng.permutation(all_pids).tolist()

    n_total = len(shuffled_pids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_pids = set(shuffled_pids[:n_train])
    val_pids = set(shuffled_pids[n_train : n_train + n_val])
    test_pids = set(shuffled_pids[n_train + n_val :])

    train_samples = [s for pid in train_pids for s in records_by_prompt[pid]]
    val_samples = [s for pid in val_pids for s in records_by_prompt[pid]]
    test_samples = [s for pid in test_pids for s in records_by_prompt[pid]]

    train_ds = RouterDataset(train_samples, t5_dir, cand_steps, primary_lambda, tau)
    val_ds = RouterDataset(val_samples, t5_dir, cand_steps, primary_lambda, tau)
    test_ds = RouterDataset(test_samples, t5_dir, cand_steps, primary_lambda, tau)

    meta = {
        "total_prompts": n_total,
        "train_prompts": len(train_pids),
        "val_prompts": len(val_pids),
        "test_prompts": len(test_pids),
        "train_trajectories": len(train_samples),
        "val_trajectories": len(val_samples),
        "test_trajectories": len(test_samples),
        "candidate_steps": cand_steps,
        "primary_lambda": primary_lambda,
    }

    return train_ds, val_ds, test_ds, meta


def get_dataloaders(
    dataset_dir: str | Path,
    batch_size: int = 32,
    num_workers: int = 2,
    seed: int = 42,
    primary_lambda: float = 0.05,
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
