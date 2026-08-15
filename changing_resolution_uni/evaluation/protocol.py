from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import lmdb
import torch

from . import PROTOCOL_VERSION


def canonical_scale(value: str | float) -> str:
    return f"{float(value):g}"


def checkpoint_split_config(payload: dict[str, Any]) -> dict[str, Any]:
    train = payload.get("config", {}).get("train", {})
    return {
        "seed": int(train.get("seed", 1234)),
        "val_ratio": float(train.get("val_ratio", 0.0)),
        "val_max_samples": int(train.get("val_max_samples", 0)),
    }


def source_split(
    source_count: int,
    *,
    split: str,
    seed: int,
    val_ratio: float,
    val_max_samples: int,
) -> list[int]:
    """Reproduce the source-level split in ``changing_resolution_uni.train``."""

    if source_count < 1:
        return []
    if val_ratio <= 0 or source_count < 2:
        val_sources: list[int] = []
        train_sources = list(range(source_count))
    else:
        count = min(source_count - 1, max(1, int(round(source_count * val_ratio))))
        if val_max_samples > 0:
            count = min(count, val_max_samples)
        shuffled = list(range(source_count))
        random.Random(seed).shuffle(shuffled)
        val_sources = sorted(shuffled[:count])
        train_sources = sorted(shuffled[count:])

    if split == "val":
        return val_sources
    if split == "train":
        return train_sources
    if split == "all":
        return list(range(source_count))
    raise ValueError(f"split must be val, train, or all, got {split!r}")


def dataset_fingerprint(dataset: Any) -> str:
    """Hash shard metadata and source identity metadata without reading latent blobs."""

    digest = hashlib.sha256()
    for shard_id, shard in enumerate(dataset.shards):
        meta = dataset.shard_meta[shard_id]
        digest.update(shard.name.encode("utf-8"))
        digest.update(_canonical_json(meta).encode("utf-8"))
        env = lmdb.open(
            str(shard), readonly=True, lock=False, readahead=False, meminit=False
        )
        try:
            with env.begin() as txn:
                for row_id in range(int(meta["num_samples"])):
                    raw = txn.get(f"meta_{row_id:08d}".encode("utf-8")) or b"{}"
                    digest.update(row_id.to_bytes(8, byteorder="little", signed=False))
                    digest.update(raw)
        finally:
            env.close()
    return digest.hexdigest()


def build_manifest(
    dataset: Any,
    *,
    data_dir: str | Path,
    split: str,
    split_config: dict[str, Any],
) -> dict[str, Any]:
    selected = source_split(
        dataset.num_source_samples,
        split=split,
        seed=int(split_config["seed"]),
        val_ratio=float(split_config["val_ratio"]),
        val_max_samples=int(split_config["val_max_samples"]),
    )
    records = []
    for source_index in selected:
        shard_id, row_id = dataset.index[source_index]
        sample_meta = _read_source_meta(dataset.shards[shard_id], row_id)
        identity = {
            "shard": dataset.shards[shard_id].name,
            "row_id": int(row_id),
            "source_video": sample_meta.get("source_video"),
            "clip_index": sample_meta.get("clip_index"),
        }
        source_uid = hashlib.sha256(
            _canonical_json(identity).encode("utf-8")
        ).hexdigest()[:16]
        records.append(
            {
                "source_uid": source_uid,
                "source_index": int(source_index),
                "shard": dataset.shards[shard_id].name,
                "row_id": int(row_id),
                "prompt": Path(str(sample_meta.get("source_video", ""))).stem,
                "source_meta": sample_meta,
                "scales": list(dataset.scales),
            }
        )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "split_config": dict(split_config),
        "data_dir": str(Path(data_dir).resolve()),
        "dataset_fingerprint": dataset_fingerprint(dataset),
        "num_dataset_sources": int(dataset.num_source_samples),
        "num_selected_sources": len(records),
        "scales": list(dataset.scales),
        "sources": records,
    }


def load_or_create_manifest(
    path: str | Path,
    dataset: Any,
    *,
    data_dir: str | Path,
    split: str,
    split_config: dict[str, Any],
    allow_dataset_mismatch: bool = False,
) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        manifest = build_manifest(
            dataset,
            data_dir=data_dir,
            split=split,
            split_config=split_config,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return manifest

    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(
            f"Manifest protocol mismatch: {manifest.get('protocol_version')!r} vs {PROTOCOL_VERSION!r}"
        )
    expected_split = {
        "seed": int(split_config["seed"]),
        "val_ratio": float(split_config["val_ratio"]),
        "val_max_samples": int(split_config["val_max_samples"]),
    }
    if manifest.get("split") != split or manifest.get("split_config") != expected_split:
        raise ValueError(
            "Existing manifest does not match the requested split/config. "
            "Use the original training split or choose a different manifest path."
        )
    current_fingerprint = dataset_fingerprint(dataset)
    if (
        manifest.get("dataset_fingerprint") != current_fingerprint
        and not allow_dataset_mismatch
    ):
        raise ValueError(
            "Dataset fingerprint differs from the frozen evaluation manifest. "
            "Refusing to evaluate a silently changed dataset."
        )
    if tuple(manifest.get("scales", ())) != tuple(dataset.scales):
        raise ValueError("Dataset scales differ from the frozen evaluation manifest")
    return manifest


def select_manifest_sources(
    manifest: dict[str, Any],
    *,
    offset: int = 0,
    max_sources: int = 0,
    rank: int = 0,
    world_size: int = 1,
) -> list[dict[str, Any]]:
    records = list(manifest["sources"])[max(0, offset) :]
    if max_sources > 0:
        records = records[:max_sources]
    return [
        record for index, record in enumerate(records) if index % world_size == rank
    ]


def virtual_index(
    dataset: Any, source_record: dict[str, Any], scale: str | float
) -> int:
    scale = canonical_scale(scale)
    scale_offset = list(dataset.scales).index(scale)
    return int(source_record["source_index"]) * len(dataset.scales) + scale_offset


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def environment_record(
    *,
    checkpoints: Sequence[str | Path],
    data_dir: str | Path,
    manifest: dict[str, Any],
    argv: Sequence[str] | None = None,
    checkpoint_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    git_commit = _git_output(["rev-parse", "HEAD"])
    git_dirty = bool(_git_output(["status", "--porcelain"]))
    cuda_names = []
    if torch.cuda.is_available():
        cuda_names = [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
    return {
        "protocol_version": PROTOCOL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": list(argv or sys.argv),
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "pid": os.getpid(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_devices": cuda_names,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "data_dir": str(Path(data_dir).resolve()),
        "dataset_fingerprint": manifest["dataset_fingerprint"],
        "checkpoints": [
            {
                "path": str(Path(path).resolve()),
                "sha256": (
                    checkpoint_hashes[str(path)]
                    if checkpoint_hashes is not None and str(path) in checkpoint_hashes
                    else sha256_file(path)
                ),
            }
            for path in checkpoints
        ],
    }


def _read_source_meta(shard: Path, row_id: int) -> dict[str, Any]:
    env = lmdb.open(
        str(shard), readonly=True, lock=False, readahead=False, meminit=False
    )
    try:
        with env.begin() as txn:
            raw = txn.get(f"meta_{row_id:08d}".encode("utf-8"))
        if raw is None:
            return {}
        value = json.loads(raw.decode("utf-8"))
        return value if isinstance(value, dict) else {"value": value}
    finally:
        env.close()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _git_output(args: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""
