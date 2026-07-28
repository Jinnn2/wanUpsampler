#!/usr/bin/env python3
"""Export LMDB schema/count/split evidence without copying latent tensors."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path
from typing import Any

import lmdb


DATASETS = [
    ("wan50_itu", ("data/changing_resolution/lmdb_368x640_720x1248_1k",), 0.05, 100),
    ("wan50_ttd_step40", ("data/changing_resolution/lmdb_tail_skip_lora_step40_to_step50",), 0.02, 64),
    ("wan50_ttd_step45", ("data/changing_resolution/lmdb_tail_skip_lora_step45_to_step50",), 0.02, 64),
    (
        "distill4_itu",
        ("data/changing_resolution_distill/lmdb_clean_368x640_720x1248_14b_cfgdistill_5k",),
        0.05,
        100,
    ),
    (
        "distill4_ttd_step3",
        (
            "data/changing_resolution_distill/lmdb_last_step_skip_lora_368x640_14b_cfgdistill_5k_step3",
            "data/changing_resolution_distill/lmdb_last_step_skip_lora_14b_cfgdistill_5k_step3",
        ),
        0.02,
        64,
    ),
]


def decode_json(value: bytes | None) -> dict[str, Any]:
    if value is None:
        return {}
    try:
        decoded = json.loads(value.decode("utf-8"))
        return decoded if isinstance(decoded, dict) else {"value": decoded}
    except Exception:
        return {"unparsed_sha256": hashlib.sha256(value).hexdigest()}


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key in {"source_video", "source_lmdb", "model_root", "path"}:
                result[key] = f"<sanitized:{Path(str(item)).name}>"
            elif key == "prompt":
                encoded = str(item).encode("utf-8")
                result[key] = {
                    "sha256": hashlib.sha256(encoded).hexdigest(),
                    "utf8_bytes": len(encoded),
                }
            else:
                result[key] = sanitize(item)
        return result
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    return value


def inspect_dataset(
    project: Path,
    name: str,
    relative_candidates: tuple[str, ...],
    val_ratio: float,
    val_cap: int,
) -> dict[str, Any]:
    relative = relative_candidates[0]
    root = project / relative
    for candidate in relative_candidates:
        candidate_root = project / candidate
        if any(candidate_root.rglob("data.mdb")):
            relative = candidate
            root = candidate_root
            break
    shards = sorted(path.parent for path in root.rglob("data.mdb"))
    result: dict[str, Any] = {
        "name": name,
        "relative_path": relative,
        "candidate_relative_paths": list(relative_candidates),
        "exists": bool(shards),
        "shards": [],
    }
    total = 0
    prompt_hashes: set[str] = set()
    seed_min: int | None = None
    seed_max: int | None = None
    for shard in shards:
        env = lmdb.open(str(shard), readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin() as txn:
            metadata = decode_json(txn.get(b"metadata"))
            count_raw = txn.get(b"num_samples")
            count = int(count_raw.decode("utf-8")) if count_raw else int(metadata.get("num_samples", 0))
            example = decode_json(txn.get(b"meta_00000000_data"))
            for row in range(count):
                prompt = txn.get(f"prompt_{row:08d}_data".encode())
                if prompt:
                    prompt_hashes.add(hashlib.sha256(prompt).hexdigest())
                seed = txn.get(f"seed_{row:08d}_data".encode())
                if seed:
                    parsed = int(seed.decode("utf-8"))
                    seed_min = parsed if seed_min is None else min(seed_min, parsed)
                    seed_max = parsed if seed_max is None else max(seed_max, parsed)
        env.close()
        total += count
        result["shards"].append(
            {
                "name": shard.relative_to(root).as_posix(),
                "samples": count,
                "metadata": sanitize(metadata),
                "first_record_metadata": sanitize(example),
                "data_mdb_bytes": (shard / "data.mdb").stat().st_size,
            }
        )

    val_count = 0
    val_indices: list[int] = []
    if total >= 2 and val_ratio > 0:
        val_count = min(max(1, round(total * val_ratio)), val_cap, total - 1)
        indices = list(range(total))
        random.Random(1234).shuffle(indices)
        val_indices = sorted(indices[:val_count])
    result.update(
        {
            "total_samples": total,
            "unique_prompt_hashes": len(prompt_hashes),
            "seed_min": seed_min,
            "seed_max": seed_max,
            "split": {
                "algorithm": "random.Random(1234).shuffle(indices)",
                "training_samples": total - val_count,
                "validation_samples": val_count,
                "validation_indices": val_indices,
            },
        }
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project = args.project_root.resolve()
    commit = None
    try:
        completed = subprocess.run(
            ["git", "-C", str(project), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        commit = completed.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        pass
    report = {
        "schema_version": 1,
        "project_commit": commit,
        "datasets": [
            inspect_dataset(project, name, candidates, ratio, cap)
            for name, candidates, ratio, cap in DATASETS
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    missing = [item["name"] for item in report["datasets"] if not item["exists"]]
    print(f"wrote {args.output}")
    if missing:
        print("missing dataset roots: " + ", ".join(missing))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
