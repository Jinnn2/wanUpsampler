from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from UNIV_adaptor.combined_v3 import (  # noqa: E402
    ALLOWED_SPLITS,
    MERGED_DATASET_SCHEMA,
    QUALITY_DIMENSIONS,
    SCORED_DATASET_SCHEMA,
    action_catalog,
    load_json,
    validate_scored_record,
    verify_file,
)
from UNIV_adaptor.data_protocol import (  # noqa: E402
    canonical_sha256,
    sha256_file,
    write_json_atomic,
)


def validate_scored_dataset(value: dict[str, Any]) -> dict[str, Any]:
    if value.get("schema") != SCORED_DATASET_SCHEMA:
        raise ValueError(f"unsupported scored dataset: {value.get('schema')}")
    body = {
        key: item
        for key, item in value.items()
        if key not in {"schema", "dataset_sha256"}
    }
    if canonical_sha256(body) != value.get("dataset_sha256"):
        raise ValueError("scored dataset manifest hash mismatch")
    if value.get("selected_splits") != list(ALLOWED_SPLITS) or value.get(
        "test_accessed"
    ):
        raise ValueError("merged selection input must contain train/validation only")
    if value.get("quality_dimensions") != list(QUALITY_DIMENSIONS):
        raise ValueError("scored dataset uses a non-canonical quality profile")
    return value


def merge(args: argparse.Namespace) -> None:
    scored_path = Path(args.scored_manifest).resolve()
    scored = validate_scored_dataset(load_json(scored_path))
    out_root = Path(args.output_root).resolve()
    prompt_owners: dict[str, str] = {}
    prompts: dict[str, dict[str, Any]] = {}
    indexed_records = []
    record_keys = set()
    prompt_seed_keys = set()
    catalog_hash = scored["action_catalog_sha256"]
    for item in scored["records"]:
        path = verify_file(item["path"], item["file_sha256"], label="scored record")
        record = load_json(path)
        validate_scored_record(record)
        if record["record_sha256"] != item["record_sha256"]:
            raise RuntimeError(f"scored record identity mismatch: {path}")
        if canonical_sha256(action_catalog(record)) != catalog_hash:
            raise RuntimeError(f"scored action catalog mismatch: {path}")
        for field in ("split", "trajectory_key", "prompt_sha256", "seed"):
            if record[field] != item[field]:
                raise RuntimeError(f"scored record index mismatch for {field}: {path}")
        prompt_hash = record["prompt_sha256"]
        record_key = (item["shard_id"], record["split"], record["trajectory_key"])
        prompt_seed_key = (item["shard_id"], prompt_hash, int(record["base_seed"]))
        if record_key in record_keys or prompt_seed_key in prompt_seed_keys:
            raise RuntimeError(f"duplicate scored trajectory identity: {path}")
        record_keys.add(record_key)
        prompt_seed_keys.add(prompt_seed_key)
        owner = prompt_owners.get(prompt_hash)
        if owner is not None and owner != item["shard_id"]:
            raise RuntimeError(
                f"prompt overlap across shards: {prompt_hash} in {owner} and {item['shard_id']}"
            )
        prompt_owners[prompt_hash] = item["shard_id"]
        prompt = prompts.setdefault(
            prompt_hash,
            {
                "split": record["split"],
                "prompt": record["prompt"],
                "prompt_sha256": prompt_hash,
                "shard_id": item["shard_id"],
                "local_prompt_id": record["prompt_id"],
                "base_seeds": [],
            },
        )
        expected_prompt = {
            key: prompt[key]
            for key in ("split", "prompt", "shard_id", "local_prompt_id")
        }
        actual_prompt = {
            "split": record["split"],
            "prompt": record["prompt"],
            "shard_id": item["shard_id"],
            "local_prompt_id": record["prompt_id"],
        }
        if expected_prompt != actual_prompt:
            raise RuntimeError(f"inconsistent repeated prompt identity: {path}")
        prompt["base_seeds"].append(int(record["base_seed"]))
        indexed_records.append(
            {
                "shard_id": item["shard_id"],
                "split": record["split"],
                "prompt_sha256": prompt_hash,
                "local_prompt_id": record["prompt_id"],
                "base_seed": record["base_seed"],
                "seed": record["seed"],
                "trajectory_key": record["trajectory_key"],
                "record_path": str(path),
                "record_file_sha256": item["file_sha256"],
                "record_sha256": record["record_sha256"],
            }
        )

    ordered_prompts = sorted(
        prompts.values(),
        key=lambda row: (ALLOWED_SPLITS.index(row["split"]), row["prompt_sha256"]),
    )
    global_ids = {}
    for global_id, prompt in enumerate(ordered_prompts):
        prompt["global_prompt_id"] = global_id
        prompt["base_seeds"] = sorted(set(prompt["base_seeds"]))
        prompt["embedding_relative_path"] = f"t5_embeddings/prompt_{global_id:06d}.npz"
        prompt["embedding_metadata_relative_path"] = (
            f"t5_embeddings/prompt_{global_id:06d}.json"
        )
        global_ids[prompt["prompt_sha256"]] = global_id
    for record in indexed_records:
        record["global_prompt_id"] = global_ids[record["prompt_sha256"]]
    indexed_records.sort(
        key=lambda row: (
            ALLOWED_SPLITS.index(row["split"]),
            row["global_prompt_id"],
            row["base_seed"],
            row["shard_id"],
        )
    )

    prompt_counts = collections.Counter(row["split"] for row in ordered_prompts)
    trajectory_counts = collections.Counter(row["split"] for row in indexed_records)
    expected_prompt_counts = {
        "train": args.expected_train_prompts,
        "validation": args.expected_validation_prompts,
    }
    if dict(prompt_counts) != expected_prompt_counts:
        raise RuntimeError(
            f"prompt coverage mismatch: expected={expected_prompt_counts}, observed={dict(prompt_counts)}"
        )
    expected_trajectories = {
        "train": args.expected_train_prompts,
        "validation": args.expected_validation_prompts * args.validation_seeds,
    }
    if dict(trajectory_counts) != expected_trajectories:
        raise RuntimeError(
            "trajectory coverage mismatch: "
            f"expected={expected_trajectories}, observed={dict(trajectory_counts)}"
        )
    for prompt in ordered_prompts:
        expected_seed_count = 1 if prompt["split"] == "train" else args.validation_seeds
        if len(prompt["base_seeds"]) != expected_seed_count:
            raise RuntimeError(
                f"prompt {prompt['prompt_sha256']} has {len(prompt['base_seeds'])} "
                f"base seeds; expected {expected_seed_count}"
            )

    prompt_text = "".join(f"{row['prompt']}\n" for row in ordered_prompts)
    body = {
        "source_scored_manifest": str(scored_path),
        "source_scored_manifest_file_sha256": sha256_file(scored_path),
        "source_scored_dataset_sha256": scored["dataset_sha256"],
        "selection_stage": "validation_only",
        "selected_splits": list(ALLOWED_SPLITS),
        "test_accessed": False,
        "quality_profile": scored["quality_profile"],
        "quality_dimensions": scored["quality_dimensions"],
        "diagnostic_dimensions": scored["diagnostic_dimensions"],
        "vbench": scored["vbench"],
        "model_root": scored["model_root"],
        "action_catalog": scored["action_catalog"],
        "action_catalog_sha256": catalog_hash,
        "prompt_count": len(ordered_prompts),
        "trajectory_count": len(indexed_records),
        "video_reference_count": len(indexed_records) * 10,
        "prompts_by_split": dict(prompt_counts),
        "trajectories_by_split": dict(trajectory_counts),
        "prompts_file": str((out_root / "prompts.txt").resolve()),
        "prompts_sha256": canonical_sha256([row["prompt"] for row in ordered_prompts]),
        "prompts": ordered_prompts,
        "records": indexed_records,
        "merger_sha256": sha256_file(Path(__file__).resolve()),
    }
    merged = {
        "schema": MERGED_DATASET_SCHEMA,
        "dataset_sha256": canonical_sha256(body),
        **body,
    }
    index_path = out_root / "dataset_index.json"
    prompts_path = out_root / "prompts.txt"
    if index_path.is_file():
        previous = load_json(index_path)
        if previous.get("dataset_sha256") != merged["dataset_sha256"]:
            raise RuntimeError(
                f"refusing to replace a different merged index: {index_path}"
            )
        if (
            not prompts_path.is_file()
            or prompts_path.read_text(encoding="utf-8") != prompt_text
        ):
            raise RuntimeError(f"existing merged prompts differ: {prompts_path}")
    else:
        out_root.mkdir(parents=True, exist_ok=True)
        prompts_path.write_text(prompt_text, encoding="utf-8")
        write_json_atomic(index_path, merged)
    print(
        json.dumps(
            {
                "dataset_index": str(index_path),
                "dataset_sha256": merged["dataset_sha256"],
                "prompts_by_split": merged["prompts_by_split"],
                "trajectories_by_split": merged["trajectories_by_split"],
                "video_reference_count": merged["video_reference_count"],
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--expected-train-prompts", type=int, default=600)
    parser.add_argument("--expected-validation-prompts", type=int, default=200)
    parser.add_argument("--validation-seeds", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    merge(parse_args())
