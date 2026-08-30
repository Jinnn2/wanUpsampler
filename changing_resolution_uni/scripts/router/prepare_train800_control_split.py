#!/usr/bin/env python3
"""Derive a deterministic 800/200 control split from prepared train states."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


DATASET_SCHEMA = "variable_lambda_online_state_dataset_v1"
DERIVATION_SCHEMA = "train800_control200_hash_split_v1"
DEFAULT_SALT = "train800_control200_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--validation-count", type=int, default=200)
    parser.add_argument("--split-salt", default=DEFAULT_SALT)
    parser.add_argument("--expected-source-prompts", type=int, default=1000)
    parser.add_argument("--expected-base-seed", type=int, default=42)
    args = parser.parse_args()
    if args.validation_count < 1:
        parser.error("validation-count must be positive")
    if args.expected_source_prompts < 2:
        parser.error("expected-source-prompts must be at least two")
    if args.validation_count >= args.expected_source_prompts:
        parser.error("validation-count must be smaller than expected-source-prompts")
    if not str(args.split_salt).strip():
        parser.error("split-salt must not be empty")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            rows.append(payload)
    return rows


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def normalized_prompt(text: str) -> str:
    return " ".join(str(text).casefold().split())


def split_key(prompt_id: int, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{prompt_id}".encode("utf-8")).hexdigest()


def resolve_source_path(source_dir: Path, value: Any, field: str) -> Path:
    raw = Path(str(value))
    path = raw if raw.is_absolute() else source_dir / raw
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {field}: {path}")
    return path


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dataset_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite derived dataset: {output_dir}")

    source_manifest_path = source_dir / "dataset_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("schema") != DATASET_SCHEMA:
        raise ValueError(f"Unexpected source dataset schema: {source_manifest_path}")
    if source_manifest.get("is_complete") is not True:
        raise ValueError("Source state dataset is incomplete")
    if source_manifest.get("test_accessed") is not False:
        raise ValueError("Source state dataset accessed test")
    if set(source_manifest.get("selected_splits", [])) != {"train", "validation"}:
        raise ValueError("Source dataset must contain train and validation only")

    source_train = source_manifest.get("splits", {}).get("train", {})
    source_index_path = source_dir / str(source_train.get("index_file", ""))
    source_index_sha256 = sha256_file(source_index_path)
    if source_index_sha256 != source_train.get("index_sha256"):
        raise ValueError(f"Source train index SHA256 mismatch: {source_index_path}")
    rows = read_jsonl(source_index_path)
    if len(rows) != args.expected_source_prompts:
        raise ValueError(
            f"Expected {args.expected_source_prompts} source trajectories, got {len(rows)}"
        )

    prompt_ids = [int(row["prompt_id"]) for row in rows]
    if len(set(prompt_ids)) != len(prompt_ids):
        raise ValueError("Source train index has duplicate prompt IDs")
    if set(prompt_ids) != set(range(args.expected_source_prompts)):
        raise ValueError("Source train prompt IDs are not the expected 0-based range")
    base_seeds = {int(row["seed"]) - int(row["prompt_id"]) for row in rows}
    if base_seeds != {args.expected_base_seed}:
        raise ValueError(f"Unexpected source base seeds: {sorted(base_seeds)}")

    normalized = [normalized_prompt(str(row.get("prompt_text", ""))) for row in rows]
    if any(not value for value in normalized):
        raise ValueError("Source train index contains an empty prompt")
    duplicates = [text for text, count in Counter(normalized).items() if count > 1]
    if duplicates:
        raise ValueError(
            "Normalized prompt text is not unique; refusing a prompt-leaking split: "
            f"{duplicates[:5]}"
        )

    validation_ids = {
        prompt_id
        for prompt_id in sorted(
            prompt_ids, key=lambda value: (split_key(value, args.split_salt), value)
        )[: args.validation_count]
    }
    train_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    feature_inventory: list[dict[str, Any]] = []
    for source_row in sorted(rows, key=lambda item: int(item["prompt_id"])):
        row = dict(source_row)
        prompt_id = int(row["prompt_id"])
        split = "validation" if prompt_id in validation_ids else "train"
        feature_path = resolve_source_path(
            source_dir, row.get("feature_file"), "state feature"
        )
        t5_path = resolve_source_path(
            source_dir, row.get("t5_embedding_path"), "T5 embedding"
        )
        row["split"] = split
        row["feature_file"] = str(feature_path)
        row["feature_sha256"] = sha256_file(feature_path)
        row["t5_embedding_path"] = str(t5_path)
        row["t5_embedding_sha256"] = sha256_file(t5_path)
        feature_inventory.append(
            {
                "prompt_id": prompt_id,
                "feature_sha256": row["feature_sha256"],
                "t5_embedding_sha256": row["t5_embedding_sha256"],
            }
        )
        (validation_rows if split == "validation" else train_rows).append(row)

    expected_train_count = args.expected_source_prompts - args.validation_count
    if (
        len(train_rows) != expected_train_count
        or len(validation_rows) != args.validation_count
    ):
        raise RuntimeError(
            "Derived split sizes do not match the requested 800/200 contract"
        )
    if {int(row["prompt_id"]) for row in train_rows} & validation_ids:
        raise RuntimeError("Derived train and validation prompt IDs overlap")

    output_dir.mkdir(parents=True)
    train_index = output_dir / "train_trajectories.jsonl"
    validation_index = output_dir / "validation_trajectories.jsonl"
    write_jsonl_atomic(train_index, train_rows)
    write_jsonl_atomic(validation_index, validation_rows)

    train_ids = [int(row["prompt_id"]) for row in train_rows]
    validation_ids_sorted = [int(row["prompt_id"]) for row in validation_rows]
    manifest = {
        **{
            key: source_manifest[key]
            for key in (
                "schema",
                "generation_root",
                "generation_plan",
                "generation_plan_sha256",
                "candidate_steps",
                "quality_dimensions",
                "feature_names",
                "feature_groups",
                "feature_count",
                "lambda_dependent_features",
                "latency_profile",
            )
        },
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "selected_splits": ["train", "validation"],
        "test_accessed": False,
        "splits": {
            "train": {
                "prompt_count": len(train_ids),
                "trajectory_count": len(train_rows),
                "base_seeds": [args.expected_base_seed],
                "physical_dataset": "train",
                "index_file": train_index.name,
                "index_sha256": sha256_file(train_index),
                "prompt_ids": train_ids,
                "prompt_ids_sha256": sha256_json(train_ids),
            },
            "validation": {
                "prompt_count": len(validation_ids_sorted),
                "trajectory_count": len(validation_rows),
                "base_seeds": [args.expected_base_seed],
                "physical_dataset": "train",
                "index_file": validation_index.name,
                "index_sha256": sha256_file(validation_index),
                "prompt_ids": validation_ids_sorted,
                "prompt_ids_sha256": sha256_json(validation_ids_sorted),
            },
        },
        "scored_sources": {"train": source_manifest["scored_sources"]["train"]},
        "derivation": {
            "schema": DERIVATION_SCHEMA,
            "purpose": "in_distribution_train800_control_validation200",
            "split_algorithm": "ascending_sha256_of_salt_colon_prompt_id",
            "split_salt": args.split_salt,
            "validation_take": "first_n",
            "source_dataset_manifest": str(source_manifest_path),
            "source_dataset_manifest_sha256": sha256_file(source_manifest_path),
            "source_train_index": str(source_index_path),
            "source_train_index_sha256": source_index_sha256,
            "source_prompt_count": len(rows),
            "source_base_seed": args.expected_base_seed,
            "normalized_prompt_text_unique": True,
            "source_validation_index_accessed": False,
            "source_test_accessed": False,
            "feature_inventory_sha256": sha256_json(feature_inventory),
            "latency_profile_role": "frozen_external_h100_hardware_calibration",
        },
        "is_complete": True,
    }
    manifest_path = output_dir / "dataset_manifest.json"
    write_json_atomic(manifest_path, manifest)
    print(
        json.dumps(
            {
                "dataset": str(output_dir),
                "manifest_sha256": sha256_file(manifest_path),
                "train_prompts": len(train_ids),
                "validation_prompts": len(validation_ids_sorted),
                "validation_prompt_ids": validation_ids_sorted,
                "test_accessed": False,
                "latency_profile_sha256": manifest["latency_profile"]["sha256"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
