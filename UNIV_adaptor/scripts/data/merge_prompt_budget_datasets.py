from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from UNIV_adaptor.data_protocol import (  # noqa: E402
    canonical_sha256,
    sha256_file,
    validate_collection_plan,
    validate_trajectory_record,
    write_json_atomic,
)
from UNIV_adaptor.scripts.data.run_prompt_budget_generation import (  # noqa: E402
    load_json,
    validate_manifest,
)


MERGED_SCHEMA = "univ_prompt_budget_merged_dataset_v1"
SPLIT_ORDER = {"train": 0, "validation": 1, "test": 2}


def parse_shard(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--shard must use SHARD_ID=/absolute/root")
    shard_id, root_text = value.split("=", 1)
    shard_id = shard_id.strip()
    if not shard_id or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for char in shard_id):
        raise argparse.ArgumentTypeError(
            "shard id must contain lowercase letters, digits, underscore, or hyphen"
        )
    root = Path(root_text).resolve()
    return shard_id, root


def compatibility_descriptor(
    manifest: dict[str, Any], plan: dict[str, Any]
) -> dict[str, Any]:
    protocol = plan["protocol"]
    cases = {
        case["case_id"]: {
            "kind": case["kind"],
            "model_cls": case["model_cls"],
            "config_sha256": case["config_sha256"],
        }
        for case in manifest["cases"]
    }
    implementation = manifest.get("source", {}).get("implementation_sha256", {})
    return {
        "protocol_sha256": manifest["protocol_sha256"],
        "controller_factorization": protocol["controller_factorization"],
        "observation_mode": protocol["observation_mode"],
        "trajectory_origin": protocol["trajectory_origin"],
        "reference_nfe": protocol["reference_nfe"],
        "target_latent_shape": protocol["target_latent_shape"],
        "transition": protocol["transition"],
        "budget_presets": protocol["budget_presets"],
        "quality_dimensions": protocol["quality_dimensions"],
        "model_root": manifest["model_root"],
        "cases": cases,
        "generation_implementation_sha256": implementation,
    }


def record_path(root: Path, assignment: dict[str, Any]) -> Path:
    return (
        root
        / "records"
        / assignment["split"]
        / f"{assignment['trajectory_key']}.json"
    )


def candidate_plan_fields(candidate: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "budget_id",
        "target_cost_ratio",
        "allocation_source",
        "action_key",
        "requested_action",
        "transition",
        "proxy_compute_density",
        "resolved_schedule",
    )
    return {field: candidate[field] for field in fields}


def verify_artifact(
    artifact: dict[str, Any],
    *,
    verify_hash: bool,
    label: str,
) -> None:
    path = Path(artifact["video_path"])
    if not path.is_file() or path.stat().st_size < 1024:
        raise RuntimeError(f"missing or undersized {label}: {path}")
    if verify_hash and sha256_file(path) != artifact["video_sha256"]:
        raise RuntimeError(f"video hash mismatch for {label}: {path}")
    sidecar_text = str(artifact.get("runtime_sidecar_path", "")).strip()
    if sidecar_text:
        sidecar = Path(sidecar_text)
        if not sidecar.is_file() or sidecar.stat().st_size == 0:
            raise RuntimeError(f"missing runtime sidecar for {label}: {sidecar}")
        if verify_hash and sha256_file(sidecar) != artifact["runtime_sidecar_sha256"]:
            raise RuntimeError(f"sidecar hash mismatch for {label}: {sidecar}")


def merge_datasets(
    shards: list[tuple[str, Path]],
    *,
    splits: list[str],
    require_scores: bool,
    verify_hashes: bool,
) -> dict[str, Any]:
    if len(shards) < 2:
        raise ValueError("at least two dataset shards are required")
    shard_ids = [shard_id for shard_id, _ in shards]
    roots = [str(root) for _, root in shards]
    if len(shard_ids) != len(set(shard_ids)):
        raise ValueError("shard ids must be unique")
    if len(roots) != len(set(roots)):
        raise ValueError("shard roots must be unique")
    selected_splits = set(splits)
    if not selected_splits or selected_splits - set(SPLIT_ORDER):
        raise ValueError("splits must be selected from train, validation, and test")

    shard_payloads = []
    descriptors = []
    all_prompt_owners: dict[str, str] = {}
    prompt_payloads: dict[str, dict[str, Any]] = {}
    indexed_records = []
    for shard_id, root in sorted(shards):
        manifest_path = root / "generation_manifest.json"
        manifest = validate_manifest(load_json(manifest_path))
        plan_path = Path(manifest["plan_path"])
        plan = load_json(plan_path)
        validate_collection_plan(plan)
        if plan["plan_sha256"] != manifest["plan_sha256"]:
            raise RuntimeError(f"plan hash does not match manifest for shard {shard_id}")
        descriptor = compatibility_descriptor(manifest, plan)
        descriptors.append((shard_id, descriptor))

        for assignment in plan["assignments"]:
            prompt_hash = assignment["prompt_sha256"]
            previous_owner = all_prompt_owners.get(prompt_hash)
            if previous_owner is not None and previous_owner != shard_id:
                raise RuntimeError(
                    "prompt overlap across shards: "
                    f"{prompt_hash} belongs to {previous_owner} and {shard_id}"
                )
            all_prompt_owners[prompt_hash] = shard_id

        selected_assignments = [
            row for row in plan["assignments"] if row["split"] in selected_splits
        ]
        shard_record_count = 0
        for assignment in selected_assignments:
            prompt_hash = assignment["prompt_sha256"]
            prompt_payloads.setdefault(
                prompt_hash,
                {
                    "split": assignment["split"],
                    "prompt": assignment["prompt"],
                    "prompt_sha256": prompt_hash,
                    "shard_id": shard_id,
                    "local_prompt_id": assignment["prompt_id"],
                },
            )
            path = record_path(root, assignment)
            if not path.is_file():
                raise RuntimeError(f"missing trajectory record: {path}")
            record = load_json(path)
            validate_trajectory_record(
                record,
                expected_plan_sha256=plan["plan_sha256"],
                require_scores=require_scores,
            )
            expected_identity = {
                field: assignment[field]
                for field in (
                    "trajectory_key",
                    "split",
                    "prompt_id",
                    "prompt",
                    "prompt_sha256",
                    "base_seed",
                    "seed",
                )
            }
            actual_identity = {
                field: record.get(field) for field in expected_identity
            }
            if actual_identity != expected_identity:
                raise RuntimeError(f"record identity does not match plan: {path}")
            expected_candidates = [
                candidate_plan_fields(candidate)
                for candidate in assignment["budget_candidates"]
            ]
            actual_candidates = [
                candidate_plan_fields(candidate)
                for candidate in record["budget_candidates"]
            ]
            if actual_candidates != expected_candidates:
                raise RuntimeError(f"record budget candidates do not match plan: {path}")
            verify_artifact(
                record["native_teacher"],
                verify_hash=verify_hashes,
                label=f"{shard_id}/{record['trajectory_key']}/native",
            )
            for candidate in record["budget_candidates"]:
                verify_artifact(
                    candidate,
                    verify_hash=verify_hashes,
                    label=(
                        f"{shard_id}/{record['trajectory_key']}/"
                        f"{candidate['budget_id']}"
                    ),
                )
            record_uid = hashlib.sha256(
                f"{shard_id}\0{plan['plan_sha256']}\0{record['trajectory_key']}".encode(
                    "utf-8"
                )
            ).hexdigest()
            indexed_records.append(
                {
                    "record_uid": record_uid,
                    "shard_id": shard_id,
                    "split": record["split"],
                    "prompt_sha256": prompt_hash,
                    "local_prompt_id": record["prompt_id"],
                    "seed": record["seed"],
                    "trajectory_key": record["trajectory_key"],
                    "record_path": str(path.resolve()),
                    "record_relative_path": str(path.relative_to(root)),
                    "generation_status": record.get("generation_status", "unknown"),
                }
            )
            shard_record_count += 1
        shard_payloads.append(
            {
                "shard_id": shard_id,
                "root": str(root),
                "generation_manifest": str(manifest_path),
                "generation_manifest_sha256": manifest["manifest_sha256"],
                "plan": str(plan_path),
                "plan_sha256": plan["plan_sha256"],
                "prompts_file": manifest["prompts_file"],
                "prompts_file_sha256": manifest["prompts_file_sha256"],
                "record_count": shard_record_count,
            }
        )

    reference_id, reference_descriptor = descriptors[0]
    for shard_id, descriptor in descriptors[1:]:
        if canonical_sha256(descriptor) != canonical_sha256(reference_descriptor):
            differences = sorted(
                key
                for key in set(reference_descriptor) | set(descriptor)
                if reference_descriptor.get(key) != descriptor.get(key)
            )
            raise RuntimeError(
                f"shard {shard_id} is incompatible with {reference_id}: {differences}"
            )

    ordered_prompts = sorted(
        prompt_payloads.values(),
        key=lambda value: (
            SPLIT_ORDER[value["split"]],
            value["prompt_sha256"],
        ),
    )
    global_ids = {
        prompt["prompt_sha256"]: index for index, prompt in enumerate(ordered_prompts)
    }
    for prompt in ordered_prompts:
        prompt["global_prompt_id"] = global_ids[prompt["prompt_sha256"]]
    for record in indexed_records:
        record["global_prompt_id"] = global_ids[record["prompt_sha256"]]
    indexed_records.sort(
        key=lambda value: (
            SPLIT_ORDER[value["split"]],
            value["global_prompt_id"],
            value["seed"],
            value["shard_id"],
        )
    )
    split_records = collections.Counter(row["split"] for row in indexed_records)
    split_prompts = collections.Counter(row["split"] for row in ordered_prompts)
    body = {
        "compatibility_sha256": canonical_sha256(reference_descriptor),
        "compatibility": reference_descriptor,
        "selected_splits": sorted(selected_splits, key=SPLIT_ORDER.__getitem__),
        "require_scores": require_scores,
        "artifact_hashes_verified": verify_hashes,
        "shards": shard_payloads,
        "prompt_count": len(ordered_prompts),
        "trajectory_count": len(indexed_records),
        "video_count": len(indexed_records) * 6,
        "prompts_by_split": dict(sorted(split_prompts.items())),
        "trajectories_by_split": dict(sorted(split_records.items())),
        "prompts": ordered_prompts,
        "records": indexed_records,
    }
    return {"schema": MERGED_SCHEMA, "dataset_sha256": canonical_sha256(body), **body}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and zero-copy merge UNIV prompt-budget dataset shards."
    )
    parser.add_argument("--shard", action="append", required=True, type=parse_shard)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation", "test"],
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-scores", action="store_true")
    parser.add_argument("--verify-artifact-hashes", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merged = merge_datasets(
        args.shard,
        splits=args.splits,
        require_scores=args.require_scores,
        verify_hashes=args.verify_artifact_hashes,
    )
    output = Path(args.output).resolve()
    if output.is_file():
        previous = load_json(output)
        if previous.get("dataset_sha256") != merged["dataset_sha256"]:
            raise RuntimeError(f"refusing to replace a different merged index: {output}")
    else:
        write_json_atomic(output, merged)
    summary = {
        key: merged[key]
        for key in (
            "dataset_sha256",
            "selected_splits",
            "prompt_count",
            "trajectory_count",
            "video_count",
            "prompts_by_split",
            "trajectories_by_split",
        )
    }
    print(json.dumps(summary, indent=2))
    print(f"Merged dataset index: {output}")


if __name__ == "__main__":
    main()
