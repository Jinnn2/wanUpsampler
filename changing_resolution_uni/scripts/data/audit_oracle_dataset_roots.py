#!/usr/bin/env python3
"""Read-only coverage and overlap audit for oracle dataset roots."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
AUDIT_VERSION = 2
DEFAULT_DATASETS = [
    "oracle_dataset_1k",
    "oracle_dataset_2k",
    "oracle_dataset_500_1000",
]
RECORD_NAME = re.compile(r"^p(?P<prompt>\d+)_s(?P<seed>-?\d+)\.json$")
FORMAL_STEPS = {30, 35, *range(40, 51)}
QUALITY5_DIMENSIONS = {
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "aesthetic_quality",
    "imaging_quality",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit oracle dataset shards, prompt coverage, and overlaps."
    )
    parser.add_argument(
        "--data-root",
        default=str(REPO_ROOT / "data" / "changing_resolution_uni"),
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--expected-total-prompts", type=int, default=2000)
    parser.add_argument("--expected-seeds", type=int, nargs="+", default=[42, 100, 2024])
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args()


def compress_ranges(values: Iterable[int]) -> str:
    ordered = sorted(set(values))
    if not ordered:
        return "none"
    ranges: list[str] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def inspect_dataset(root: Path, expected_seeds: set[int]) -> dict:
    record_files = sorted(root.rglob("records/p*_s*.json")) if root.is_dir() else []
    prompt_seeds: dict[int, set[int]] = defaultdict(set)
    keys: set[tuple[int, int]] = set()
    paths_by_key: dict[tuple[int, int], list[Path]] = defaultdict(list)
    invalid_record_names = []
    parse_errors = []
    scored_keys: set[tuple[int, int]] = set()
    dimension_complete_keys: set[tuple[int, int]] = set()
    strict_trainable_keys: set[tuple[int, int]] = set()
    router_ready_keys: set[tuple[int, int]] = set()
    fixed_seed_keys: set[tuple[int, int]] = set()
    offset_seed_keys: set[tuple[int, int]] = set()

    for path in record_files:
        match = RECORD_NAME.fullmatch(path.name)
        if match is None:
            invalid_record_names.append(str(path))
            continue
        prompt_id = int(match.group("prompt"))
        seed = int(match.group("seed"))
        prompt_seeds[prompt_id].add(seed)
        key = (prompt_id, seed)
        keys.add(key)
        paths_by_key[key].append(path)
        if seed in expected_seeds:
            fixed_seed_keys.add(key)
        if seed - prompt_id in expected_seeds:
            offset_seed_keys.add(key)

        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            candidates = record.get("candidates", [])
            by_step = {
                int(candidate["step"]): candidate
                for candidate in candidates
                if isinstance(candidate, dict) and "step" in candidate
            }
            candidate_scored = (
                set(by_step) == FORMAL_STEPS
                and all(float(by_step[step].get("vbench5", 0.0)) > 0.1 for step in FORMAL_STEPS)
                and all(float(by_step[step].get("latency_seconds", 0.0)) > 0.0 for step in FORMAL_STEPS)
            )
            dimension_complete = candidate_scored and all(
                QUALITY5_DIMENSIONS.issubset(
                    set(by_step[step].get("dimensions", {}))
                )
                for step in FORMAL_STEPS
            )
            native_complete = (
                float(record.get("native_vbench5", 0.0)) > 0.1
                and float(record.get("native_latency_seconds", 0.0)) > 0.0
                and QUALITY5_DIMENSIONS.issubset(
                    set(record.get("native_dimensions", {}))
                )
            )
            native_scalar_complete = (
                float(record.get("native_vbench5", 0.0)) > 0.1
                and float(record.get("native_latency_seconds", 0.0)) > 0.0
            )
            if candidate_scored:
                scored_keys.add(key)
            if dimension_complete:
                dimension_complete_keys.add(key)
            if dimension_complete and native_complete:
                strict_trainable_keys.add(key)
            if candidate_scored and native_scalar_complete:
                router_ready_keys.add(key)
        except Exception as exc:
            parse_errors.append(f"{path}: {exc}")

    prompts = set(prompt_seeds)
    incomplete = {
        prompt_id: sorted(seeds)
        for prompt_id, seeds in prompt_seeds.items()
        if seeds != expected_seeds
    }

    def fixed_complete_prompt_count(candidate_keys: set[tuple[int, int]]) -> int:
        return sum(
            all((prompt_id, seed) in candidate_keys for seed in expected_seeds)
            for prompt_id in prompts
        )

    def offset_complete_prompt_count(candidate_keys: set[tuple[int, int]]) -> int:
        return sum(
            all(
                (prompt_id, base_seed + prompt_id) in candidate_keys
                for base_seed in expected_seeds
            )
            for prompt_id in prompts
        )

    duplicate_keys = {key: paths for key, paths in paths_by_key.items() if len(paths) > 1}
    identical_duplicate_keys = 0
    conflicting_duplicate_keys = 0
    for paths in duplicate_keys.values():
        hashes = {
            hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths
        }
        if len(hashes) == 1:
            identical_duplicate_keys += 1
        else:
            conflicting_duplicate_keys += 1

    step_videos = 0
    native_videos = 0
    manifest_files = 0
    if root.is_dir():
        manifest_files = sum(1 for _ in root.rglob("manifests/*.json"))
        for path in root.rglob("*.mp4"):
            if path.parent.name.startswith("step"):
                step_videos += 1
            elif path.parent.name == "native_hr":
                native_videos += 1

    root_manifest_path = root / "dataset_manifest.json"
    root_manifest = None
    if root_manifest_path.is_file():
        try:
            payload = json.loads(root_manifest_path.read_text(encoding="utf-8"))
            root_manifest = {
                key: payload.get(key)
                for key in (
                    "schema",
                    "total_prompts_found",
                    "expected_prompts",
                    "total_trajectories",
                    "expected_trajectories",
                    "expected_seeds",
                    "is_complete",
                )
                if key in payload
            }
        except Exception as exc:
            root_manifest = {"parse_error": str(exc)}

    return {
        "path": str(root),
        "exists": root.is_dir(),
        "record_files": len(record_files),
        "root_record_files": sum(
            1 for path in record_files if path.parent == root / "records"
        ),
        "part_record_files": sum(
            1 for path in record_files if path.parent != root / "records"
        ),
        "unique_keys": len(keys),
        "duplicate_key_files": len(record_files) - len(keys),
        "duplicate_keys": len(duplicate_keys),
        "identical_duplicate_keys": identical_duplicate_keys,
        "conflicting_duplicate_keys": conflicting_duplicate_keys,
        "unique_prompts": len(prompts),
        "prompt_range": compress_ranges(prompts),
        "prompts": prompts,
        "keys": keys,
        "incomplete_seed_prompts": len(incomplete),
        "incomplete_seed_examples": list(sorted(incomplete.items()))[:10],
        "invalid_record_names": invalid_record_names[:10],
        "parse_errors": parse_errors[:10],
        "fixed_seed_keys": len(fixed_seed_keys),
        "offset_seed_keys": len(offset_seed_keys),
        "fixed_seed_complete_prompts": fixed_complete_prompt_count(fixed_seed_keys),
        "offset_seed_complete_prompts": offset_complete_prompt_count(offset_seed_keys),
        "scored_keys": len(scored_keys),
        "dimension_complete_keys": len(dimension_complete_keys),
        "strict_trainable_keys": len(strict_trainable_keys),
        "router_ready_keys": len(router_ready_keys),
        "scored_fixed_seed_complete_prompts": fixed_complete_prompt_count(
            scored_keys & fixed_seed_keys
        ),
        "scored_offset_seed_complete_prompts": offset_complete_prompt_count(
            scored_keys & offset_seed_keys
        ),
        "router_ready_offset_prompts": offset_complete_prompt_count(
            router_ready_keys & offset_seed_keys
        ),
        "dimension_fixed_seed_complete_prompts": fixed_complete_prompt_count(
            dimension_complete_keys & fixed_seed_keys
        ),
        "dimension_offset_seed_complete_prompts": offset_complete_prompt_count(
            dimension_complete_keys & offset_seed_keys
        ),
        "strict_fixed_seed_complete_prompts": fixed_complete_prompt_count(
            strict_trainable_keys & fixed_seed_keys
        ),
        "strict_offset_seed_complete_prompts": offset_complete_prompt_count(
            strict_trainable_keys & offset_seed_keys
        ),
        "t5_embeddings": (
            sum(1 for _ in (root / "t5_embeddings").glob("prompt_*.npz"))
            if root.is_dir()
            else 0
        ),
        "sample_manifests": manifest_files,
        "step_videos": step_videos,
        "native_videos": native_videos,
        "root_manifest": root_manifest,
    }


def json_safe(info: dict) -> dict:
    return {
        key: sorted(value) if isinstance(value, set) else value
        for key, value in info.items()
    }


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    expected_seeds = {int(seed) for seed in args.expected_seeds}
    reports = {
        name: inspect_dataset(data_root / name, expected_seeds)
        for name in args.datasets
    }

    overlaps = []
    for left, right in combinations(args.datasets, 2):
        prompt_overlap = reports[left]["prompts"] & reports[right]["prompts"]
        key_overlap = reports[left]["keys"] & reports[right]["keys"]
        overlaps.append(
            {
                "left": left,
                "right": right,
                "prompt_overlap": len(prompt_overlap),
                "prompt_overlap_range": compress_ranges(prompt_overlap),
                "key_overlap": len(key_overlap),
            }
        )

    union_prompts = set().union(*(report["prompts"] for report in reports.values()))
    expected_prompts = set(range(args.expected_total_prompts))
    union = {
        "unique_prompts": len(union_prompts),
        "prompt_range": compress_ranges(union_prompts),
        "missing_expected": compress_ranges(expected_prompts - union_prompts),
        "outside_expected": compress_ranges(union_prompts - expected_prompts),
    }

    if args.json:
        print(
            json.dumps(
                {
                    "audit_version": AUDIT_VERSION,
                    "data_root": str(data_root),
                    "datasets": {name: json_safe(report) for name, report in reports.items()},
                    "overlaps": overlaps,
                    "union": union,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    print(f"audit_version={AUDIT_VERSION}")
    print(f"data_root={data_root}")
    for name, report in reports.items():
        print(f"\n[{name}]")
        print(
            " ".join(
                [
                    f"exists={report['exists']}",
                    f"prompts={report['unique_prompts']}",
                    f"range={report['prompt_range']}",
                    f"keys={report['unique_keys']}",
                    f"record_files={report['record_files']}",
                    f"root_records={report['root_record_files']}",
                    f"part_records={report['part_record_files']}",
                ]
            )
        )
        print(
            " ".join(
                [
                    f"fixed_seed_keys={report['fixed_seed_keys']}",
                    f"offset_seed_keys={report['offset_seed_keys']}",
                    f"fixed_complete_prompts={report['fixed_seed_complete_prompts']}",
                    f"offset_complete_prompts={report['offset_seed_complete_prompts']}",
                ]
            )
        )
        print(
            " ".join(
                [
                    f"scored_keys={report['scored_keys']}",
                    f"router_ready_keys={report['router_ready_keys']}",
                    f"dimension_keys={report['dimension_complete_keys']}",
                    f"strict_keys={report['strict_trainable_keys']}",
                    f"scored_fixed_prompts={report['scored_fixed_seed_complete_prompts']}",
                    f"scored_offset_prompts={report['scored_offset_seed_complete_prompts']}",
                    f"router_ready_offset_prompts={report['router_ready_offset_prompts']}",
                    f"dimension_fixed_prompts={report['dimension_fixed_seed_complete_prompts']}",
                    f"dimension_offset_prompts={report['dimension_offset_seed_complete_prompts']}",
                    f"strict_fixed_prompts={report['strict_fixed_seed_complete_prompts']}",
                    f"strict_offset_prompts={report['strict_offset_seed_complete_prompts']}",
                ]
            )
        )
        print(
            " ".join(
                [
                    f"t5={report['t5_embeddings']}",
                    f"manifests={report['sample_manifests']}",
                    f"step_videos={report['step_videos']}",
                    f"native_videos={report['native_videos']}",
                    f"incomplete_seed_prompts={report['incomplete_seed_prompts']}",
                ]
            )
        )
        print(f"root_manifest={report['root_manifest']}")
        print(
            f"duplicate_keys={report['duplicate_keys']} "
            f"identical={report['identical_duplicate_keys']} "
            f"conflicting={report['conflicting_duplicate_keys']}"
        )
        if report["incomplete_seed_examples"]:
            print(f"seed_mismatch_examples={report['incomplete_seed_examples']}")
        if report["invalid_record_names"]:
            print(f"invalid_record_names={report['invalid_record_names']}")
        if report["parse_errors"]:
            print(f"parse_errors={report['parse_errors']}")

    print("\n[overlaps]")
    for row in overlaps:
        print(
            f"{row['left']} <-> {row['right']}: "
            f"prompts={row['prompt_overlap']} "
            f"range={row['prompt_overlap_range']} keys={row['key_overlap']}"
        )

    print("\n[union]")
    print(
        f"prompts={union['unique_prompts']} range={union['prompt_range']} "
        f"missing={union['missing_expected']} outside={union['outside_expected']}"
    )


if __name__ == "__main__":
    main()
