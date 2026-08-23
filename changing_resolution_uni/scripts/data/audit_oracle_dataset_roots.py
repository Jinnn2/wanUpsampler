#!/usr/bin/env python3
"""Read-only coverage and overlap audit for oracle dataset roots."""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASETS = [
    "oracle_dataset_1k",
    "oracle_dataset_2k",
    "oracle_dataset_500_1000",
]
RECORD_NAME = re.compile(r"^p(?P<prompt>\d+)_s(?P<seed>-?\d+)\.json$")


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
    invalid_record_names = []

    for path in record_files:
        match = RECORD_NAME.fullmatch(path.name)
        if match is None:
            invalid_record_names.append(str(path))
            continue
        prompt_id = int(match.group("prompt"))
        seed = int(match.group("seed"))
        prompt_seeds[prompt_id].add(seed)
        keys.add((prompt_id, seed))

    prompts = set(prompt_seeds)
    incomplete = {
        prompt_id: sorted(seeds)
        for prompt_id, seeds in prompt_seeds.items()
        if seeds != expected_seeds
    }

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
        "unique_keys": len(keys),
        "duplicate_key_files": len(record_files) - len(keys),
        "unique_prompts": len(prompts),
        "prompt_range": compress_ranges(prompts),
        "prompts": prompts,
        "keys": keys,
        "incomplete_seed_prompts": len(incomplete),
        "incomplete_seed_examples": list(sorted(incomplete.items()))[:10],
        "invalid_record_names": invalid_record_names[:10],
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
                    f"duplicate_key_files={report['duplicate_key_files']}",
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
        if report["incomplete_seed_examples"]:
            print(f"seed_mismatch_examples={report['incomplete_seed_examples']}")
        if report["invalid_record_names"]:
            print(f"invalid_record_names={report['invalid_record_names']}")

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
