#!/usr/bin/env python3
"""Plan exact missing oracle trajectories for an exclusive H100 tail run."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any


STEPS = [30, 35, *range(40, 51)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--prompts-file", type=Path, required=True)
    parser.add_argument("--plan-out", type=Path, required=True)
    parser.add_argument("--base-prompt-offset", type=int, default=0)
    parser.add_argument("--micro-batch-prompts", type=int, default=2)
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def nonempty(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def read_json(path: Path) -> dict[str, Any] | None:
    if not nonempty(path):
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def part_ranges(offset: int, count: int) -> list[tuple[int, int, int]]:
    base, remainder = divmod(count, 8)
    ranges = []
    current = offset
    for part in range(8):
        size = base + int(part < remainder)
        ranges.append((part, current, size))
        current += size
    return ranges


def group_consecutive(values: list[int], maximum: int) -> list[list[int]]:
    groups: list[list[int]] = []
    current: list[int] = []
    for value in values:
        if current and (value != current[-1] + 1 or len(current) >= maximum):
            groups.append(current)
            current = []
        current.append(value)
    if current:
        groups.append(current)
    return groups


def inspect_trajectory(
    part_root: Path,
    prompt_id: int,
    base_seed: int,
) -> tuple[bool, Counter[str]]:
    reasons: Counter[str] = Counter()
    sample_id = f"{prompt_id:04d}_seed{base_seed + prompt_id}"
    seed_root = part_root / "raw_samples" / f"seed_{base_seed}"
    manifest = read_json(seed_root / "manifests" / f"{sample_id}.json")
    if manifest is None:
        reasons["manifest_missing_or_invalid"] += 1
        manifest_steps: set[int] = set()
        native_manifest = False
    else:
        manifest_steps = {
            int(row["candidate_step"])
            for row in manifest.get("branches", [])
            if isinstance(row, dict) and "candidate_step" in row
        }
        native_manifest = isinstance(manifest.get("native_hr"), dict)
    for step in STEPS:
        if step not in manifest_steps:
            reasons["manifest_candidate_rows"] += 1
        video = seed_root / "videos" / f"step{step:02d}" / f"{sample_id}_step{step:02d}.mp4"
        latent = seed_root / "latents" / f"step{step:02d}" / f"{sample_id}_step{step:02d}.pt"
        if not nonempty(video):
            reasons["candidate_videos"] += 1
        if not nonempty(latent):
            reasons["latents"] += 1
    native = seed_root / "videos" / "native_hr" / f"{sample_id}_native_hr.mp4"
    if not native_manifest:
        reasons["manifest_native_row"] += 1
    if not nonempty(native):
        reasons["native_videos"] += 1
    return not reasons, reasons


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    if args.base_prompt_offset < 0:
        raise SystemExit("base-prompt-offset must be non-negative")
    if args.micro_batch_prompts < 1:
        raise SystemExit("micro-batch-prompts must be positive")
    out_root = args.out_root.resolve()
    prompts_file = args.prompts_file.resolve()
    if not out_root.is_dir():
        raise SystemExit(f"output root does not exist: {out_root}")
    if not prompts_file.is_file():
        raise SystemExit(f"prompts file does not exist: {prompts_file}")

    split_specs = (
        ("train", args.base_prompt_offset, 1000, [42]),
        ("eval", args.base_prompt_offset + 1000, 500, [42, 100, 2024]),
    )
    tasks: list[dict[str, Any]] = []
    missing_reasons: Counter[str] = Counter()
    missing_trajectories = 0
    complete_trajectories = 0

    for split, split_offset, split_count, seeds in split_specs:
        for part, canonical_offset, canonical_count in part_ranges(split_offset, split_count):
            part_root = out_root / split / "_parts" / f"part_{part:02d}"
            for base_seed in seeds:
                missing_ids: list[int] = []
                for prompt_id in range(canonical_offset, canonical_offset + canonical_count):
                    complete, reasons = inspect_trajectory(part_root, prompt_id, base_seed)
                    if complete:
                        complete_trajectories += 1
                    else:
                        missing_trajectories += 1
                        missing_ids.append(prompt_id)
                        missing_reasons.update(reasons)
                for group in group_consecutive(missing_ids, args.micro_batch_prompts):
                    prompt_offset = group[0]
                    limit = len(group)
                    task_id = (
                        f"{split}_p{part:02d}_s{base_seed}_"
                        f"o{prompt_offset:04d}_n{limit:02d}"
                    )
                    tasks.append(
                        {
                            "task_id": task_id,
                            "split": split,
                            "part": part,
                            "base_seed": base_seed,
                            "prompt_offset": prompt_offset,
                            "limit": limit,
                            "canonical_prompt_offset": canonical_offset,
                            "canonical_prompt_count": canonical_count,
                        }
                    )

    expected_trajectories = 2500
    payload = {
        "schema": "oracle_1500_h100_tail_plan_v1",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "out_root": str(out_root),
        "prompts_file": str(prompts_file),
        "prompts_sha256": hashlib.sha256(prompts_file.read_bytes()).hexdigest(),
        "micro_batch_prompts": args.micro_batch_prompts,
        "candidate_steps": STEPS,
        "expected_trajectories": expected_trajectories,
        "complete_trajectories": complete_trajectories,
        "missing_trajectories": missing_trajectories,
        "missing_artifact_counts": dict(sorted(missing_reasons.items())),
        "task_count": len(tasks),
        "tasks": tasks,
    }
    if complete_trajectories + missing_trajectories != expected_trajectories:
        raise SystemExit("internal trajectory accounting mismatch")
    write_json_atomic(args.plan_out.resolve(), payload)
    print(json.dumps({key: payload[key] for key in (
        "complete_trajectories", "missing_trajectories", "missing_artifact_counts", "task_count"
    )}, ensure_ascii=False, indent=2))
    if args.require_complete and tasks:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
