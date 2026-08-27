#!/usr/bin/env python3
"""Report retained-artifact coverage before an in-place 1,500-prompt resume."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


STEPS = [30, 35, *range(40, 51)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--base-prompt-offset", type=int, default=0)
    return parser.parse_args()


def nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def part_ranges(offset: int, count: int) -> list[tuple[int, int]]:
    base, remainder = divmod(count, 8)
    ranges = []
    current = offset
    for part in range(8):
        size = base + int(part < remainder)
        ranges.append((current, size))
        current += size
    return ranges


def inspect_split(
    root: Path,
    *,
    prompt_offset: int,
    prompt_count: int,
    seeds: list[int],
) -> dict[str, object]:
    counts = {"candidate_videos": 0, "native_videos": 0, "latents": 0, "manifests": 0, "records": 0}
    complete_trajectories = 0
    for part, (offset, count) in enumerate(part_ranges(prompt_offset, prompt_count)):
        part_root = root / "_parts" / f"part_{part:02d}"
        for prompt_id in range(offset, offset + count):
            for base_seed in seeds:
                sample_id = f"{prompt_id:04d}_seed{base_seed + prompt_id}"
                seed_root = part_root / "raw_samples" / f"seed_{base_seed}"
                candidate_ok = 0
                latent_ok = 0
                for step in STEPS:
                    if nonempty(seed_root / "videos" / f"step{step:02d}" / f"{sample_id}_step{step:02d}.mp4"):
                        counts["candidate_videos"] += 1
                        candidate_ok += 1
                    if nonempty(seed_root / "latents" / f"step{step:02d}" / f"{sample_id}_step{step:02d}.pt"):
                        counts["latents"] += 1
                        latent_ok += 1
                native_ok = nonempty(seed_root / "videos" / "native_hr" / f"{sample_id}_native_hr.mp4")
                manifest_ok = nonempty(seed_root / "manifests" / f"{sample_id}.json")
                record_ok = nonempty(part_root / "records" / f"p{prompt_id:06d}_s{base_seed}.json")
                counts["native_videos"] += int(native_ok)
                counts["manifests"] += int(manifest_ok)
                counts["records"] += int(record_ok)
                if candidate_ok == len(STEPS) and latent_ok == len(STEPS) and native_ok and manifest_ok:
                    complete_trajectories += 1
    trajectories = prompt_count * len(seeds)
    expected = {"candidate_videos": trajectories * len(STEPS), "native_videos": trajectories,
                "latents": trajectories * len(STEPS), "manifests": trajectories, "records": trajectories}
    return {"prompt_offset": prompt_offset, "prompt_count": prompt_count, "seeds": seeds,
            "trajectories": trajectories, "complete_trajectories": complete_trajectories,
            "counts": counts, "expected": expected}


def main() -> None:
    args = parse_args()
    root = args.out_root.resolve()
    train = inspect_split(root / "train", prompt_offset=args.base_prompt_offset, prompt_count=1000, seeds=[42])
    evaluation = inspect_split(root / "eval", prompt_offset=args.base_prompt_offset + 1000, prompt_count=500, seeds=[42, 100, 2024])
    totals = {
        key: int(train["counts"][key]) + int(evaluation["counts"][key])
        for key in train["counts"]
    }
    expected_totals = {
        key: int(train["expected"][key]) + int(evaluation["expected"][key])
        for key in train["expected"]
    }
    report = {"schema": "oracle_1500_resume_coverage_v1", "out_root": str(root),
              "train": train, "eval": evaluation, "totals": totals,
              "expected_totals": expected_totals,
              "total_videos": totals["candidate_videos"] + totals["native_videos"],
              "expected_total_videos": expected_totals["candidate_videos"] + expected_totals["native_videos"],
              "remaining_candidate_videos": expected_totals["candidate_videos"] - totals["candidate_videos"],
              "remaining_native_videos": expected_totals["native_videos"] - totals["native_videos"],
              "remaining_latents": expected_totals["latents"] - totals["latents"],
              "complete_trajectories": int(train["complete_trajectories"]) + int(evaluation["complete_trajectories"]),
              "expected_trajectories": int(train["trajectories"]) + int(evaluation["trajectories"])}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
